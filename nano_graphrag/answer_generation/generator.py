"""
答案生成模块

提供答案生成相关功能，包括LLM调用、提示生成等
"""

import os
import time
import json
import hashlib
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field

import aiohttp
from openai import OpenAI, AsyncOpenAI

from .prompts import PromptTemplate, BasicPromptTemplate, ConfidenceAwarePrompt
from ..feedback import FeedbackProcessor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("answer_generator")


@dataclass
class AnswerGeneratorConfig:
    """答案生成器配置"""
    
    # LLM配置
    model_name: str = "qwen-turbo"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 30.0
    
    # 生成配置
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1500
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 缓存配置
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    # 提示模板配置
    prompt_template: Optional[PromptTemplate] = None
    system_prompt: Optional[str] = None
    
    # 高级配置
    batch_size: int = 1
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    # API类型
    api_type: str = "dashscope"  # 可选值: "openai", "dashscope", "siliconflow"


class AnswerGenerator:
    """答案生成器，处理LLM交互"""
    
    def __init__(
        self,
        config: Optional[AnswerGeneratorConfig] = None,
        feedback_processor: Optional[FeedbackProcessor] = None
    ):
        """
        初始化答案生成器
        
        Args:
            config: 答案生成器配置，如果为None则使用默认配置
            feedback_processor: 反馈处理器，用于收集和处理反馈
        """
        self.config = config or AnswerGeneratorConfig()
        self.feedback_processor = feedback_processor
        
        # 初始化缓存
        self._init_cache()
        
        # 初始化提示模板
        self.prompt_template = self.config.prompt_template or BasicPromptTemplate()
        
        # 初始化客户端
        self._init_clients()
        
        # 计数器
        self.call_count = 0
        self.cache_hits = 0
        
    def _init_cache(self):
        """初始化缓存"""
        self.cache = {}
        self.cache_dir = self.config.cache_dir
        
        if self.config.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _init_clients(self):
        """初始化API客户端"""
        # 获取API密钥，优先使用配置中的，其次使用环境变量
        api_key = self.config.api_key
        
        if api_key is None:
            if self.config.api_type == "dashscope":
                api_key = os.environ.get("DASHSCOPE_API_KEY")
            elif self.config.api_type == "siliconflow":
                api_key = os.environ.get("SILICONFLOW_API_KEY")
            else:  # openai
                api_key = os.environ.get("OPENAI_API_KEY")
        
        # 获取基础URL，支持兼容模式的API
        base_url = self.config.base_url
        
        if base_url is None:
            if self.config.api_type == "dashscope":
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            elif self.config.api_type == "siliconflow":
                base_url = "https://api.siliconflow.cn/v1"
        
        # 初始化客户端参数
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        # 初始化同步和异步客户端
        try:
            self.client = OpenAI(**client_kwargs)
            self.async_client = AsyncOpenAI(**client_kwargs)
            logger.info(f"初始化{self.config.model_name}客户端完成 (API类型: {self.config.api_type})")
        except Exception as e:
            logger.error(f"初始化客户端失败: {str(e)}")
            raise
    
    def _compute_cache_key(self, query: str, context: str, **kwargs) -> str:
        """
        计算缓存键
        
        Args:
            query: 用户查询
            context: 上下文信息
            **kwargs: 其他参数
            
        Returns:
            缓存键
        """
        # 构建缓存键的输入
        cache_input = {
            "query": query,
            "context_hash": hashlib.md5(context.encode()).hexdigest(),
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ["hashing_kv"]}
        }
        
        # 计算缓存键
        cache_key = hashlib.md5(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()
        
        return cache_key
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        从缓存获取结果
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存的结果，如果没有则返回None
        """
        if not self.config.use_cache:
            return None
        
        # 检查内存缓存
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # 检查文件缓存
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        result = json.load(f)
                        self.cache[cache_key] = result
                        self.cache_hits += 1
                        return result
                except Exception as e:
                    logger.warning(f"读取缓存失败: {str(e)}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """
        保存结果到缓存
        
        Args:
            cache_key: 缓存键
            result: 要缓存的结果
        """
        if not self.config.use_cache:
            return
        
        # 保存到内存缓存
        self.cache[cache_key] = result
        
        # 保存到文件缓存
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"写入缓存失败: {str(e)}")
    
    async def generate_answer(
        self,
        query: str,
        context: str,
        confidence: Optional[float] = None,
        complexity: Optional[str] = None,
        prompt_template: Optional[PromptTemplate] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        生成答案
        
        Args:
            query: 用户查询
            context: 上下文信息
            confidence: 置信度，如果提供则使用置信度感知的提示模板
            complexity: 复杂度类型
            prompt_template: 自定义提示模板，如果提供则使用它
            **kwargs: 其他参数
            
        Returns:
            (生成的答案, 元数据)
        """
        start_time = time.time()
        self.call_count += 1
        
        # 计算缓存键
        cache_key = self._compute_cache_key(query, context, confidence=confidence, complexity=complexity, **kwargs)
        
        # 检查缓存
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result["answer"], cached_result["metadata"]
        
        # 选择提示模板
        template = prompt_template or self.prompt_template
        
        # 如果有置信度信息，且当前模板不是置信度感知的，则使用置信度感知的模板
        if confidence is not None and not isinstance(template, ConfidenceAwarePrompt):
            template = ConfidenceAwarePrompt()
        
        # 生成提示
        prompt_input = {
            "query": query,
            "context": context,
            **kwargs
        }
        
        if confidence is not None:
            prompt_input["confidence"] = confidence
            
        if complexity is not None:
            prompt_input["complexity"] = complexity
        
        prompt = template.generate(**prompt_input)
        
        # 准备消息格式
        messages = []
        
        # 处理系统提示
        system_prompt = self.config.system_prompt
        if isinstance(prompt, dict) and "system" in prompt:
            system_prompt = prompt["system"]
            prompt = prompt["user"]
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加用户消息
        messages.append({"role": "user", "content": prompt})
        
        # 调用LLM生成回答
        try:
            response = await self._call_llm(messages)
            answer = response["choices"][0]["message"]["content"].strip()
            
            # 构建元数据
            metadata = {
                "model": self.config.model_name,
                "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": response.get("usage", {}).get("total_tokens", 0),
                "response_time": time.time() - start_time,
                "confidence": confidence,
                "complexity": complexity,
                "cached": False,
                "api_type": self.config.api_type
            }
            
            # 缓存结果
            self._save_to_cache(cache_key, {"answer": answer, "metadata": metadata})
            
            return answer, metadata
            
        except Exception as e:
            logger.error(f"生成答案失败: {str(e)}")
            
            # 返回错误信息
            error_msg = f"很抱歉，生成答案时出现错误: {str(e)}"
            metadata = {
                "error": str(e),
                "response_time": time.time() - start_time,
                "cached": False,
                "api_type": self.config.api_type
            }
            
            return error_msg, metadata
    
    async def _call_llm(self, messages: List[Dict[str, str]], retry_count: int = 0) -> Dict[str, Any]:
        """
        调用LLM API
        
        Args:
            messages: 消息列表
            retry_count: 当前重试次数
            
        Returns:
            API响应
            
        Raises:
            Exception: 如果调用失败且超过最大重试次数
        """
        try:
            # 准备API调用参数
            common_params = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
                "timeout": self.config.timeout,
                **self.config.extra_params
            }
            
            # 调用API
            completion = await self.async_client.chat.completions.create(**common_params)
            
            # 转换为字典形式
            return completion.model_dump()
            
        except Exception as e:
            if retry_count < self.config.max_retries:
                # 递增重试延迟
                delay = self.config.retry_delay * (2 ** retry_count)
                logger.warning(f"调用LLM失败，{delay:.2f}秒后重试: {str(e)}")
                
                await asyncio.sleep(delay)
                return await self._call_llm(messages, retry_count + 1)
            else:
                raise Exception(f"超过最大重试次数({self.config.max_retries}): {str(e)}")
    
    async def generate_embedding(self, text: Union[str, List[str]]) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        生成文本嵌入向量
        
        Args:
            text: 单个文本字符串或文本列表
            
        Returns:
            (嵌入向量列表, 元数据)
        """
        start_time = time.time()
        
        # 确保文本是列表形式
        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = text
        
        try:
            # 调用嵌入API
            if self.config.api_type == "siliconflow":
                # 使用SiliconFlow API
                model = "BAAI/bge-m3"
                response = await self.async_client.embeddings.create(
                    model=model,
                    input=text_list,
                    encoding_format="float"
                )
            else:
                # 使用默认API
                response = await self.async_client.embeddings.create(
                    model=self.config.model_name,
                    input=text_list
                )
            
            # 提取嵌入向量
            embeddings = [item.embedding for item in response.data]
            
            # 构建元数据
            metadata = {
                "model": response.model,
                "total_tokens": response.usage.total_tokens,
                "response_time": time.time() - start_time,
                "api_type": self.config.api_type
            }
            
            return embeddings, metadata
            
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "call_count": self.call_count,
            "cache_hits": self.cache_hits,
            "cache_hit_ratio": self.cache_hits / max(1, self.call_count),
            "model_name": self.config.model_name,
            "api_type": self.config.api_type
        }
    
    def add_feedback(self, query_id: str, feedback_data: Dict[str, Any]) -> bool:
        """
        添加反馈
        
        Args:
            query_id: 查询ID
            feedback_data: 反馈数据
            
        Returns:
            是否成功添加反馈
        """
        if self.feedback_processor:
            return self.feedback_processor.add_feedback(query_id, feedback_data)
        return False
    
    async def generate_answer_with_confidence(
        self, 
        query: str, 
        context: str, 
        confidence: float = 0.8
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        生成带置信度的答案
        
        Args:
            query: 查询文本
            context: 上下文信息
            confidence: 置信度阈值
            
        Returns:
            (答案文本, 置信度, 元数据)
        """
        answer, metadata = await self.generate_answer(
            query=query, 
            context=context, 
            confidence=confidence
        )
        return answer, confidence, metadata 