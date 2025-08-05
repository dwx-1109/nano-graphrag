"""
银标签生成模块 - 使用大模型自动标注数据

本模块提供了使用大语言模型（如Qwen-Turbo）自动生成训练数据标签的功能。
主要用于生成复杂度分类器的训练数据，包括zero-hop、one-hop和multi-hop三种复杂度级别。
"""

import os
import json
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("silver-labeling")


@dataclass
class SilverLabelConfig:
    """银标签生成配置"""
    
    # 输入输出配置
    input_file: str = "datasets/data_raw.json"
    output_file: str = "datasets/data_labeled.json"
    log_file: str = "logs/silver_label_log.jsonl"
    
    # API配置
    api_type: str = "openai"  # openai, qwen, custom
    api_base: str = None
    api_key: str = None
    api_model: str = "qwen-turbo"
    
    # 生成配置
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: int = 5
    temperature: float = 0.1
    
    # 字段配置
    text_field: str = "query"
    label_field: str = "complexity"
    
    # 提示词配置
    system_prompt: str = """你是一个专业的查询复杂度分类器。你的任务是将用户的查询分类为以下三种复杂度级别之一：

1. zero_hop: 简单查询，无需检索外部知识，LLM可以直接回答，如常识性问题、简单定义等。
2. one_hop: 中等复杂度查询，需要单步检索，如具体事实、单一实体信息等。
3. multi_hop: 复杂查询，需要多步推理或关系理解，如比较、因果分析、多实体关系等。

请仔细分析查询的特点，并给出最合适的复杂度分类。"""
    
    user_prompt_template: str = """请分析以下查询，并将其分类为zero_hop、one_hop或multi_hop三种复杂度之一。

查询: {query}

请以JSON格式回答，包含以下字段:
1. complexity: 复杂度分类，必须是"zero_hop"、"one_hop"或"multi_hop"之一
2. reasoning: 你的分类理由

JSON格式:
{{
  "complexity": "复杂度分类",
  "reasoning": "分类理由"
}}"""
    
    # 并发配置
    max_concurrency: int = 5
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)


class SilverLabelGenerator:
    """银标签生成器"""
    
    def __init__(self, config: SilverLabelConfig = None):
        """
        初始化银标签生成器
        
        Args:
            config: 银标签生成配置，如果为None则使用默认配置
        """
        self.config = config or SilverLabelConfig()
        self.client = None
        self.semaphore = None
    
    def _init_client(self):
        """初始化API客户端"""
        if self.client:
            return
        
        try:
            if self.config.api_type == "openai":
                from openai import OpenAI
                
                # 设置API密钥
                api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("未提供OpenAI API密钥")
                
                # 设置API基础URL
                api_base = self.config.api_base or os.environ.get("OPENAI_API_BASE")
                
                # 创建客户端
                self.client = OpenAI(api_key=api_key, base_url=api_base)
                logger.info("已初始化OpenAI客户端")
                
            elif self.config.api_type == "qwen":
                from dashscope import Generation
                
                # 设置API密钥
                api_key = self.config.api_key or os.environ.get("DASHSCOPE_API_KEY")
                if not api_key:
                    raise ValueError("未提供DashScope API密钥")
                
                # 创建客户端
                self.client = Generation()
                os.environ["DASHSCOPE_API_KEY"] = api_key
                logger.info("已初始化DashScope客户端")
                
            elif self.config.api_type == "custom":
                # 使用自定义API
                import requests
                self.client = requests
                logger.info("已初始化自定义API客户端")
                
            else:
                raise ValueError(f"不支持的API类型: {self.config.api_type}")
                
        except ImportError as e:
            logger.error(f"初始化客户端失败，缺少必要的依赖: {e}")
            raise
        except Exception as e:
            logger.error(f"初始化客户端失败: {e}")
            raise
    
    async def _call_api(self, query: str) -> Dict[str, Any]:
        """
        调用API进行标注
        
        Args:
            query: 查询文本
            
        Returns:
            标注结果
        """
        # 格式化提示词
        user_prompt = self.config.user_prompt_template.format(query=query)
        
        # 初始化结果
        result = {
            "query": query,
            "timestamp": time.time(),
            "success": False,
            "complexity": None,
            "reasoning": None,
            "error": None,
            "raw_response": None
        }
        
        # 重试机制
        for attempt in range(self.config.max_retries):
            try:
                if self.config.api_type == "openai":
                    # 使用OpenAI API
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=self.config.api_model,
                        messages=[
                            {"role": "system", "content": self.config.system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.config.temperature
                    )
                    
                    # 提取响应
                    response_text = response.choices[0].message.content
                    result["raw_response"] = response_text
                    
                elif self.config.api_type == "qwen":
                    # 使用DashScope API
                    response = await asyncio.to_thread(
                        self.client.call,
                        model=self.config.api_model,
                        prompt=[
                            {"role": "system", "content": self.config.system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.config.temperature
                    )
                    
                    # 提取响应
                    response_text = response["output"]["text"]
                    result["raw_response"] = response_text
                    
                elif self.config.api_type == "custom":
                    # 使用自定义API
                    api_base = self.config.api_base or "http://localhost:8000/v1/chat/completions"
                    
                    response = await asyncio.to_thread(
                        self.client.post,
                        api_base,
                        json={
                            "model": self.config.api_model,
                            "messages": [
                                {"role": "system", "content": self.config.system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "temperature": self.config.temperature
                        },
                        headers={"Content-Type": "application/json"}
                    )
                    
                    # 提取响应
                    response_json = response.json()
                    response_text = response_json["choices"][0]["message"]["content"]
                    result["raw_response"] = response_text
                
                # 解析JSON响应
                try:
                    # 提取JSON部分
                    import re
                    json_match = re.search(r'({[\s\S]*})', response_text)
                    if json_match:
                        json_str = json_match.group(1)
                        parsed = json.loads(json_str)
                    else:
                        parsed = json.loads(response_text)
                    
                    # 提取复杂度和理由
                    complexity = parsed.get("complexity", "").strip().lower()
                    reasoning = parsed.get("reasoning", "").strip()
                    
                    # 验证复杂度
                    if complexity not in ["zero_hop", "one_hop", "multi_hop"]:
                        raise ValueError(f"无效的复杂度分类: {complexity}")
                    
                    # 更新结果
                    result["success"] = True
                    result["complexity"] = complexity
                    result["reasoning"] = reasoning
                    
                    # 标注成功，返回结果
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"解析JSON响应失败: {e}")
                    result["error"] = f"JSON解析错误: {e}"
                    
                except Exception as e:
                    logger.warning(f"处理响应失败: {e}")
                    result["error"] = f"处理响应错误: {e}"
            
            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt+1}/{self.config.max_retries}): {e}")
                result["error"] = f"API调用错误: {e}"
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay)
        
        # 所有重试都失败
        return result
    
    async def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理一批数据
        
        Args:
            batch: 数据批次
            
        Returns:
            处理结果
        """
        tasks = []
        
        for item in batch:
            # 获取查询文本
            query = item.get(self.config.text_field, "")
            
            # 创建任务
            task = asyncio.create_task(self._process_item(query))
            tasks.append((item, task))
        
        # 等待所有任务完成
        results = []
        for item, task in tasks:
            try:
                result = await task
                
                # 复制原始数据项
                labeled_item = item.copy()
                
                # 添加标签和理由
                if result["success"]:
                    labeled_item[self.config.label_field] = result["complexity"]
                    labeled_item["reasoning"] = result["reasoning"]
                
                # 添加到结果
                results.append((labeled_item, result))
                
            except Exception as e:
                logger.error(f"处理项目失败: {e}")
                results.append((item, {
                    "query": item.get(self.config.text_field, ""),
                    "timestamp": time.time(),
                    "success": False,
                    "error": f"处理失败: {e}"
                }))
        
        return results
    
    async def _process_item(self, query: str) -> Dict[str, Any]:
        """
        处理单个数据项
        
        Args:
            query: 查询文本
            
        Returns:
            处理结果
        """
        # 使用信号量限制并发
        async with self.semaphore:
            return await self._call_api(query)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        加载数据
        
        Returns:
            数据列表
        """
        try:
            with open(self.config.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 如果数据是字典，尝试提取items字段
            if isinstance(data, dict) and "items" in data:
                data = data["items"]
            
            # 确保数据是列表
            if not isinstance(data, list):
                logger.warning(f"输入数据不是列表，尝试转换")
                data = [data]
            
            logger.info(f"从 {self.config.input_file} 加载了 {len(data)} 条数据")
            return data
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return []
    
    def _save_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        保存数据
        
        Args:
            data: 数据列表
            
        Returns:
            是否保存成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(self.config.output_file)), exist_ok=True)
            
            with open(self.config.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"已保存 {len(data)} 条数据到 {self.config.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            return False
    
    def _append_log(self, log_entry: Dict[str, Any]) -> bool:
        """
        追加日志
        
        Args:
            log_entry: 日志条目
            
        Returns:
            是否追加成功
        """
        try:
            # 确保日志目录存在
            os.makedirs(os.path.dirname(os.path.abspath(self.config.log_file)), exist_ok=True)
            
            # 追加日志
            with open(self.config.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"追加日志失败: {e}")
            return False
    
    async def generate(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        生成银标签
        
        Returns:
            (标注数据, 统计信息)
        """
        # 初始化客户端
        self._init_client()
        
        # 创建信号量
        self.semaphore = asyncio.Semaphore(self.config.max_concurrency)
        
        # 加载数据
        data = self._load_data()
        if not data:
            logger.error("没有数据可处理")
            return [], {"total": 0, "success": 0, "failed": 0}
        
        # 初始化结果
        labeled_data = []
        stats = {
            "total": len(data),
            "success": 0,
            "failed": 0,
            "complexity_distribution": {
                "zero_hop": 0,
                "one_hop": 0,
                "multi_hop": 0
            }
        }
        
        # 分批处理
        for i in range(0, len(data), self.config.batch_size):
            batch = data[i:i + self.config.batch_size]
            logger.info(f"处理批次 {i//self.config.batch_size + 1}/{(len(data)-1)//self.config.batch_size + 1} ({len(batch)} 条)")
            
            # 处理批次
            batch_results = await self._process_batch(batch)
            
            # 处理结果
            for item, result in batch_results:
                # 追加日志
                self._append_log(result)
                
                # 添加到结果
                labeled_data.append(item)
                
                # 更新统计信息
                if result.get("success", False):
                    stats["success"] += 1
                    complexity = result.get("complexity")
                    if complexity in stats["complexity_distribution"]:
                        stats["complexity_distribution"][complexity] += 1
                else:
                    stats["failed"] += 1
            
            # 每批次保存一次
            self._save_data(labeled_data)
            
            # 打印进度
            logger.info(f"进度: {len(labeled_data)}/{len(data)} ({len(labeled_data)/len(data)*100:.1f}%)")
            logger.info(f"成功: {stats['success']}, 失败: {stats['failed']}")
        
        # 最终保存
        self._save_data(labeled_data)
        
        # 返回结果
        return labeled_data, stats
    
    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        运行银标签生成
        
        Returns:
            (标注数据, 统计信息)
        """
        # 创建事件循环
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 运行生成
        return loop.run_until_complete(self.generate())


def generate_silver_labels(config_path: str = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    生成银标签的便捷函数
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
        
    Returns:
        (标注数据, 统计信息)
    """
    # 加载配置
    if config_path and os.path.exists(config_path):
        logger.info(f"加载配置: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = SilverLabelConfig(**config_dict)
    else:
        logger.info("使用默认配置")
        config = SilverLabelConfig()
    
    # 创建生成器
    generator = SilverLabelGenerator(config)
    
    # 运行生成
    return generator.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成银标签")
    parser.add_argument("--config", type=str, help="配置文件路径")
    args = parser.parse_args()
    
    labeled_data, stats = generate_silver_labels(args.config)
    
    logger.info("银标签生成完成")
    logger.info(f"总数: {stats['total']}")
    logger.info(f"成功: {stats['success']}")
    logger.info(f"失败: {stats['failed']}")
    logger.info(f"复杂度分布: {stats['complexity_distribution']}") 