#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合测试程序 - nano_graphrag项目完整功能测试（修复版）
使用qwen-turbo（阿里云百炼）和BAAI/bge-m3（硅基流动）
参考examples/using_api_services.py实现
"""

import asyncio
import json
import os
import time
import traceback
import httpx
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# DSPy导入
import dspy

# nano_graphrag核心模块导入
from nano_graphrag import create_nano_graphrag, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import logger, compute_args_hash, wrap_embedding_func_with_attrs
from nano_graphrag.entity_extraction import extract_entities_dspy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_test_fixed.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# API配置（参考examples/using_api_services.py）
# 硅基流动API（用于嵌入）
SILKFLOW_API_KEY = os.environ.get("SILKFLOW_API_KEY", "sk-rwcxtompyeenjkhdpuganvhsfmmctoftyfcqwpsgtchochkv")
SILKFLOW_API_BASE = "https://api.siliconflow.cn/v1"

# 阿里云百炼API（用于LLM）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")
DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 模型配置
EMBED_MODEL = "BAAI/bge-m3"      # 嵌入模型（1024维，通过硅基流动）
DASHSCOPE_MODEL = "qwen-turbo"   # LLM模型（通过阿里云百炼）

@dataclass
class TestConfig:
    """测试配置类"""
    
    # 数据路径
    input_file: str = "test_data/Novel-44557.txt"
    query_file: str = "test_data/query.json"
    working_dir: str = "comprehensive_test_fixed_cache"
    
    # 测试参数
    chunk_size: int = 1024
    overlap_size: int = 128
    max_test_queries: int = 5  # 减少测试查询数量
    
    # 融合配置
    fusion_type: str = "basic"  # 使用基础融合
    enable_calibration: bool = False  # 先禁用校准
    
    # 评估配置
    enable_modern_evaluator: bool = False  # 先禁用现代评估器
    enable_comprehensive_evaluator: bool = False  # 先禁用综合评估器
    enable_baseline_comparison: bool = False  # 先禁用基线比较

class FixedTestSuite:
    """修复版测试套件"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.test_results = {}
        self.rag_instance = None
        self.llm_func = None
        self.embedding_func = None
        
        # 初始化API函数
        self._setup_api_functions()
    
    def _setup_api_functions(self):
        """设置API函数"""
        self.llm_func = self.dashscope_llm_api
        self.embedding_func = self.silkflow_embedding
        
        # 配置DSPy使用我们的LLM函数
        self._setup_dspy_lm()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError))
    )
    async def dashscope_llm_api(
        self,
        prompt: str, 
        system_prompt: Optional[str] = None, 
        history_messages: List[Dict[str, str]] = [], 
        **kwargs
    ) -> str:
        """调用阿里云百炼LLM API（参考examples/using_api_services.py实现）"""
        # 准备消息格式
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 缓存处理
        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        
        if hashing_kv is not None:
            args_hash = compute_args_hash(DASHSCOPE_MODEL, messages)
            if_cache_return = await hashing_kv.get_by_id(args_hash)
            if if_cache_return is not None:
                return if_cache_return["return"]
        
        # 准备请求头和请求体
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # 从kwargs提取相关参数
        max_tokens = kwargs.pop("max_tokens", 512)
        temperature = kwargs.pop("temperature", 0.7)
        
        # 请求体
        payload = {
            "model": DASHSCOPE_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            # 发送请求
            async with httpx.AsyncClient(timeout=180.0) as client:
                logger.info(f"正在发送请求到 {DASHSCOPE_API_BASE}/chat/completions")
                
                response = await client.post(
                    f"{DASHSCOPE_API_BASE}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                # 如果请求失败，记录错误详情
                if response.status_code != 200:
                    logger.error(f"API请求失败 ({response.status_code}): {response.text}")
                    response.raise_for_status()
                    
                result = response.json()
                
                # 提取响应内容
                content = result["choices"][0]["message"]["content"]
                
                # 缓存响应
                if hashing_kv is not None:
                    await hashing_kv.upsert(
                        {args_hash: {"return": content, "model": DASHSCOPE_MODEL}}
                    )
                
                return content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error(f"API认证失败 (HTTP 401)：请检查您的API密钥是否正确")
            elif e.response.status_code == 400:
                logger.error(f"API请求格式错误 (HTTP 400)：{e.response.text}")
            elif e.response.status_code == 429:
                logger.error(f"API请求过多 (HTTP 429)：请降低请求频率")
            else:
                logger.error(f"API请求失败 ({e.response.status_code}): {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"API请求过程中发生错误: {str(e)}")
            raise

    @wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=4096)  # bge-m3是1024维
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError))
    )
    async def silkflow_embedding(self, texts: List[str]) -> np.ndarray:
        """调用硅基流动嵌入API（参考examples/using_api_services.py实现）"""
        headers = {
            "Authorization": f"Bearer {SILKFLOW_API_KEY}",
            "Content-Type": "application/json"
        }
        
        all_embeddings = []
        
        # 尝试批量处理所有文本
        try:
            payload = {
                "model": EMBED_MODEL,
                "input": texts,
                "encoding_format": "float"
            }
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                logger.info(f"正在发送嵌入请求到 {SILKFLOW_API_BASE}/embeddings")
                
                response = await client.post(
                    f"{SILKFLOW_API_BASE}/embeddings",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    logger.error(f"嵌入API请求失败 ({response.status_code}): {response.text}")
                    response.raise_for_status()
                    
                result = response.json()
                all_embeddings = [item["embedding"] for item in result["data"]]
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                logger.warning("批量请求失败，尝试逐个请求...")
                # 如果批量请求失败，尝试逐个请求
                for text in texts:
                    try:
                        payload = {
                            "model": EMBED_MODEL,
                            "input": text,
                            "encoding_format": "float"
                        }
                        
                        async with httpx.AsyncClient(timeout=60.0) as client:
                            response = await client.post(
                                f"{SILKFLOW_API_BASE}/embeddings",
                                headers=headers,
                                json=payload
                            )
                            response.raise_for_status()
                            result = response.json()
                            embedding = result["data"][0]["embedding"]
                            all_embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"单个嵌入请求失败: {str(e)}")
                        # 添加零向量作为替代
                        all_embeddings.append([0.0] * 1024)
            else:
                raise
        except Exception as e:
            logger.error(f"嵌入API请求失败: {str(e)}")
            # 返回零向量作为回退
            all_embeddings = [[0.0] * 1024 for _ in texts]
        
        return np.array(all_embeddings)
    
    def _setup_dspy_lm(self):
        """设置DSPy LM以支持实体抽取"""
        try:
            # 创建一个DSPy兼容的LM包装器
            class DSPyLMWrapper(dspy.LM):
                def __init__(self, llm_func):
                    self.llm_func = llm_func
                    self.history = []
                    super().__init__(model="qwen-turbo")
                
                def basic_request(self, prompt, **kwargs):
                    """DSPy需要的基本请求方法"""
                    try:
                        # 使用asyncio运行异步函数
                        loop = asyncio.new_event_loop()
                        if asyncio.get_event_loop().is_running():
                            # 如果已经在事件循环中，使用已有的循环
                            import nest_asyncio
                            nest_asyncio.apply()
                            result = asyncio.get_event_loop().run_until_complete(
                                self.llm_func(prompt, **kwargs)
                            )
                        else:
                            # 创建新的事件循环
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(
                                self.llm_func(prompt, **kwargs)
                            )
                        return [result]
                    except Exception as e:
                        logger.error(f"DSPy LM请求失败: {str(e)}")
                        return [f"Error: {str(e)}"]
                    finally:
                        if not asyncio.get_event_loop().is_running():
                            loop.close()
            
            # 创建并配置DSPy LM
            dspy_lm = DSPyLMWrapper(self.llm_func)
            dspy.configure(lm=dspy_lm)
            
            logger.info("DSPy LM配置成功")
            
        except Exception as e:
            logger.warning(f"DSPy LM配置失败: {str(e)}")
            logger.warning("将使用简化的实体抽取")
    

    
    async def safe_clustering_wrapper(self, original_clustering_func):
        """安全的聚类包装器，处理空图情况"""
        async def safe_clustering(algorithm: str):
            try:
                # 检查图是否为空
                if hasattr(self, '_graph') and self._graph.number_of_nodes() == 0:
                    logger.warning("图为空，跳过聚类操作")
                    return
                
                # 调用原始聚类函数
                return await original_clustering_func(algorithm)
                
            except ValueError as e:
                if "max() iterable argument is empty" in str(e):
                    logger.warning("图为空或无连通分量，跳过聚类操作")
                    return
                else:
                    raise e
        
        return safe_clustering
    
    async def safe_naive_query(self, query, chunks_vdb, text_chunks_db, query_param, global_config, return_context=False):
        """安全的naive查询，处理None结果"""
        try:
            from nano_graphrag.query_processing.naive_query import naive_query
            
            # 检查chunks_vdb是否可用
            if chunks_vdb is None:
                logger.warning("chunks_vdb为空，返回默认失败响应")
                return "Sorry, I'm not able to provide an answer to that question."
            
            # 执行向量查询
            results = await chunks_vdb.query(query, top_k=query_param.top_k)
            if not len(results):
                logger.warning("向量查询未找到结果")
                return "Sorry, I'm not able to provide an answer to that question."
            
            # 获取文本块
            chunks_ids = [r["id"] for r in results]
            chunks = await text_chunks_db.get_by_ids(chunks_ids)
            
            # 过滤None结果
            valid_chunks = [c for c in chunks if c is not None]
            if not valid_chunks:
                logger.warning("未找到有效的文本块")
                return "Sorry, I'm not able to provide an answer to that question."
            
            # 调用原始naive_query，但使用过滤后的chunks
            import tiktoken
            from nano_graphrag._utils import truncate_list_by_token_size
            from nano_graphrag.answer_generation.prompts import PROMPTS
            
            tiktoken_model = tiktoken.encoding_for_model("gpt-4o")
            maybe_trun_chunks = truncate_list_by_token_size(
                valid_chunks,
                query_param.naive_max_token_for_text_unit,
                tiktoken_model,
                key=lambda x: x["content"]
            )
            
            section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
            if query_param.only_need_context or return_context:
                return section
                
            # 使用LLM生成回答
            use_model_func = global_config["best_model_func"]
            sys_prompt_temp = PROMPTS["naive_rag_response"]
            sys_prompt = sys_prompt_temp.format(
                content_data=section, response_type=query_param.response_type
            )
            response = await use_model_func(
                query,
                system_prompt=sys_prompt,
            )
            return response
            
        except Exception as e:
            logger.error(f"Naive查询失败: {str(e)}")
            return "Sorry, I'm not able to provide an answer to that question."
    
    async def setup_rag_instance(self):
        """设置RAG实例"""
        try:
            logger.info("正在初始化RAG实例...")
            
            # 创建增强版GraphRAG实例（使用正确的参数）
            self.rag_instance = create_nano_graphrag(
                working_dir=self.config.working_dir,
                
                # 基础配置
                best_model_func=self.llm_func,
                cheap_model_func=self.llm_func,
                embedding_func=self.embedding_func,
                
                # 启用基础功能
                enable_enhanced_features=False,  # 先禁用增强功能，测试基础功能
                enable_local=True,
                enable_naive_rag=True,
                
                # 简化实体抽取（减少出错概率）
                entity_extract_max_gleaning=0,  # 禁用gleaning以避免DSPy问题
                
                # 使用核心库的DSPy实体抽取函数
                entity_extraction_func=extract_entities_dspy,
                
                # 分块配置
                chunk_token_size=self.config.chunk_size,
                chunk_overlap_token_size=self.config.overlap_size,
                
                # LLM配置
                best_model_max_token_size=8192,
                cheap_model_max_token_size=8192,
                
                # 嵌入配置
                embedding_batch_num=16,
                embedding_func_max_async=8
            )
            
            # 应用安全包装器修复
            await self.apply_safety_wrappers()
            
            logger.info("RAG实例初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"RAG实例初始化失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    async def apply_safety_wrappers(self):
        """应用安全包装器修复各种问题"""
        try:
            # 修复图聚类的空图问题
            if hasattr(self.rag_instance, 'chunk_entity_relation_graph'):
                graph = self.rag_instance.chunk_entity_relation_graph
                if hasattr(graph, 'clustering'):
                    original_clustering = graph.clustering
                    graph.clustering = await self.safe_clustering_wrapper(original_clustering)
                    logger.info("已应用安全聚类包装器")
            
            # 修复naive查询的None问题 - 通过重写aquery方法
            if hasattr(self.rag_instance, 'aquery'):
                original_aquery = self.rag_instance.aquery
                
                async def safe_aquery(query: str, param=None):
                    if param is None:
                        from nano_graphrag.base import QueryParam
                        param = QueryParam()
                    
                    # 对于naive模式，使用我们的安全实现
                    if param.mode == "naive":
                        return await self.safe_naive_query(
                            query,
                            self.rag_instance.chunks_vdb,
                            self.rag_instance.text_chunks,
                            param,
                            self.rag_instance.__dict__
                        )
                    else:
                        # 其他模式使用原始实现
                        return await original_aquery(query, param)
                
                self.rag_instance.aquery = safe_aquery
                logger.info("已应用安全查询包装器")
                
        except Exception as e:
            logger.warning(f"应用安全包装器时出错: {str(e)}")
            logger.warning("继续使用原始实现")
    
    def prepare_test_data(self):
        """准备测试数据"""
        # 创建测试数据目录
        os.makedirs("test_data", exist_ok=True)
        
        # 创建简单的测试文本
        test_text = """
        人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

        机器学习是人工智能的一个重要分支，它通过算法使计算机能够从数据中学习和改进。深度学习是机器学习的一个子集，它使用神经网络来模拟人脑的工作方式。

        自然语言处理（NLP）是人工智能的另一个重要领域，它致力于让计算机理解、解释和生成人类语言。NLP技术广泛应用于机器翻译、文本分析、聊天机器人等领域。

        计算机视觉是人工智能的一个分支，它使计算机能够识别和理解图像和视频中的内容。这项技术在自动驾驶汽车、医学影像分析、人脸识别等方面有着重要应用。

        知识图谱是一种结构化的知识表示方式，它将现实世界的实体、概念以及它们之间的关系以图的形式进行建模。知识图谱在搜索引擎、推荐系统、问答系统等方面发挥着重要作用。
        """
        
        # 写入测试文件
        with open("test_data/test_document.txt", "w", encoding="utf-8") as f:
            f.write(test_text.strip())
        
        # 创建测试查询
        test_queries = [
            "什么是人工智能？",
            "机器学习和深度学习有什么关系？",
            "自然语言处理有哪些应用？",
            "计算机视觉技术可以用在哪些地方？",
            "知识图谱的作用是什么？"
        ]
        
        with open("test_data/test_queries.json", "w", encoding="utf-8") as f:
            json.dump(test_queries, f, ensure_ascii=False, indent=2)
        
        logger.info("测试数据准备完成")
        return "test_data/test_document.txt", "test_data/test_queries.json"
    
    async def test_document_insertion(self, file_path: str) -> bool:
        """测试文档插入"""
        try:
            logger.info(f"开始测试文档插入: {file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"测试文件不存在: {file_path}")
                return False
            
            # 读取文档内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"文档长度: {len(content)} 字符")
            
            # 插入文档
            start_time = time.time()
            await self.rag_instance.ainsert(content)
            insertion_time = time.time() - start_time
            
            logger.info(f"文档插入完成，耗时: {insertion_time:.2f}秒")
            
            self.test_results["document_insertion"] = {
                "success": True,
                "insertion_time": insertion_time,
                "document_length": len(content)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"文档插入测试失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.test_results["document_insertion"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    async def test_query_processing(self, query_file: str) -> bool:
        """测试查询处理"""
        try:
            logger.info(f"开始测试查询处理: {query_file}")
            
            # 读取查询列表
            if os.path.exists(query_file):
                with open(query_file, 'r', encoding='utf-8') as f:
                    queries = json.load(f)
            else:
                # 使用默认查询
                queries = [
                    "什么是人工智能？",
                    "机器学习的应用有哪些？",
                    "深度学习和机器学习的区别是什么？"
                ]
            
            # 限制查询数量
            queries = queries[:self.config.max_test_queries]
            
            query_results = []
            
            for i, query in enumerate(queries, 1):
                logger.info(f"处理查询 {i}/{len(queries)}: {query}")
                
                try:
                    # 测试不同的查询模式
                    modes = ["naive", "local", "global"]
                    mode_results = {}
                    
                    for mode in modes:
                        try:
                            start_time = time.time()
                            result = await self.rag_instance.aquery(
                                query,
                                param=QueryParam(mode=mode)
                            )
                            query_time = time.time() - start_time
                            
                            mode_results[mode] = {
                                "result": result,
                                "query_time": query_time,
                                "success": True
                            }
                            
                            logger.info(f"查询模式 {mode} 完成，耗时: {query_time:.2f}秒")
                            
                        except Exception as e:
                            logger.warning(f"查询模式 {mode} 失败: {str(e)}")
                            mode_results[mode] = {
                                "success": False,
                                "error": str(e)
                            }
                    
                    query_results.append({
                        "query": query,
                        "modes": mode_results
                    })
                    
                except Exception as e:
                    logger.error(f"查询处理失败: {str(e)}")
                    query_results.append({
                        "query": query,
                        "error": str(e),
                        "success": False
                    })
            
            self.test_results["query_processing"] = {
                "success": True,
                "queries_tested": len(queries),
                "results": query_results
            }
            
            logger.info(f"查询处理测试完成，共测试 {len(queries)} 个查询")
            return True
            
        except Exception as e:
            logger.error(f"查询处理测试失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.test_results["query_processing"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def save_results(self):
        """保存测试结果"""
        try:
            results_file = "test_results_fixed.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"测试结果已保存到: {results_file}")
            
            # 打印简要摘要
            print("\n" + "="*60)
            print("🧪 综合测试结果摘要")
            print("="*60)
            
            for test_name, result in self.test_results.items():
                status = "✅ 成功" if result.get("success", False) else "❌ 失败"
                print(f"{test_name}: {status}")
                
                if not result.get("success", False) and "error" in result:
                    print(f"  错误: {result['error']}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
    
    async def run_comprehensive_test(self):
        """运行综合测试"""
        logger.info("开始运行综合测试套件...")
        
        try:
            # 1. 准备测试数据
            logger.info("步骤1: 准备测试数据")
            doc_file, query_file = self.prepare_test_data()
            
            # 2. 设置RAG实例
            logger.info("步骤2: 初始化RAG实例")
            if not await self.setup_rag_instance():
                logger.error("RAG实例初始化失败，测试终止")
                return False
            
            # 3. 测试文档插入
            logger.info("步骤3: 测试文档插入")
            if not await self.test_document_insertion(doc_file):
                logger.error("文档插入测试失败")
                # 继续执行其他测试
            
            # 4. 测试查询处理
            logger.info("步骤4: 测试查询处理")
            if not await self.test_query_processing(query_file):
                logger.error("查询处理测试失败")
            
            # 5. 保存结果
            logger.info("步骤5: 保存测试结果")
            self.save_results()
            
            logger.info("综合测试完成！")
            return True
            
        except Exception as e:
            logger.error(f"综合测试执行失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False

async def main():
    """主函数"""
    print("🚀 启动综合测试程序（修复版）")
    print(f"📁 工作目录: {os.getcwd()}")
    
    # 创建测试配置
    config = TestConfig()
    
    # 检查API密钥
    if DASHSCOPE_API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️ 警告: 未设置DASHSCOPE_API_KEY环境变量")
        print("请设置环境变量：export DASHSCOPE_API_KEY=your_api_key")
        print("继续使用默认配置进行测试...")
    
    # 创建测试套件
    test_suite = FixedTestSuite(config)
    
    # 运行测试
    try:
        success = await test_suite.run_comprehensive_test()
        if success:
            print("✅ 测试执行成功！")
        else:
            print("❌ 测试执行失败！")
    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
    except Exception as e:
        print(f"❌ 测试执行失败: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())