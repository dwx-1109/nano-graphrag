"""
Enhanced GraphRAG System
集成了最新的Graph Transformer、增强融合策略、现代评估器等模块
适配本科生项目的实际需求，保持高性能和可用性
"""

import os
import asyncio
from datetime import datetime
from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Type, Union, List, Dict, Any, Optional, Callable

import tiktoken

from ._utils import (
    logger,
    always_get_an_event_loop,
    limit_async_func_call,
    convert_response_to_json,
    compute_mdhash_id,
)

# 核心存储和处理模块
from .base import (
    BaseVectorStorage,
    BaseKVStorage, 
    BaseGraphStorage,
    QueryParam,
    CommunitySchema,
    TextChunkSchema,
    StorageNameSpace,
)

# 数据处理模块
from .chunking import get_chunks, chunking_by_token_size
from .entity_extraction import extract_entities
from .community import generate_community_report

# 查询处理模块
from .query_processing import local_query, global_query, naive_query

# 存储实现
from ._storage import JsonKVStorage, NetworkXStorage, NanoVectorDBStorage

# LLM和嵌入函数
from ._llm import (
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_gpt_4o_mini_complete,
    azure_openai_embedding,
    amazon_bedrock_embedding,
    create_amazon_bedrock_complete_function,
)

# 新增的高级模块
from .graph_neural_network.graph_transformer import (
    create_graph_transformer,
    create_graph_transformer_config,
    GraphTransformerConfig
)
from .fusion_advanced.enhanced_fusion import (
    create_enhanced_fusion,
    create_enhanced_fusion_config,
    EnhancedFusionConfig
)
from .evaluation_advanced.modern_evaluator import (
    create_modern_evaluator,
    create_modern_evaluator_config,
    ModernEvaluatorConfig
)
from .optimizer.simple_optimizer import (
    create_simple_optimizer,
    create_simple_optimizer_config,
    SimpleOptimizerConfig
)
from .complexity.router import ComplexityAwareRouter


@dataclass
class EnhancedGraphRAG:
    """
    增强版GraphRAG系统
    集成了最新的技术模块：Graph Transformer、增强融合、现代评估等
    """
    
    # 基础配置
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    
    # 模式控制
    enable_local: bool = True
    enable_naive_rag: bool = True  # 默认启用，增加灵活性
    enable_enhanced_features: bool = True  # 是否启用增强功能
    
    # 文本分块配置
    chunk_func: Callable = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"
    
    # 实体提取配置
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    
    # 图聚类配置
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    
    # 节点嵌入配置（保留兼容性）
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )
    
    # 社区报告配置
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    
    # 嵌入配置
    embedding_func: Callable = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2
    
    # LLM配置
    using_azure_openai: bool = False
    using_amazon_bedrock: bool = False
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"
    best_model_func: callable = gpt_4o_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16
    
    # 实体提取函数
    entity_extraction_func: callable = extract_entities
    
    # DSPy实体提取配置
    use_compiled_dspy_entity_relationship: bool = False  # 禁用编译模型避免字段冲突
    
    # 存储配置
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True
    
    # 增强功能配置
    graph_transformer_config: Optional[GraphTransformerConfig] = None
    enhanced_fusion_config: Optional[EnhancedFusionConfig] = None
    modern_evaluator_config: Optional[ModernEvaluatorConfig] = None
    simple_optimizer_config: Optional[SimpleOptimizerConfig] = None
    
    # 扩展配置
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        """初始化增强版GraphRAG系统"""
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"Enhanced GraphRAG init with param:\n\n  {_print_config}\n")

        # 配置LLM提供商
        self._setup_llm_providers()
        
        # 创建工作目录
        self._setup_working_directory()
        
        # 初始化存储组件
        self._setup_storage_components()
        
        # 初始化LLM和嵌入函数
        self._setup_llm_and_embedding()
        
        # 初始化增强功能模块
        if self.enable_enhanced_features:
            self._setup_enhanced_modules()
    
    def _setup_llm_providers(self):
        """设置LLM提供商"""
        if self.using_azure_openai:
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info("Switched to Azure OpenAI")

        if self.using_amazon_bedrock:
            self.best_model_func = create_amazon_bedrock_complete_function(self.best_model_id)
            self.cheap_model_func = create_amazon_bedrock_complete_function(self.cheap_model_id)
            self.embedding_func = amazon_bedrock_embedding
            logger.info("Switched to Amazon Bedrock")
    
    def _setup_working_directory(self):
        """设置工作目录"""
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
    
    def _setup_storage_components(self):
        """设置存储组件"""
        config_dict = asdict(self)
        
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=config_dict
        )
        
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=config_dict
        )
        
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=config_dict
            )
            if self.enable_llm_cache
            else None
        )
        
        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=config_dict
        )
        
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=config_dict
        )
    
    def _setup_llm_and_embedding(self):
        """设置LLM和嵌入函数"""
        # 限制并发调用
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        
        # 设置向量数据库
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )
        
        # 设置LLM函数
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )
    
    def _setup_enhanced_modules(self):
        """设置增强功能模块"""
        try:
            # Graph Transformer
            if self.graph_transformer_config is None:
                self.graph_transformer_config = create_graph_transformer_config()
            self.graph_transformer = create_graph_transformer(self.graph_transformer_config)
            logger.info("Graph Transformer initialized successfully")
            
            # Enhanced Fusion
            if self.enhanced_fusion_config is None:
                self.enhanced_fusion_config = create_enhanced_fusion_config()
            self.enhanced_fusion = create_enhanced_fusion(self.enhanced_fusion_config)
            logger.info("Enhanced Fusion initialized successfully")
            
            # Modern Evaluator
            if self.modern_evaluator_config is None:
                self.modern_evaluator_config = create_modern_evaluator_config()
            self.modern_evaluator = create_modern_evaluator(self.modern_evaluator_config)
            logger.info("Modern Evaluator initialized successfully")
            
            # Simple Optimizer
            if self.simple_optimizer_config is None:
                self.simple_optimizer_config = create_simple_optimizer_config()
            self.simple_optimizer = create_simple_optimizer(self.simple_optimizer_config)
            logger.info("Simple Optimizer initialized successfully")
            
            # Complexity Router
            self.complexity_router = ComplexityAwareRouter()
            logger.info("Complexity Router initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced modules: {e}")
            logger.warning("Falling back to basic functionality")
            self.enable_enhanced_features = False

    def insert(self, string_or_strings):
        """同步插入接口"""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        """同步查询接口"""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))
    
    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """
        异步查询接口
        集成了复杂度感知路由和增强融合策略
        """
        # 验证模式可用性
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        
        # 使用复杂度感知路由（如果启用）
        if self.enable_enhanced_features and hasattr(self, 'complexity_router'):
            try:
                available_modes = ["llm_only", "naive", "bm25", "local", "global"]
                if not self.enable_local:
                    available_modes = [m for m in available_modes if m not in ["local", "global"]]
                if not self.enable_naive_rag:
                    available_modes = [m for m in available_modes if m != "naive"]
                
                # 使用路由器选择最佳模式
                optimal_mode = await self.complexity_router.route(query, available_modes)
                if optimal_mode != param.mode:
                    logger.info(f"Complexity router suggests mode: {optimal_mode} (original: {param.mode})")
                    param.mode = optimal_mode
            except Exception as e:
                logger.warning(f"Complexity routing failed: {e}, using original mode")
        
        # 执行查询
        if param.mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self._get_query_config(),
            )
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self._get_query_config(),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                self._get_query_config(),
            )
        elif param.mode == "bm25":
            # 导入bm25查询函数
            from .query_processing.bm25_query import bm25_query
            response = await bm25_query(
                query,
                self.bm25_storage if hasattr(self, 'bm25_storage') else None,
                self.text_chunks,
                param,
                self._get_query_config(),
            )
        elif param.mode == "llm_only":
            # 导入llm_only查询函数
            from .query_processing.llm_only_query import llm_only_query
            response = await llm_only_query(
                query,
                param,
                self._get_query_config(),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        
        await self._query_done()
        return response

    async def ainsert(self, string_or_strings):
        """异步插入接口"""
        await self._insert_start()
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
                
            # 处理新文档
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # 文本分块
            inserting_chunks = get_chunks(
                new_docs=new_docs,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
            )

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # 清理社区报告（需要重新生成）
            await self.community_reports.drop()

            # 实体提取和图构建
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.entity_extraction_func(
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
                using_amazon_bedrock=self.using_amazon_bedrock,
            )
            
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            # 图聚类和社区报告生成
            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, asdict(self)
            )

            # 提交更新
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
            
        finally:
            await self._insert_done()

    async def evaluate_system(self, 
                            questions: List[str],
                            answers: List[str],
                            contexts_list: List[List[str]],
                            complexity_types: List[str] = None,
                            ground_truths: List[str] = None) -> Dict[str, Any]:
        """
        评估系统性能
        使用现代评估器进行全面评估
        """
        if not self.enable_enhanced_features or not hasattr(self, 'modern_evaluator'):
            logger.warning("Modern evaluator not available, skipping evaluation")
            return {"error": "Modern evaluator not available"}
        
        try:
            evaluation_results = self.modern_evaluator.evaluate_batch(
                questions=questions,
                answers=answers,
                contexts_list=contexts_list,
                complexity_types=complexity_types,
                ground_truths=ground_truths
            )
            
            logger.info("System evaluation completed successfully")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"System evaluation failed: {e}")
            return {"error": str(e)}

    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = {
            "basic_info": {
                "working_dir": self.working_dir,
                "enable_enhanced_features": self.enable_enhanced_features,
                "enable_local": self.enable_local,
                "enable_naive_rag": self.enable_naive_rag
            }
        }
        
        # 添加增强功能统计
        if self.enable_enhanced_features:
            if hasattr(self, 'simple_optimizer'):
                stats["optimizer_stats"] = self.simple_optimizer.get_statistics()
            
            if hasattr(self, 'complexity_router'):
                stats["complexity_stats"] = self.complexity_router.get_complexity_stats()
            
            if hasattr(self, 'modern_evaluator'):
                stats["evaluation_summary"] = self.modern_evaluator.get_evaluation_summary()
        
        return stats

    async def _insert_start(self):
        """插入开始回调"""
        tasks = []
        for storage_inst in [self.chunk_entity_relation_graph]:
            if storage_inst is None:
                continue
            tasks.append(storage_inst.index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        """插入完成回调"""
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(storage_inst.index_done_callback())
        await asyncio.gather(*tasks)

    def _get_query_config(self) -> Dict[str, Any]:
        """获取查询配置字典，避免asdict的序列化问题"""
        return {
            "best_model_func": self.best_model_func,
            "cheap_model_func": self.cheap_model_func,
            "convert_response_to_json_func": self.convert_response_to_json_func,
            "embedding_func": self.embedding_func,
            "entity_extraction_func": self.entity_extraction_func,
            "best_model_max_token_size": self.best_model_max_token_size,
            "cheap_model_max_token_size": self.cheap_model_max_token_size,
            "tiktoken_model_name": self.tiktoken_model_name,
            "special_community_report_llm_kwargs": self.special_community_report_llm_kwargs,
            # 添加其他查询函数可能需要的配置
            "embedding_batch_num": self.embedding_batch_num,
            "embedding_func_max_async": self.embedding_func_max_async,
            "query_better_than_threshold": self.query_better_than_threshold,
        }

    async def _query_done(self):
        """查询完成回调"""
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(storage_inst.index_done_callback())
        await asyncio.gather(*tasks)


# 为了向后兼容，保留原有的GraphRAG类
GraphRAG = EnhancedGraphRAG


def create_enhanced_graphrag(**kwargs) -> EnhancedGraphRAG:
    """创建增强版GraphRAG实例"""
    return EnhancedGraphRAG(**kwargs)


def create_basic_graphrag(**kwargs) -> EnhancedGraphRAG:
    """创建基础版GraphRAG实例（禁用增强功能）"""
    kwargs['enable_enhanced_features'] = False
    return EnhancedGraphRAG(**kwargs)
