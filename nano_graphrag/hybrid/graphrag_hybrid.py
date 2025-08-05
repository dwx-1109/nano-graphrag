"""
GraphRAG混合检索扩展

扩展GraphRAG类，添加并行检索和结果融合功能
"""

import os
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict

from ..graphrag import GraphRAG
from ..base import QueryParam
from .._utils import logger

from .multi_retrieval import ParallelRetriever, RetrievalResult
from .fusion_strategies import LinearFusion, RRFFusion, DynamicWeightFusion
from .adaptive_management import ConfidenceAwareRouter


@dataclass
class HybridQueryParam(QueryParam):
    """混合检索参数"""
    # 并行检索参数
    parallel: bool = True
    parallel_timeout: float = 30.0
    parallel_max_concurrency: int = 5
    
    # 融合参数
    fusion_method: str = "linear"  # "linear", "rrf", "dynamic"
    rrf_k: float = 60.0
    
    # 动态权重参数
    dynamic_weight_gamma: float = 2.0
    dynamic_weight_alpha_zero: float = 0.9
    dynamic_weight_alpha_one: float = 0.6
    dynamic_weight_alpha_multi: float = 0.3
    
    # 混合检索路径权重
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "naive": 1.0,
            "bm25": 0.7,
            "local": 0.5,
            "global": 0.8,
            "llm_only": 0.3
        }
    )
    
    # 路由参数
    use_confidence_router: bool = True
    calibration_path: Optional[str] = None
    
    # 混合查询特有参数
    full_hybrid_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "naive": 1.0,
            "bm25": 0.7,
            "graph": 0.8
        }
    )
    # 混合图查询特有参数
    hybrid_graph_balance_factor: float = 0.5  # 0-1之间，1表示完全依赖local


class GraphRAGHybrid(GraphRAG):
    """GraphRAG混合检索扩展类"""
    
    def __init__(self, *args, **kwargs):
        """初始化GraphRAG混合检索扩展类"""
        super().__init__(*args, **kwargs)
        
        # 初始化并行检索控制器
        self.parallel_retriever = ParallelRetriever(
            retrievers={},
            max_concurrency=5,
            timeout=30.0
        )
        
        # 初始化置信度感知路由器
        router_kwargs = self.router_kwargs.copy()
        router_kwargs["model_path"] = self.model_path
        
        if self.router_cls == ConfidenceAwareRouter:
            self.confidence_router = self.router
        else:
            self.confidence_router = ConfidenceAwareRouter(**router_kwargs)
        
        # 注册检索器
        self._register_retrievers()
    
    def _register_retrievers(self):
        """注册内置检索器"""
        # 向量检索
        if self.enable_naive_rag:
            self.parallel_retriever.add_retriever(
                "naive",
                self._naive_retriever
            )
        
        # BM25检索
        if self.enable_bm25:
            self.parallel_retriever.add_retriever(
                "bm25",
                self._bm25_retriever
            )
        
        # 本地图检索
        if self.enable_local:
            self.parallel_retriever.add_retriever(
                "local",
                self._local_retriever
            )
        
        # 全局图检索
        self.parallel_retriever.add_retriever(
            "global",
            self._global_retriever
        )
        
        # 直接LLM
        self.parallel_retriever.add_retriever(
            "llm_only",
            self._llm_only_retriever
        )
    
    async def _naive_retriever(self, query: str, top_k: int = 10) -> RetrievalResult:
        """向量检索器"""
        from .._op import naive_query
        
        param = QueryParam(
            mode="naive",
            only_need_context=True,
            top_k=top_k
        )
        
        # 执行检索，只返回上下文
        documents = await naive_query(
            query,
            self.chunks_vdb,
            self.text_chunks,
            param,
            asdict(self),
            return_context=True
        )
        
        # 处理结果格式
        docs = []
        if isinstance(documents, str):
            chunks = documents.split("--New Chunk--\n")
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    docs.append({
                        "id": f"naive-{i}",
                        "text": chunk.strip(),
                        "score": 1.0 - (i / len(chunks)) if len(chunks) > 1 else 1.0
                    })
        
        return RetrievalResult(
            documents=docs,
            retriever_name="naive",
            query=query
        )
    
    async def _bm25_retriever(self, query: str, top_k: int = 10) -> RetrievalResult:
        """BM25检索器"""
        from .._op import bm25_query
        
        param = QueryParam(
            mode="bm25",
            only_need_context=True,
            top_k=top_k
        )
        
        # 执行检索，只返回上下文
        documents = await bm25_query(
            query,
            self.bm25_store,
            self.text_chunks,
            param,
            asdict(self),
            return_context=True
        )
        
        # 处理结果格式
        docs = []
        if isinstance(documents, str):
            chunks = documents.split("--New Chunk--\n")
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    docs.append({
                        "id": f"bm25-{i}",
                        "text": chunk.strip(),
                        "score": 1.0 - (i / len(chunks)) if len(chunks) > 1 else 1.0
                    })
        
        return RetrievalResult(
            documents=docs,
            retriever_name="bm25",
            query=query
        )
    
    async def _local_retriever(self, query: str, top_k: int = 10) -> RetrievalResult:
        """本地图检索器"""
        from .._op import local_query
        
        param = QueryParam(
            mode="local",
            only_need_context=True,
            top_k=top_k
        )
        
        # 执行检索，只返回上下文
        context = await local_query(
            query,
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.community_reports,
            self.text_chunks,
            param,
            asdict(self),
            return_context=True
        )
        
        # 处理结果格式
        docs = []
        if isinstance(context, str):
            sections = context.split("-----")
            # 优先使用Sources部分的内容
            sources_section = ""
            for section in sections:
                if "Sources" in section:
                    sources_section = section
                    break
            
            if sources_section:
                # 提取CSV内容
                import re
                csv_match = re.search(r'```csv(.*?)```', sources_section, re.DOTALL)
                if csv_match:
                    csv_content = csv_match.group(1).strip()
                    lines = csv_content.split('\n')[1:]  # 跳过标题行
                    for i, line in enumerate(lines):
                        parts = line.split(',', 1)
                        if len(parts) > 1:
                            doc_id = f"local-{parts[0].strip()}"
                            text = parts[1].strip().strip('"')
                            docs.append({
                                "id": doc_id,
                                "text": text,
                                "score": 1.0 - (i / len(lines)) if len(lines) > 1 else 1.0
                            })
            
            # 如果Sources部分没有内容，使用整个上下文
            if not docs:
                docs.append({
                    "id": "local-context",
                    "text": context,
                    "score": 1.0
                })
        
        return RetrievalResult(
            documents=docs,
            retriever_name="local",
            query=query
        )
    
    async def _global_retriever(self, query: str, top_k: int = 10) -> RetrievalResult:
        """全局图检索器"""
        from .._op import global_query
        
        param = QueryParam(
            mode="global",
            only_need_context=True,
            top_k=top_k
        )
        
        # 执行检索，只返回上下文
        context = await global_query(
            query,
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.community_reports,
            self.text_chunks,
            param,
            asdict(self),
            return_context=True
        )
        
        # 处理结果格式
        docs = []
        if isinstance(context, str):
            analysts = context.split("----Analyst")
            for i, analyst in enumerate(analysts):
                if not analyst.strip():
                    continue
                docs.append({
                    "id": f"global-{i}",
                    "text": analyst.strip(),
                    "score": 1.0 - (i / len(analysts)) if len(analysts) > 1 else 1.0
                })
        
        return RetrievalResult(
            documents=docs,
            retriever_name="global",
            query=query
        )
    
    async def _llm_only_retriever(self, query: str, top_k: int = 10) -> RetrievalResult:
        """直接LLM检索器（无检索）"""
        # LLM-only模式不需要检索，但为了兼容接口，返回空结果
        return RetrievalResult(
            documents=[],
            retriever_name="llm_only",
            query=query
        )
    
    def _create_fusion_strategy(self, param: HybridQueryParam):
        """创建融合策略"""
        if param.fusion_method == "rrf":
            return RRFFusion(
                weights=param.weights,
                k=param.rrf_k
            )
        elif param.fusion_method == "dynamic":
            # 创建动态权重计算函数
            def weight_func(query, confidence):
                # 获取复杂度预测（同步方式）
                complexity = self.router._rule_based_complexity_sync(query)["complexity"]
                
                # 计算动态权重
                return self.confidence_router.calculate_dynamic_weights(
                    query, confidence, complexity
                )
            
            # 使用动态权重融合策略
            return DynamicWeightFusion(
                weight_func=weight_func,
                fusion_strategy=LinearFusion()
            )
        else:  # 默认使用线性融合
            return LinearFusion(weights=param.weights)
    
    async def hybrid_query(
        self,
        query: str,
        param: Optional[HybridQueryParam] = None
    ) -> str:
        """混合检索查询
        
        Args:
            query: 查询文本
            param: 混合检索参数
            
        Returns:
            回答文本
        """
        start_time = time.time()
        
        # 创建默认参数
        if param is None:
            param = HybridQueryParam()
        
        # 获取复杂度和置信度
        complexity_result = await self.confidence_router.predict_complexity_with_calibration(query)
        complexity = complexity_result["complexity"]
        confidence = complexity_result["confidence"]
        
        # 更新并行检索器配置
        self.parallel_retriever.timeout = param.parallel_timeout
        self.parallel_retriever.max_concurrency = param.parallel_max_concurrency
        
        # 创建融合策略
        fusion_strategy = self._create_fusion_strategy(param)
        
        # 执行并行检索
        results = await self.parallel_retriever.retrieve_all(
            query=query,
            top_k=param.top_k
        )
        
        # 如果只需要上下文，返回融合结果
        if param.only_need_context:
            fused_result = await fusion_strategy.fuse(
                results=results,
                query=query,
                confidence=confidence,
                top_k=param.top_k
            )
            
            # 格式化为文本
            return "\n\n".join([
                f"---- 文档 {i+1} (得分: {doc['score']:.4f}) ----\n{doc['text']}"
                for i, doc in enumerate(fused_result.documents)
            ])
        
        # 准备LLM提示
        use_model_func = self.best_model_func
        
        # 融合检索结果
        fused_result = await fusion_strategy.fuse(
            results=results,
            query=query,
            confidence=confidence,
            top_k=param.top_k
        )
        
        # 构建上下文
        context = "\n\n".join([
            f"---- 文档 {i+1} ----\n{doc['text']}"
            for i, doc in enumerate(fused_result.documents)
        ])
        
        # 生成系统提示
        from ..answer_generation.prompts import PROMPTS
        sys_prompt_temp = PROMPTS["naive_rag_response"]
        sys_prompt = sys_prompt_temp.format(
            content_data=context, 
            response_type=param.response_type
        )
        
        # 生成回答
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"混合检索查询完成，耗时{elapsed_time:.4f}秒")
        
        return response
    
    def create_hybrid_query_param(
        self, 
        query: str, 
        **kwargs
    ) -> HybridQueryParam:
        """创建混合检索参数
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            混合检索参数
        """
        # 根据查询复杂度动态创建参数
        loop = asyncio.get_event_loop()
        complexity_result = loop.run_until_complete(
            self.confidence_router.predict_complexity_with_calibration(query)
        )
        
        complexity = complexity_result["complexity"]
        confidence = complexity_result["confidence"]
        
        # 创建基础参数
        param = HybridQueryParam(**kwargs)
        
        # 根据复杂度调整参数
        if complexity == "zero_hop":
            param.mode = "llm_only"
        elif complexity == "one_hop":
            param.mode = "naive"  # 或 "bm25"
            # 调整权重
            param.weights = {
                "naive": 1.0,
                "bm25": 0.8,
                "local": 0.3,
                "global": 0.2,
                "llm_only": 0.1
            }
        elif complexity == "multi_hop":
            param.mode = "local"  # 或 "global"
            # 调整权重
            param.weights = {
                "naive": 0.4,
                "bm25": 0.3,
                "local": 0.8,
                "global": 1.0,
                "llm_only": 0.1
            }
        
        return param 