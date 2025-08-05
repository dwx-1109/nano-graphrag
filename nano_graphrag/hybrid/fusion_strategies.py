"""
融合策略模块

整合所有融合策略，提供统一的接口：
- 基础融合策略（线性融合、RRF融合）
- 加权融合策略
- 动态权重融合策略
- 置信度感知融合策略
"""

import os
import json
import math
import time
import logging
import asyncio
import random
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from .._utils import logger
from .multi_retrieval import RetrievalResult
from .normalization import ScoreNormalizer, normalize_retrieval_scores
from ..evaluation.metrics import RetrievalMetrics
from ..feedback import FeedbackProcessor
from .adaptive_management import AdaptiveWeightManager


class FusionStrategy(ABC):
    """融合策略基类"""
    
    @abstractmethod
    async def fuse(
        self,
        results: Dict[str, RetrievalResult],
        **kwargs
    ) -> RetrievalResult:
        """融合多个检索结果
        
        Args:
            results: 检索结果字典，键为检索器名称，值为检索结果
            **kwargs: 其他参数
            
        Returns:
            融合后的检索结果
        """
        pass


@dataclass
class FusionResult:
    """融合结果数据类"""
    fused_documents: List[Dict[str, Any]]
    fusion_scores: List[float]
    method: str
    weights: Dict[str, float]
    metadata: Dict[str, Any]


class LinearFusion(FusionStrategy):
    """线性融合策略"""
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        name: str = "linear_fusion"
    ):
        """初始化线性融合策略
        
        Args:
            weights: 检索器权重字典，键为检索器名称，值为权重
            name: 融合器名称
        """
        self.weights = weights or {}
        self.name = name
    
    async def fuse(
        self,
        results: Dict[str, RetrievalResult],
        query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        线性融合检索结果
        
        Args:
            results: 检索结果字典
            query: 查询字符串
            metadata: 元数据字典，可以包含权重和top_k
            **kwargs: 其他参数
            
        Returns:
            融合后的检索结果
        """
        start_time = time.time()
        
        if not results:
            return RetrievalResult(
                documents=[],
                retriever_name=self.name,
                query=query or "",
                metrics={"error": "no_results"}
            )
        
        # 获取查询
        if query is None:
            query = next(iter(results.values())).query
        
        # 从metadata或kwargs中获取top_k
        top_k = 10  # 默认值
        if metadata and "top_k" in metadata:
            try:
                top_k = int(metadata["top_k"])
            except (TypeError, ValueError):
                logger.warning(f"无效的top_k值: {metadata['top_k']}, 使用默认值10")
        elif "top_k" in kwargs:
            try:
                top_k = int(kwargs["top_k"])
            except (TypeError, ValueError):
                logger.warning(f"无效的top_k值: {kwargs['top_k']}, 使用默认值10")
        
        # 收集所有文档
        all_documents = {}
        document_sources = defaultdict(list)
        
        for retriever_name, result in results.items():
            weight = self.weights.get(retriever_name, 1.0)
            
            for i, doc in enumerate(result.documents):
                doc_id = doc.get('id', f"{retriever_name}_{i}")
                score = doc.get('score', 0.0)
                
                if doc_id not in all_documents:
                    all_documents[doc_id] = {
                        'id': doc_id,
                        'content': doc.get('content', ''),
                        'metadata': doc.get('metadata', {}),
                        'sources': [],
                        'fusion_score': 0.0
                    }
                
                all_documents[doc_id]['fusion_score'] += weight * score
                all_documents[doc_id]['sources'].append({
                    'source': retriever_name,
                    'score': score,
                    'weight': weight,
                    'rank': i
                })
                document_sources[doc_id].append(retriever_name)
        
        # 按融合分数排序
        sorted_documents = sorted(
            all_documents.values(),
            key=lambda x: x['fusion_score'],
            reverse=True
        )[:top_k]
        
        # 计算融合分数
        fusion_scores = [doc['fusion_score'] for doc in sorted_documents]
        
        # 计算指标
        metrics = {
            "fusion_time": time.time() - start_time,
            "num_sources": len(results),
            "num_documents": len(sorted_documents),
            "method": "linear_fusion",
            "weights": self.weights
        }
        
        return RetrievalResult(
            documents=sorted_documents,
            retriever_name=self.name,
            query=query,
            metrics=metrics
        )


class RRFFusion(FusionStrategy):
    """倒数排名融合策略（Reciprocal Rank Fusion）"""
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        k: float = 60.0,
        name: str = "rrf_fusion"
    ):
        """初始化RRF融合策略
        
        Args:
            weights: 检索器权重字典
            k: RRF参数，控制排名的影响程度
            name: 融合器名称
        """
        self.weights = weights or {}
        self.k = k
        self.name = name
    
    async def fuse(
        self,
        results: Dict[str, RetrievalResult],
        query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        RRF融合检索结果
        
        Args:
            results: 检索结果字典
            query: 查询字符串
            metadata: 元数据字典
            **kwargs: 其他参数
            
        Returns:
            融合后的检索结果
        """
        start_time = time.time()
        
        if not results:
            return RetrievalResult(
                documents=[],
                retriever_name=self.name,
                query=query or "",
                metrics={"error": "no_results"}
            )
        
        # 获取查询
        if query is None:
            query = next(iter(results.values())).query
        
        # 从metadata或kwargs中获取top_k
        top_k = 10  # 默认值
        if metadata and "top_k" in metadata:
            try:
                top_k = int(metadata["top_k"])
            except (TypeError, ValueError):
                logger.warning(f"无效的top_k值: {metadata['top_k']}, 使用默认值10")
        elif "top_k" in kwargs:
            try:
                top_k = int(kwargs["top_k"])
            except (TypeError, ValueError):
                logger.warning(f"无效的top_k值: {kwargs['top_k']}, 使用默认值10")
        
        # 收集所有文档
        all_documents = {}
        document_sources = defaultdict(list)
        
        for retriever_name, result in results.items():
            weight = self.weights.get(retriever_name, 1.0)
            
            for i, doc in enumerate(result.documents):
                doc_id = doc.get('id', f"{retriever_name}_{i}")
                rank = i + 1  # 排名从1开始
                
                # RRF公式：RRF(d) = Σ(weight / (k + rank))
                rrf_score = weight / (self.k + rank)
                
                if doc_id not in all_documents:
                    all_documents[doc_id] = {
                        'id': doc_id,
                        'content': doc.get('content', ''),
                        'metadata': doc.get('metadata', {}),
                        'sources': [],
                        'fusion_score': 0.0
                    }
                
                all_documents[doc_id]['fusion_score'] += rrf_score
                all_documents[doc_id]['sources'].append({
                    'source': retriever_name,
                    'rank': rank,
                    'weight': weight,
                    'rrf_score': rrf_score
                })
                document_sources[doc_id].append(retriever_name)
        
        # 按融合分数排序
        sorted_documents = sorted(
            all_documents.values(),
            key=lambda x: x['fusion_score'],
            reverse=True
        )[:top_k]
        
        # 计算融合分数
        fusion_scores = [doc['fusion_score'] for doc in sorted_documents]
        
        # 计算指标
        metrics = {
            "fusion_time": time.time() - start_time,
            "num_sources": len(results),
            "num_documents": len(sorted_documents),
            "method": "rrf_fusion",
            "k": self.k,
            "weights": self.weights
        }
        
        return RetrievalResult(
            documents=sorted_documents,
            retriever_name=self.name,
            query=query,
            metrics=metrics
        )


class WeightedLinearFusion:
    """线性加权融合器（兼容旧接口）"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        初始化线性加权融合器
        
        Args:
            weights: 各检索源的权重字典
        """
        self.weights = weights or {}
        
    def fuse(self, retrieval_results: Dict[str, List[Dict[str, Any]]], 
             normalization_method: str = "minmax") -> FusionResult:
        """
        执行线性加权融合：S(d) = Σ(w_i * s_i(d))
        
        Args:
            retrieval_results: 各检索源的结果字典
            normalization_method: 归一化方法
            
        Returns:
            FusionResult: 融合结果
        """
        # 归一化各检索源的分数
        normalized_results = {}
        for source, results in retrieval_results.items():
            if results:
                normalized_results[source] = normalize_retrieval_scores(
                    results, method=normalization_method
                )
        
        # 收集所有文档
        all_documents = {}
        document_sources = defaultdict(list)
        
        for source, results in normalized_results.items():
            weight = self.weights.get(source, 1.0)
            for i, result in enumerate(results):
                doc_id = result.get('id', f"{source}_{i}")
                score = result.get('score', 0.0)
                
                if doc_id not in all_documents:
                    all_documents[doc_id] = {
                        'id': doc_id,
                        'content': result.get('content', ''),
                        'metadata': result.get('metadata', {}),
                        'sources': [],
                        'fusion_score': 0.0
                    }
                
                all_documents[doc_id]['fusion_score'] += weight * score
                all_documents[doc_id]['sources'].append({
                    'source': source,
                    'score': score,
                    'weight': weight,
                    'rank': i
                })
                document_sources[doc_id].append(source)
        
        # 按融合分数排序
        sorted_documents = sorted(
            all_documents.values(),
            key=lambda x: x['fusion_score'],
            reverse=True
        )
        
        fusion_scores = [doc['fusion_score'] for doc in sorted_documents]
        
        return FusionResult(
            fused_documents=sorted_documents,
            fusion_scores=fusion_scores,
            method="weighted_linear",
            weights=self.weights,
            metadata={
                "normalization_method": normalization_method,
                "num_sources": len(retrieval_results),
                "num_documents": len(sorted_documents)
            }
        )


class WeightedRRFFusion:
    """加权RRF融合器（兼容旧接口）"""
    
    def __init__(self, k: float = 60.0, weights: Optional[Dict[str, float]] = None):
        """
        初始化加权RRF融合器
        
        Args:
            k: RRF参数
            weights: 各检索源的权重字典
        """
        self.k = k
        self.weights = weights or {}
        
    def fuse(self, retrieval_results: Dict[str, List[Dict[str, Any]]]) -> FusionResult:
        """
        执行加权RRF融合
        
        Args:
            retrieval_results: 各检索源的结果字典
            
        Returns:
            FusionResult: 融合结果
        """
        # 收集所有文档
        all_documents = {}
        document_sources = defaultdict(list)
        
        for source, results in retrieval_results.items():
            weight = self.weights.get(source, 1.0)
            for i, result in enumerate(results):
                doc_id = result.get('id', f"{source}_{i}")
                rank = i + 1  # 排名从1开始
                
                # RRF公式：RRF(d) = Σ(weight / (k + rank))
                rrf_score = weight / (self.k + rank)
                
                if doc_id not in all_documents:
                    all_documents[doc_id] = {
                        'id': doc_id,
                        'content': result.get('content', ''),
                        'metadata': result.get('metadata', {}),
                        'sources': [],
                        'fusion_score': 0.0
                    }
                
                all_documents[doc_id]['fusion_score'] += rrf_score
                all_documents[doc_id]['sources'].append({
                    'source': source,
                    'rank': rank,
                    'weight': weight,
                    'rrf_score': rrf_score
                })
                document_sources[doc_id].append(source)
        
        # 按融合分数排序
        sorted_documents = sorted(
            all_documents.values(),
            key=lambda x: x['fusion_score'],
            reverse=True
        )
        
        fusion_scores = [doc['fusion_score'] for doc in sorted_documents]
        
        return FusionResult(
            fused_documents=sorted_documents,
            fusion_scores=fusion_scores,
            method="weighted_rrf",
            weights=self.weights,
            metadata={
                "k": self.k,
                "num_sources": len(retrieval_results),
                "num_documents": len(sorted_documents)
            }
        )


class DynamicWeightFusion(FusionStrategy):
    """动态权重融合策略"""
    
    def __init__(
        self,
        retriever_ids: List[str] = None,
        weight_func: Callable[[str, float], Dict[str, float]] = None,
        fusion_strategy: FusionStrategy = None,
        name: str = "dynamic_fusion"
    ):
        """
        初始化动态权重融合策略
        
        Args:
            retriever_ids: 检索器ID列表
            weight_func: 权重计算函数
            fusion_strategy: 基础融合策略
            name: 融合器名称
        """
        self.retriever_ids = retriever_ids or []
        self.weight_func = weight_func
        self.fusion_strategy = fusion_strategy or LinearFusion()
        self.name = name
    
    async def fuse(
        self,
        results: Dict[str, RetrievalResult],
        query: str,
        confidence: float = 0.5,
        top_k: int = 10,
        **kwargs
    ) -> RetrievalResult:
        """
        动态权重融合
        
        Args:
            results: 检索结果字典
            query: 查询字符串
            confidence: 置信度
            top_k: 返回文档数量
            **kwargs: 其他参数
            
        Returns:
            融合后的检索结果
        """
        if not results:
            return RetrievalResult(
                documents=[],
                retriever_name=self.name,
                query=query,
                metrics={"error": "no_results"}
            )
        
        # 计算动态权重
        if self.weight_func:
            weights = self.weight_func(query, confidence)
        else:
            # 默认权重计算
            weights = self._default_weight_calculation(query, confidence, results)
        
        # 使用基础融合策略
        self.fusion_strategy.weights = weights
        result = await self.fusion_strategy.fuse(results, query=query, top_k=top_k, **kwargs)
        
        # 更新指标
        result.metrics.update({
            "method": "dynamic_fusion",
            "confidence": confidence,
            "dynamic_weights": weights
        })
        
        return result
    
    def _default_weight_calculation(
        self,
        query: str,
        confidence: float,
        results: Dict[str, RetrievalResult]
    ) -> Dict[str, float]:
        """默认权重计算"""
        # 简单的基于置信度的权重分配
        num_retrievers = len(results)
        if num_retrievers == 0:
            return {}
        
        # 高置信度时更倾向于向量检索
        if confidence > 0.7:
            weights = {"vector": 0.6, "bm25": 0.3, "graph": 0.1}
        elif confidence > 0.4:
            weights = {"vector": 0.4, "bm25": 0.4, "graph": 0.2}
        else:
            weights = {"vector": 0.2, "bm25": 0.3, "graph": 0.5}
        
        # 只返回实际存在的检索器权重
        return {k: v for k, v in weights.items() if k in results}


class ConfidenceAwareFusion(DynamicWeightFusion):
    """置信度感知融合策略"""
    
    def __init__(
        self,
        retriever_ids: List[str] = None,
        confidence_thresholds: Dict[str, float] = None,
        complexity_weights: Dict[str, Dict[str, float]] = None,
        min_weight: float = 0.01
    ):
        """
        初始化置信度感知融合策略
        
        Args:
            retriever_ids: 检索器ID列表
            confidence_thresholds: 置信度阈值
            complexity_weights: 复杂度权重配置
            min_weight: 最小权重
        """
        super().__init__(retriever_ids=retriever_ids, name="confidence_aware_fusion")
        
        self.confidence_thresholds = confidence_thresholds or {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }
        
        self.complexity_weights = complexity_weights or {
            "zero_hop": {"llm_only": 0.9, "vector": 0.1},
            "one_hop": {"vector": 0.6, "bm25": 0.3, "graph": 0.1},
            "multi_hop": {"graph": 0.7, "vector": 0.2, "bm25": 0.1}
        }
        
        self.min_weight = min_weight
    
    async def fuse(
        self,
        results: Dict[str, RetrievalResult],
        query: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        置信度感知融合
        
        Args:
            results: 检索结果字典
            query: 查询字符串
            metadata: 元数据
            **kwargs: 其他参数
            
        Returns:
            融合后的检索结果
        """
        if not results:
            return RetrievalResult([], self.name, query)
        
        # 获取复杂度和置信度
        complexity = metadata.get("complexity", "one_hop") if metadata else "one_hop"
        confidence = metadata.get("confidence", 0.5) if metadata else 0.5
        
        # 计算权重
        weights = self._calculate_weights(query, results, complexity, confidence, metadata, **kwargs)
        
        # 应用权重
        fused_result = self._apply_weights(results, weights, query)
        
        return fused_result
    
    def _calculate_weights(
        self,
        query: str,
        results: Dict[str, RetrievalResult],
        complexity: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """计算置信度感知权重"""
        # 获取复杂度对应的基础权重
        base_weights = self.complexity_weights.get(complexity, {})
        
        # 根据置信度调整权重
        if confidence > self.confidence_thresholds["high"]:
            # 高置信度：更倾向于主要检索器
            adjusted_weights = {}
            for retriever, weight in base_weights.items():
                if retriever in results:
                    if weight > 0.5:
                        adjusted_weights[retriever] = weight * 1.2
                    else:
                        adjusted_weights[retriever] = weight * 0.8
        elif confidence < self.confidence_thresholds["low"]:
            # 低置信度：更均衡的权重分配
            adjusted_weights = {}
            for retriever, weight in base_weights.items():
                if retriever in results:
                    adjusted_weights[retriever] = 1.0 / len(results)
        else:
            # 中等置信度：保持基础权重
            adjusted_weights = {k: v for k, v in base_weights.items() if k in results}
        
        # 归一化权重
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        else:
            # 如果所有权重都为0，使用均匀分配
            normalized_weights = {k: 1.0 / len(results) for k in results.keys()}
        
        # 确保最小权重
        for retriever in results.keys():
            if retriever not in normalized_weights:
                normalized_weights[retriever] = self.min_weight
        
        return normalized_weights
    
    def _apply_weights(
        self,
        results: Dict[str, RetrievalResult],
        weights: Dict[str, float],
        query: str
    ) -> RetrievalResult:
        """应用权重到检索结果"""
        # 收集所有文档
        all_documents = {}
        
        for retriever_name, result in results.items():
            weight = weights.get(retriever_name, 0.0)
            
            for i, doc in enumerate(result.documents):
                doc_id = doc.get('id', f"{retriever_name}_{i}")
                score = doc.get('score', 0.0)
                
                if doc_id not in all_documents:
                    all_documents[doc_id] = {
                        'id': doc_id,
                        'content': doc.get('content', ''),
                        'metadata': doc.get('metadata', {}),
                        'sources': [],
                        'fusion_score': 0.0
                    }
                
                all_documents[doc_id]['fusion_score'] += weight * score
                all_documents[doc_id]['sources'].append({
                    'source': retriever_name,
                    'score': score,
                    'weight': weight,
                    'rank': i
                })
        
        # 按融合分数排序
        sorted_documents = sorted(
            all_documents.values(),
            key=lambda x: x['fusion_score'],
            reverse=True
        )
        
        # 计算指标
        metrics = {
            "method": "confidence_aware_fusion",
            "weights": weights,
            "num_sources": len(results),
            "num_documents": len(sorted_documents)
        }
        
        return RetrievalResult(
            documents=sorted_documents,
            retriever_name=self.name,
            query=query,
            metrics=metrics
        )


class AdaptiveFusionStrategy(FusionStrategy):
    """自适应融合策略"""
    
    def __init__(
        self,
        weight_manager: Optional[AdaptiveWeightManager] = None,
        feedback_processor: Optional[FeedbackProcessor] = None,
        linear_fusion: Optional[LinearFusion] = None,
        rrf_fusion: Optional[RRFFusion] = None,
        evaluation_metrics: Optional[RetrievalMetrics] = None
    ):
        """
        初始化自适应融合策略
        
        Args:
            weight_manager: 自适应权重管理器
            feedback_processor: 反馈处理器
            linear_fusion: 线性融合器
            rrf_fusion: RRF融合器
            evaluation_metrics: 评估指标
        """
        self.weight_manager = weight_manager
        self.feedback_processor = feedback_processor
        self.linear_fusion = linear_fusion or LinearFusion()
        self.rrf_fusion = rrf_fusion or RRFFusion()
        self.evaluation_metrics = evaluation_metrics
        
        # 使用统计
        self.usage_stats = {
            "linear_fusion": 0,
            "rrf_fusion": 0,
            "total_queries": 0
        }
    
    async def fuse(
        self,
        results: Dict[str, RetrievalResult],
        query: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        自适应融合
        
        Args:
            results: 检索结果字典
            query: 查询字符串
            metadata: 元数据
            **kwargs: 其他参数
            
        Returns:
            融合后的检索结果
        """
        if not results:
            return RetrievalResult([], "adaptive_fusion", query)
        
        # 获取复杂度和置信度
        complexity = metadata.get("complexity") if metadata else None
        confidence = metadata.get("confidence") if metadata else None
        
        # 选择融合策略
        strategy = self._select_fusion_strategy(query, complexity, confidence)
        
        # 计算权重
        weights = {}
        if self.weight_manager:
            weights, _ = await self.weight_manager.calculate_weights(
                query, complexity, confidence
            )
        
        # 执行融合
        if strategy == "linear":
            self.linear_fusion.weights = weights
            result = await self.linear_fusion.fuse(results, query=query, metadata=metadata, **kwargs)
            self.usage_stats["linear_fusion"] += 1
        else:
            self.rrf_fusion.weights = weights
            result = await self.rrf_fusion.fuse(results, query=query, metadata=metadata, **kwargs)
            self.usage_stats["rrf_fusion"] += 1
        
        self.usage_stats["total_queries"] += 1
        
        # 记录使用情况
        self._record_usage(query, complexity, confidence, strategy, weights)
        
        # 更新指标
        result.metrics.update({
            "method": "adaptive_fusion",
            "strategy": strategy,
            "weights": weights
        })
        
        return result
    
    def _select_fusion_strategy(
        self,
        query: str,
        complexity: Optional[str],
        confidence: Optional[float]
    ) -> str:
        """选择融合策略"""
        # 基于复杂度的策略选择
        if complexity == "zero_hop":
            return "linear"  # 简单查询使用线性融合
        elif complexity == "multi_hop":
            return "rrf"     # 复杂查询使用RRF融合
        else:
            # 基于置信度的策略选择
            if confidence and confidence > 0.7:
                return "linear"  # 高置信度使用线性融合
            else:
                return "rrf"     # 低置信度使用RRF融合
    
    def _record_usage(
        self,
        query: str,
        complexity: Optional[str],
        confidence: Optional[float],
        strategy: str,
        weights: Dict[str, float]
    ) -> None:
        """记录使用情况"""
        # 这里可以记录到日志或数据库
        logger.debug(f"自适应融合 - 查询: {query[:50]}..., 复杂度: {complexity}, "
                    f"置信度: {confidence:.3f}, 策略: {strategy}")
    
    def process_feedback(
        self,
        query: str,
        fused_result: RetrievalResult,
        feedback_data: Dict[str, Any]
    ) -> None:
        """处理反馈"""
        if self.feedback_processor:
            self.feedback_processor.process_feedback(query, fused_result, feedback_data)
        
        if self.weight_manager:
            self.weight_manager.update_from_feedback(query, feedback_data)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        stats = self.usage_stats.copy()
        if self.usage_stats["total_queries"] > 0:
            stats["linear_fusion_ratio"] = (
                self.usage_stats["linear_fusion"] / self.usage_stats["total_queries"]
            )
            stats["rrf_fusion_ratio"] = (
                self.usage_stats["rrf_fusion"] / self.usage_stats["total_queries"]
            )
        return stats


# 工具函数
def create_confidence_aware_weights(confidence: float, 
                                   query_type: str,
                                   base_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """创建置信度感知权重"""
    if base_weights is None:
        base_weights = {"vector": 0.4, "bm25": 0.3, "graph": 0.3}
    
    # 根据置信度调整权重
    if confidence > 0.8:
        # 高置信度：更倾向于向量检索
        adjusted_weights = {
            "vector": base_weights.get("vector", 0.4) * 1.5,
            "bm25": base_weights.get("bm25", 0.3) * 0.8,
            "graph": base_weights.get("graph", 0.3) * 0.7
        }
    elif confidence < 0.3:
        # 低置信度：更均衡的权重
        adjusted_weights = {
            "vector": 0.33,
            "bm25": 0.33,
            "graph": 0.34
        }
    else:
        # 中等置信度：保持基础权重
        adjusted_weights = base_weights.copy()
    
    # 归一化
    total = sum(adjusted_weights.values())
    if total > 0:
        return {k: v / total for k, v in adjusted_weights.items()}
    else:
        return base_weights


def adaptive_fusion(retrieval_results: Dict[str, List[Dict[str, Any]]],
                   confidence: float,
                   query_type: str,
                   fusion_method: str = "auto") -> FusionResult:
    """自适应融合函数（兼容旧接口）"""
    if fusion_method == "auto":
        if confidence > 0.7:
            fusion_method = "weighted_linear"
        else:
            fusion_method = "weighted_rrf"
    
    if fusion_method == "weighted_linear":
        weights = create_confidence_aware_weights(confidence, query_type)
        fusion = WeightedLinearFusion(weights)
        return fusion.fuse(retrieval_results)
    elif fusion_method == "weighted_rrf":
        weights = create_confidence_aware_weights(confidence, query_type)
        fusion = WeightedRRFFusion(weights=weights)
        return fusion.fuse(retrieval_results)
    else:
        raise ValueError(f"不支持的融合方法: {fusion_method}")


class FusionMethodComparator:
    """融合方法比较器"""
    
    def __init__(self):
        self.fusion_methods = {
            "linear": WeightedLinearFusion(),
            "rrf": WeightedRRFFusion(),
            "confidence_aware": lambda: adaptive_fusion
        }
    
    def compare_methods(self, retrieval_results: Dict[str, List[Dict[str, Any]]], 
                       weights: Optional[Dict[str, float]] = None) -> Dict[str, FusionResult]:
        """比较不同融合方法的结果"""
        results = {}
        
        for method_name, fusion_method in self.fusion_methods.items():
            try:
                if method_name == "confidence_aware":
                    # 使用默认置信度和查询类型
                    result = fusion_method()(retrieval_results, 0.5, "general")
                else:
                    fusion = fusion_method
                    if weights:
                        fusion.weights = weights
                    result = fusion.fuse(retrieval_results)
                
                results[method_name] = result
            except Exception as e:
                logger.error(f"融合方法 {method_name} 执行失败: {e}")
        
        return results
    
    def analyze_diversity(self, fusion_results: Dict[str, FusionResult]) -> Dict[str, float]:
        """分析不同融合方法的多样性"""
        diversity_scores = {}
        
        for method_name, result in fusion_results.items():
            if not result.fused_documents:
                diversity_scores[method_name] = 0.0
                continue
            
            # 计算文档来源的多样性
            sources = set()
            for doc in result.fused_documents:
                for source_info in doc.get('sources', []):
                    sources.add(source_info.get('source', ''))
            
            diversity_scores[method_name] = len(sources) / len(result.fused_documents)
        
        return diversity_scores
    
    def analyze_overlap(self, fusion_results: Dict[str, FusionResult], 
                       top_k: int = 10) -> Dict[str, Dict[str, float]]:
        """分析不同融合方法结果的重叠度"""
        overlap_matrix = {}
        
        method_names = list(fusion_results.keys())
        
        for i, method1 in enumerate(method_names):
            overlap_matrix[method1] = {}
            
            for j, method2 in enumerate(method_names):
                if i == j:
                    overlap_matrix[method1][method2] = 1.0
                else:
                    # 计算重叠度
                    docs1 = set(doc['id'] for doc in fusion_results[method1].fused_documents[:top_k])
                    docs2 = set(doc['id'] for doc in fusion_results[method2].fused_documents[:top_k])
                    
                    if len(docs1) == 0 or len(docs2) == 0:
                        overlap = 0.0
                    else:
                        overlap = len(docs1.intersection(docs2)) / len(docs1.union(docs2))
                    
                    overlap_matrix[method1][method2] = overlap
        
        return overlap_matrix 