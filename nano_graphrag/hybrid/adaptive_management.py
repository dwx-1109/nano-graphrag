"""
自适应管理模块

整合自适应权重管理和路由功能：
- 自适应权重管理器
- 置信度感知路由器
- 反馈驱动的融合
"""

import os
import json
import math
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from ..complexity.router import ComplexityAwareRouter
from ..optimizer import AdaptiveLearner, ThompsonSamplingOptimizer, UCBOptimizer, EvolutionaryOptimizer
from ..evaluation.metrics import RetrievalMetrics
from ..feedback import FeedbackProcessor
from .._utils import logger


class AdaptiveWeightManager:
    """自适应权重管理器（重构：严格主辅路径gamma分配+配置化）"""
    
    # 配置主辅路径与检索源映射
    retriever_mapping = {
        "zero_hop": {"main": ["llm_only"], "aux": ["vector", "bm25"]},
        "one_hop": {"main": ["vector", "bm25"], "aux": ["local"]},
        "multi_hop": {"main": ["local", "global"], "aux": ["vector", "bm25"]}
    }
    # 各类型基础alpha
    base_alpha = {"zero_hop": 0.9, "one_hop": 0.7, "multi_hop": 0.6}
    gamma = 2.0

    def __init__(
        self,
        retriever_ids: List[str],
        optimizer_type: str = "adaptive",
        confidence_threshold: float = 0.6,
        min_weight: float = 0.01,
        save_path: Optional[str] = None,
        router: Optional[ComplexityAwareRouter] = None,
        feedback_processor: Optional[FeedbackProcessor] = None,
        gamma: float = None,
        base_alpha: Dict[str, float] = None,
        retriever_mapping: Dict[str, Dict[str, List[str]]] = None
    ):
        self.retriever_ids = retriever_ids
        self.optimizer_type = optimizer_type
        self.confidence_threshold = confidence_threshold
        self.min_weight = min_weight
        self.save_path = save_path
        self.router = router
        self.feedback_processor = feedback_processor
        if gamma is not None:
            self.gamma = gamma
        if base_alpha is not None:
            self.base_alpha = base_alpha
        if retriever_mapping is not None:
            self.retriever_mapping = retriever_mapping
        self.optimizer = self._create_optimizer()
        if save_path and os.path.exists(save_path):
            self.optimizer.load(save_path)
        self.query_count = 0
        self.recent_queries = []
        self.max_recent_queries = 100

    def _create_optimizer(self) -> Union[AdaptiveLearner, ThompsonSamplingOptimizer, UCBOptimizer, EvolutionaryOptimizer]:
        """
        创建优化器
        
        Returns:
            优化器实例
        """
        if self.optimizer_type == "thompson":
            return ThompsonSamplingOptimizer(
                self.retriever_ids,
                min_weight=self.min_weight
            )
        elif self.optimizer_type == "ucb":
            return UCBOptimizer(
                self.retriever_ids,
                min_weight=self.min_weight
            )
        elif self.optimizer_type == "evolutionary":
            return EvolutionaryOptimizer(
                self.retriever_ids,
                min_weight=self.min_weight
            )
        else:
            return AdaptiveLearner(
                self.retriever_ids,
                min_weight=self.min_weight
            )

    def _compute_weights(self, complexity: str, confidence: float) -> (Dict[str, float], Dict[str, Any]):
        # 1. 取主辅路检索源
        main_sources = self.retriever_mapping.get(complexity, {}).get("main", [])
        aux_sources = self.retriever_mapping.get(complexity, {}).get("aux", [])
        
        # 2. 取基础alpha
        alpha = self.base_alpha.get(complexity, 0.5)
        
        # 3. 计算gamma调整后的alpha
        adjusted_alpha = alpha * (confidence ** self.gamma)
        
        # 4. 分配权重
        weights = {}
        
        # 主路径权重分配
        if main_sources:
            main_weight_per_source = adjusted_alpha / len(main_sources)
            for source in main_sources:
                if source in self.retriever_ids:
                    weights[source] = main_weight_per_source
        
        # 辅路径权重分配
        if aux_sources:
            aux_weight_per_source = (1 - adjusted_alpha) / len(aux_sources)
            for source in aux_sources:
                if source in self.retriever_ids:
                    weights[source] = aux_weight_per_source
        
        # 5. 确保所有检索器都有权重
        remaining_weight = 1.0 - sum(weights.values())
        remaining_sources = [r for r in self.retriever_ids if r not in weights]
        if remaining_sources and remaining_weight > 0:
            weight_per_remaining = remaining_weight / len(remaining_sources)
            for source in remaining_sources:
                weights[source] = weight_per_remaining
        
        # 6. 归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # 7. 确保最小权重
        for retriever_id in self.retriever_ids:
            if retriever_id not in weights or weights[retriever_id] < self.min_weight:
                weights[retriever_id] = self.min_weight
        
        # 重新归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        metadata = {
            "complexity": complexity,
            "confidence": confidence,
            "alpha": alpha,
            "adjusted_alpha": adjusted_alpha,
            "gamma": self.gamma,
            "main_sources": main_sources,
            "aux_sources": aux_sources
        }
        
        return weights, metadata

    async def calculate_weights(
        self,
        query: str,
        complexity: Optional[str] = None,
        confidence: Optional[float] = None,
        features: Optional[Dict[str, Any]] = None,
        feedback: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        # 复杂度和置信度推断
        if complexity is None and self.router:
            try:
                complexity_result = await self.router.predict_complexity_detailed(query)
                complexity = complexity_result["complexity"]
                if confidence is None:
                    confidence = complexity_result["confidence"]
            except Exception as e:
                logger.warning(f"复杂度预测失败: {e}")
                complexity = "one_hop"
                confidence = 0.5
        
        if complexity is None:
            complexity = "one_hop"
        if confidence is None:
            confidence = 0.5
        
        # 更新查询统计
        self.query_count += 1
        self.recent_queries.append({
            "query": query[:100],  # 截断查询
            "complexity": complexity,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        
        # 保持最近查询数量
        if len(self.recent_queries) > self.max_recent_queries:
            self.recent_queries = self.recent_queries[-self.max_recent_queries:]
        
        # 计算权重
        weights, metadata = self._compute_weights(complexity, confidence)
        
        # 应用优化器调整
        if self.optimizer:
            try:
                optimized_weights = self.optimizer.optimize(weights, query, features)
                weights.update(optimized_weights)
            except Exception as e:
                logger.warning(f"优化器调整失败: {e}")
        
        # 处理反馈
        if feedback and self.feedback_processor:
            try:
                self.feedback_processor.process_feedback(query, weights, feedback)
            except Exception as e:
                logger.warning(f"反馈处理失败: {e}")
        
        metadata.update({
            "query_count": self.query_count,
            "optimizer_type": self.optimizer_type,
            "weights": weights
        })
        
        return weights, metadata

    def update_from_feedback(self, query: str, feedback_data: Dict[str, Any]) -> None:
        """从反馈更新权重"""
        if not self.optimizer:
            return
        
        try:
            # 提取反馈信息
            performance_scores = feedback_data.get("performance_scores", {})
            user_ratings = feedback_data.get("user_ratings", {})
            
            # 更新优化器
            self.optimizer.update(query, performance_scores, user_ratings)
            
            # 保存优化器状态
            if self.save_path:
                self.optimizer.save(self.save_path)
                
        except Exception as e:
            logger.error(f"反馈更新失败: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = {
            "query_count": self.query_count,
            "recent_queries_count": len(self.recent_queries),
            "optimizer_type": self.optimizer_type,
            "gamma": self.gamma,
            "base_alpha": self.base_alpha,
            "retriever_mapping": self.retriever_mapping
        }
        
        # 复杂度分布
        complexity_dist = self._get_complexity_distribution()
        stats["complexity_distribution"] = complexity_dist
        
        # 优化器统计
        if self.optimizer:
            try:
                optimizer_stats = self.optimizer.get_stats()
                stats["optimizer_stats"] = optimizer_stats
            except Exception as e:
                logger.warning(f"获取优化器统计失败: {e}")
        
        return stats

    def _get_complexity_distribution(self) -> Dict[str, float]:
        """获取复杂度分布"""
        if not self.recent_queries:
            return {}
        
        complexity_counts = {}
        total_queries = len(self.recent_queries)
        
        for query_info in self.recent_queries:
            complexity = query_info.get("complexity", "unknown")
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        return {k: v / total_queries for k, v in complexity_counts.items()}

    def _get_top_feature_weights(self, top_n: int = 5) -> Dict[str, Dict[str, float]]:
        """获取最重要的特征权重"""
        if not self.optimizer:
            return {}
        
        try:
            feature_weights = self.optimizer.get_feature_weights()
            
            # 按权重排序
            sorted_features = sorted(
                feature_weights.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_n]
            
            return dict(sorted_features)
        except Exception as e:
            logger.warning(f"获取特征权重失败: {e}")
            return {}

    def get_weight_curve(self, complexity: str = 'one_hop', num_points: int = 100):
        """获取权重曲线数据"""
        confidences = np.linspace(0, 1, num_points)
        weight_curves = {retriever_id: [] for retriever_id in self.retriever_ids}
        
        for confidence in confidences:
            weights, _ = self._compute_weights(complexity, confidence)
            for retriever_id in self.retriever_ids:
                weight_curves[retriever_id].append(weights.get(retriever_id, 0))
        
        return {
            "confidences": confidences.tolist(),
            "weight_curves": weight_curves,
            "complexity": complexity
        }

    def plot_weight_curve(self, complexity: str = 'one_hop', save_path: str = None):
        """绘制权重曲线"""
        curve_data = self.get_weight_curve(complexity)
        
        plt.figure(figsize=(10, 6))
        for retriever_id, weights in curve_data["weight_curves"].items():
            plt.plot(curve_data["confidences"], weights, label=retriever_id, linewidth=2)
        
        plt.xlabel('Confidence')
        plt.ylabel('Weight')
        plt.title(f'Weight Curves for {complexity} Complexity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class ConfidenceAwareRouter:
    """置信度感知路由器"""
    
    def __init__(
        self,
        complexity_router: Optional[ComplexityAwareRouter] = None,
        weight_manager: Optional[AdaptiveWeightManager] = None,
        confidence_threshold: float = 0.6,
        enable_fallback: bool = True
    ):
        """
        初始化置信度感知路由器
        
        Args:
            complexity_router: 复杂度路由器
            weight_manager: 权重管理器
            confidence_threshold: 置信度阈值
            enable_fallback: 是否启用回退
        """
        self.complexity_router = complexity_router or ComplexityAwareRouter()
        self.weight_manager = weight_manager
        self.confidence_threshold = confidence_threshold
        self.enable_fallback = enable_fallback
        
        # 路由统计
        self.routing_stats = {
            "total_routes": 0,
            "high_confidence_routes": 0,
            "low_confidence_routes": 0,
            "fallback_routes": 0
        }
    
    async def route_with_weights(
        self,
        query: str,
        available_modes: List[str] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
        """
        路由查询并返回权重
        
        Args:
            query: 查询字符串
            available_modes: 可用模式列表
            **kwargs: 其他参数
            
        Returns:
            (模式, 权重字典, 元数据)
        """
        self.routing_stats["total_routes"] += 1
        
        # 获取复杂度预测
        try:
            complexity_result = await self.complexity_router.predict_complexity_detailed(query)
            complexity = complexity_result["complexity"]
            confidence = complexity_result["confidence"]
            method = complexity_result["method"]
        except Exception as e:
            logger.warning(f"复杂度预测失败: {e}")
            complexity = "one_hop"
            confidence = 0.5
            method = "fallback"
        
        # 更新统计
        if confidence >= self.confidence_threshold:
            self.routing_stats["high_confidence_routes"] += 1
        else:
            self.routing_stats["low_confidence_routes"] += 1
        
        # 计算权重
        weights = {}
        if self.weight_manager:
            try:
                weights, weight_metadata = await self.weight_manager.calculate_weights(
                    query, complexity, confidence
                )
            except Exception as e:
                logger.warning(f"权重计算失败: {e}")
                weights = {}
                weight_metadata = {}
        else:
            weight_metadata = {}
        
        # 选择模式
        if confidence >= self.confidence_threshold or not self.enable_fallback:
            # 高置信度或禁用回退：使用复杂度预测的模式
            selected_mode = await self.complexity_router.route(query, available_modes)
        else:
            # 低置信度且启用回退：使用默认模式
            self.routing_stats["fallback_routes"] += 1
            if available_modes and "naive" in available_modes:
                selected_mode = "naive"
            elif available_modes:
                selected_mode = available_modes[0]
            else:
                selected_mode = "naive"
        
        # 构建元数据
        metadata = {
            "complexity": complexity,
            "confidence": confidence,
            "method": method,
            "selected_mode": selected_mode,
            "routing_stats": self.routing_stats.copy()
        }
        metadata.update(weight_metadata)
        
        return selected_mode, weights, metadata
    
    async def route(self, query: str, available_modes: List[str] = None, **kwargs) -> str:
        """路由查询（简化版本）"""
        mode, _, _ = await self.route_with_weights(query, available_modes, **kwargs)
        return mode
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        stats = self.routing_stats.copy()
        if stats["total_routes"] > 0:
            stats["high_confidence_ratio"] = (
                stats["high_confidence_routes"] / stats["total_routes"]
            )
            stats["low_confidence_ratio"] = (
                stats["low_confidence_routes"] / stats["total_routes"]
            )
            stats["fallback_ratio"] = (
                stats["fallback_routes"] / stats["total_routes"]
            )
        return stats


class FeedbackDrivenFusion:
    """反馈驱动的融合器"""
    
    def __init__(
        self,
        retriever_ids: List[str],
        adaptive_weight_manager: Optional[AdaptiveWeightManager] = None,
        feedback_processor: Optional[FeedbackProcessor] = None,
        evaluation_metrics: Optional[RetrievalMetrics] = None,
        default_weights: Optional[Dict[str, float]] = None
    ):
        """
        初始化反馈驱动的融合器
        
        Args:
            retriever_ids: 检索器ID列表
            adaptive_weight_manager: 自适应权重管理器
            feedback_processor: 反馈处理器
            evaluation_metrics: 评估指标
            default_weights: 默认权重
        """
        self.retriever_ids = retriever_ids
        self.adaptive_weight_manager = adaptive_weight_manager
        self.feedback_processor = feedback_processor
        self.evaluation_metrics = evaluation_metrics
        self.default_weights = default_weights or {r_id: 1.0 / len(retriever_ids) for r_id in retriever_ids}
        
        # 会话统计
        self.session_stats = {
            "total_queries": 0,
            "total_feedback": 0,
            "average_rating": 0.0,
            "weight_updates": 0
        }
        
        # 上下文特征
        self.context_features = {
            "recent_queries": [],
            "user_preferences": {},
            "performance_history": {}
        }
    
    async def fuse(
        self,
        query: str,
        retrieval_results: Dict[str, List[Dict[str, Any]]],
        complexity: Optional[str] = None,
        confidence: Optional[float] = None,
        features: Optional[Dict[str, Any]] = None,
        feedback: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        融合检索结果
        
        Args:
            query: 查询字符串
            retrieval_results: 检索结果字典
            complexity: 查询复杂度
            confidence: 置信度
            features: 特征字典
            feedback: 反馈数据
            
        Returns:
            (融合结果, 元数据)
        """
        self.session_stats["total_queries"] += 1
        
        # 提取特征
        extracted_features = self._extract_features(query, complexity, confidence, features)
        
        # 计算权重
        weights = {}
        if self.adaptive_weight_manager:
            try:
                weights, weight_metadata = await self.adaptive_weight_manager.calculate_weights(
                    query, complexity, confidence, extracted_features, feedback
                )
            except Exception as e:
                logger.warning(f"自适应权重计算失败: {e}")
                weights = self.default_weights.copy()
                weight_metadata = {}
        else:
            weights = self.default_weights.copy()
            weight_metadata = {}
        
        # 融合结果
        fused_results = self._combine_results(retrieval_results, weights)
        
        # 更新上下文特征
        self._update_context_features(extracted_features)
        
        # 处理反馈
        if feedback:
            self.collect_feedback(query, fused_results, feedback)
        
        # 构建元数据
        metadata = {
            "weights": weights,
            "features": extracted_features,
            "session_stats": self.session_stats.copy(),
            "context_features": self.context_features.copy()
        }
        metadata.update(weight_metadata)
        
        return fused_results, metadata
    
    def _extract_features(
        self,
        query: str,
        complexity: Optional[str],
        confidence: Optional[float],
        features: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """提取特征"""
        extracted_features = {
            "query_length": len(query.split()),
            "query_complexity": complexity or "unknown",
            "confidence": confidence or 0.5,
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加用户提供的特征
        if features:
            extracted_features.update(features)
        
        # 添加上下文特征
        extracted_features.update({
            "recent_query_count": len(self.context_features["recent_queries"]),
            "user_preferences": self.context_features["user_preferences"].copy(),
            "performance_history": self.context_features["performance_history"].copy()
        })
        
        return extracted_features
    
    def _update_context_features(self, features: Dict[str, Any]) -> None:
        """更新上下文特征"""
        # 更新最近查询
        self.context_features["recent_queries"].append({
            "query": features.get("query", "")[:50],  # 截断
            "complexity": features.get("query_complexity", "unknown"),
            "confidence": features.get("confidence", 0.5),
            "timestamp": features.get("timestamp", "")
        })
        
        # 保持最近查询数量
        if len(self.context_features["recent_queries"]) > 20:
            self.context_features["recent_queries"] = self.context_features["recent_queries"][-20:]
    
    def _combine_results(
        self,
        retrieval_results: Dict[str, List[Dict[str, Any]]],
        weights: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """合并检索结果"""
        # 收集所有文档
        all_documents = {}
        
        for retriever_id, results in retrieval_results.items():
            weight = weights.get(retriever_id, 0.0)
            
            for i, result in enumerate(results):
                doc_id = result.get('id', f"{retriever_id}_{i}")
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
                    'source': retriever_id,
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
        
        return sorted_documents
    
    def collect_feedback(
        self,
        query: str,
        fused_results: List[Dict[str, Any]],
        user_feedback: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """收集反馈"""
        self.session_stats["total_feedback"] += 1
        
        # 更新平均评分
        rating = user_feedback.get("rating", 0.0)
        if rating > 0:
            current_avg = self.session_stats["average_rating"]
            total_feedback = self.session_stats["total_feedback"]
            self.session_stats["average_rating"] = (
                (current_avg * (total_feedback - 1) + rating) / total_feedback
            )
        
        # 更新用户偏好
        if "preferences" in user_feedback:
            self.context_features["user_preferences"].update(user_feedback["preferences"])
        
        # 更新性能历史
        performance = user_feedback.get("performance", {})
        if performance:
            self.context_features["performance_history"][query] = {
                "rating": rating,
                "performance": performance,
                "timestamp": datetime.now().isoformat()
            }
        
        # 更新权重管理器
        if self.adaptive_weight_manager:
            try:
                self.adaptive_weight_manager.update_from_feedback(query, user_feedback)
                self.session_stats["weight_updates"] += 1
            except Exception as e:
                logger.warning(f"权重更新失败: {e}")
        
        # 处理反馈
        if self.feedback_processor:
            try:
                self.feedback_processor.process_feedback(query, fused_results, user_feedback)
            except Exception as e:
                logger.warning(f"反馈处理失败: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """获取会话统计"""
        stats = self.session_stats.copy()
        stats["context_features"] = {
            "recent_queries_count": len(self.context_features["recent_queries"]),
            "user_preferences_count": len(self.context_features["user_preferences"]),
            "performance_history_count": len(self.context_features["performance_history"])
        }
        return stats 