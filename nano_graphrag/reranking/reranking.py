"""
文档重排序与质量评估模块

本模块提供完整的文档重排序和质量评估功能，包括：
- 文档重排序算法（融合分数、加权、多样性感知）
- 质量评估（综合质量、来源可靠性、新鲜度）
- 多样性控制（最大边际相关性）
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)


@dataclass
class RerankingResult:
    """重排序结果数据类"""
    reranked_documents: List[Dict[str, Any]]
    reranking_scores: List[float]
    method: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class QualityAssessmentResult:
    """质量评估结果数据类"""
    quality_scores: List[float]
    quality_metrics: Dict[str, Any]
    method: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class DiversityResult:
    """多样性控制结果数据类"""
    diverse_documents: List[Dict[str, Any]]
    diversity_scores: List[float]
    method: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


class DocumentReranker:
    """文档重排序器"""
    
    def __init__(self, method: str = "fusion_score"):
        """
        初始化重排序器
        
        Args:
            method: 重排序方法，支持 "fusion_score", "weighted", "diversity_aware"
        """
        self.method = method
        self.parameters = {}
        
    def rerank_by_fusion_score(self, documents: List[Dict[str, Any]]) -> RerankingResult:
        """
        基于融合分数重排序
        
        Args:
            documents: 文档列表，每个文档包含 'fusion_score' 字段
            
        Returns:
            RerankingResult: 重排序结果
            
        Example:
            >>> reranker = DocumentReranker()
            >>> result = reranker.rerank_by_fusion_score(documents)
            >>> print(f"重排序了 {len(result.reranked_documents)} 个文档")
        """
        # 按融合分数排序
        sorted_documents = sorted(
            documents,
            key=lambda x: x.get('fusion_score', 0.0),
            reverse=True
        )
        
        reranking_scores = [doc.get('fusion_score', 0.0) for doc in sorted_documents]
        
        return RerankingResult(
            reranked_documents=sorted_documents,
            reranking_scores=reranking_scores,
            method="fusion_score",
            parameters={},
            metadata={
                "num_documents": len(sorted_documents),
                "score_range": (min(reranking_scores), max(reranking_scores)) if reranking_scores else (0, 0)
            }
        )
    
    def rerank_weighted(self, documents: List[Dict[str, Any]], 
                       weights: Dict[str, float]) -> RerankingResult:
        """
        基于加权分数重排序
        
        Args:
            documents: 文档列表
            weights: 权重字典，键为分数字段名，值为权重
            
        Returns:
            RerankingResult: 重排序结果
            
        Example:
            >>> weights = {'fusion_score': 0.7, 'quality_score': 0.3}
            >>> result = reranker.rerank_weighted(documents, weights)
        """
        # 计算加权分数
        for doc in documents:
            weighted_score = 0.0
            for field, weight in weights.items():
                score = doc.get(field, 0.0)
                weighted_score += weight * score
            doc['weighted_score'] = weighted_score
        
        # 按加权分数排序
        sorted_documents = sorted(
            documents,
            key=lambda x: x.get('weighted_score', 0.0),
            reverse=True
        )
        
        reranking_scores = [doc.get('weighted_score', 0.0) for doc in sorted_documents]
        
        return RerankingResult(
            reranked_documents=sorted_documents,
            reranking_scores=reranking_scores,
            method="weighted",
            parameters={"weights": weights},
            metadata={
                "num_documents": len(sorted_documents),
                "score_range": (min(reranking_scores), max(reranking_scores)) if reranking_scores else (0, 0)
            }
        )
    
    def rerank_diversity_aware(self, documents: List[Dict[str, Any]], 
                              diversity_weight: float = 0.3) -> RerankingResult:
        """
        多样性感知重排序
        
        Args:
            documents: 文档列表
            diversity_weight: 多样性权重
            
        Returns:
            RerankingResult: 重排序结果
        """
        if not documents:
            return RerankingResult(
                reranked_documents=[],
                reranking_scores=[],
                method="diversity_aware",
                parameters={"diversity_weight": diversity_weight},
                metadata={"num_documents": 0}
            )
        
        # 计算相似度矩阵
        similarity_matrix = self._calculate_similarity_matrix(documents)
        
        # 多样性感知排序
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        # 选择第一个文档（相关性最高的）
        first_idx = max(remaining_indices, key=lambda i: documents[i].get('fusion_score', 0.0))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # 贪心选择后续文档
        while remaining_indices:
            best_idx = None
            best_score = -float('inf')
            
            for idx in remaining_indices:
                # 计算相关性分数
                relevance_score = documents[idx].get('fusion_score', 0.0)
                
                # 计算多样性分数（与已选文档的最小相似度）
                if selected_indices:
                    similarities = [similarity_matrix[idx][i] for i in selected_indices]
                    diversity_score = 1.0 - max(similarities)
                else:
                    diversity_score = 1.0
                
                # 综合分数
                combined_score = (1 - diversity_weight) * relevance_score + diversity_weight * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        # 构建结果
        reranked_documents = [documents[i] for i in selected_indices]
        reranking_scores = [documents[i].get('fusion_score', 0.0) for i in selected_indices]
        
        return RerankingResult(
            reranked_documents=reranked_documents,
            reranking_scores=reranking_scores,
            method="diversity_aware",
            parameters={"diversity_weight": diversity_weight},
            metadata={
                "num_documents": len(reranked_documents),
                "score_range": (min(reranking_scores), max(reranking_scores)) if reranking_scores else (0, 0)
            }
        )
    
    def _calculate_similarity_matrix(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """
        计算文档相似度矩阵
        
        Args:
            documents: 文档列表
            
        Returns:
            np.ndarray: 相似度矩阵
        """
        n = len(documents)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._calculate_text_similarity(
                    documents[i].get('content', ''),
                    documents[j].get('content', '')
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        return similarity_matrix
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 相似度分数
        """
        if not text1 or not text2:
            return 0.0
        
        # 简单的Jaccard相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def rerank(self, documents: List[Dict[str, Any]], **kwargs) -> RerankingResult:
        """
        执行重排序
        
        Args:
            documents: 文档列表
            **kwargs: 重排序参数
            
        Returns:
            RerankingResult: 重排序结果
        """
        if self.method == "fusion_score":
            return self.rerank_by_fusion_score(documents)
        elif self.method == "weighted":
            weights = kwargs.get('weights', {'fusion_score': 1.0})
            return self.rerank_weighted(documents, weights)
        elif self.method == "diversity_aware":
            diversity_weight = kwargs.get('diversity_weight', 0.3)
            return self.rerank_diversity_aware(documents, diversity_weight)
        else:
            logger.warning(f"Unknown reranking method: {self.method}, using fusion_score")
            return self.rerank_by_fusion_score(documents)


class QualityAssessor:
    """质量评估器"""
    
    def __init__(self, method: str = "comprehensive"):
        """
        初始化质量评估器
        
        Args:
            method: 质量评估方法，支持 "comprehensive", "simple"
        """
        self.method = method
        self.parameters = {}
        
    def assess_comprehensive_quality(self, documents: List[Dict[str, Any]]) -> QualityAssessmentResult:
        """
        综合质量评估
        
        Args:
            documents: 文档列表
            
        Returns:
            QualityAssessmentResult: 质量评估结果
        """
        if not documents:
            return QualityAssessmentResult(
                quality_scores=[],
                quality_metrics={},
                method="comprehensive",
                parameters={},
                metadata={"num_documents": 0}
            )
        
        quality_scores = []
        quality_metrics = {
            'content_length': [],
            'source_reliability': [],
            'freshness': [],
            'relevance': []
        }
        
        for doc in documents:
            # 内容长度质量
            content = doc.get('content', '')
            length_score = min(len(content) / 100, 1.0)  # 标准化到0-1
            quality_metrics['content_length'].append(length_score)
            
            # 来源可靠性
            source = doc.get('metadata', {}).get('source', 'unknown')
            reliability_score = self._assess_source_reliability(source)
            quality_metrics['source_reliability'].append(reliability_score)
            
            # 新鲜度
            date_str = doc.get('metadata', {}).get('date', '2020-01-01')
            freshness_score = self._assess_freshness(date_str)
            quality_metrics['freshness'].append(freshness_score)
            
            # 相关性（使用融合分数作为代理）
            relevance_score = doc.get('fusion_score', 0.0)
            quality_metrics['relevance'].append(relevance_score)
            
            # 综合质量分数
            weights = {
                'content_length': 0.2,
                'source_reliability': 0.3,
                'freshness': 0.2,
                'relevance': 0.3
            }
            
            overall_score = (
                weights['content_length'] * length_score +
                weights['source_reliability'] * reliability_score +
                weights['freshness'] * freshness_score +
                weights['relevance'] * relevance_score
            )
            
            quality_scores.append(overall_score)
        
        return QualityAssessmentResult(
            quality_scores=quality_scores,
            quality_metrics=quality_metrics,
            method="comprehensive",
            parameters={"weights": weights},
            metadata={
                "num_documents": len(documents),
                "score_range": (min(quality_scores), max(quality_scores)) if quality_scores else (0, 0),
                "avg_score": np.mean(quality_scores) if quality_scores else 0.0
            }
        )
    
    def assess_simple_quality(self, documents: List[Dict[str, Any]]) -> QualityAssessmentResult:
        """
        简单质量评估
        
        Args:
            documents: 文档列表
            
        Returns:
            QualityAssessmentResult: 质量评估结果
        """
        if not documents:
            return QualityAssessmentResult(
                quality_scores=[],
                quality_metrics={},
                method="simple",
                parameters={},
                metadata={"num_documents": 0}
            )
        
        quality_scores = []
        
        for doc in documents:
            # 基于融合分数的简单质量评估
            fusion_score = doc.get('fusion_score', 0.0)
            content_length = len(doc.get('content', ''))
            
            # 简单的质量分数
            quality_score = fusion_score * 0.8 + min(content_length / 1000, 1.0) * 0.2
            quality_scores.append(quality_score)
        
        return QualityAssessmentResult(
            quality_scores=quality_scores,
            quality_metrics={'fusion_score': [doc.get('fusion_score', 0.0) for doc in documents]},
            method="simple",
            parameters={},
            metadata={
                "num_documents": len(documents),
                "score_range": (min(quality_scores), max(quality_scores)) if quality_scores else (0, 0),
                "avg_score": np.mean(quality_scores) if quality_scores else 0.0
            }
        )
    
    def _assess_source_reliability(self, source: str) -> float:
        """
        评估来源可靠性
        
        Args:
            source: 来源标识
            
        Returns:
            float: 可靠性分数
        """
        # 简单的来源可靠性映射
        reliability_map = {
            'wikipedia': 0.9,
            'arxiv': 0.8,
            'pubmed': 0.8,
            'stackoverflow': 0.7,
            'github': 0.6,
            'blog': 0.5,
            'forum': 0.4,
            'unknown': 0.3
        }
        
        # 检查是否包含已知来源关键词
        source_lower = source.lower()
        for key, score in reliability_map.items():
            if key in source_lower:
                return score
        
        return reliability_map.get('unknown', 0.3)
    
    def _assess_freshness(self, date_str: str) -> float:
        """
        评估文档新鲜度
        
        Args:
            date_str: 日期字符串
            
        Returns:
            float: 新鲜度分数
        """
        try:
            # 解析日期
            doc_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            current_date = date.today()
            
            # 计算天数差
            days_diff = (current_date - doc_date).days
            
            # 新鲜度分数（越新分数越高）
            if days_diff <= 30:
                return 1.0
            elif days_diff <= 365:
                return 0.8
            elif days_diff <= 1095:  # 3年
                return 0.6
            else:
                return 0.4
        except:
            # 如果日期解析失败，返回默认分数
            return 0.5
    
    def assess_quality(self, documents: List[Dict[str, Any]], **kwargs) -> QualityAssessmentResult:
        """
        执行质量评估
        
        Args:
            documents: 文档列表
            **kwargs: 评估参数
            
        Returns:
            QualityAssessmentResult: 质量评估结果
        """
        if self.method == "comprehensive":
            return self.assess_comprehensive_quality(documents)
        elif self.method == "simple":
            return self.assess_simple_quality(documents)
        else:
            logger.warning(f"Unknown quality assessment method: {self.method}, using comprehensive")
            return self.assess_comprehensive_quality(documents)


class DiversityController:
    """多样性控制器"""
    
    def __init__(self, method: str = "maximal_marginal_relevance"):
        """
        初始化多样性控制器
        
        Args:
            method: 多样性控制方法，支持 "maximal_marginal_relevance"
        """
        self.method = method
        self.parameters = {}
        
    def maximal_marginal_relevance(self, documents: List[Dict[str, Any]], 
                                  lambda_param: float = 0.5,
                                  top_k: Optional[int] = None) -> DiversityResult:
        """
        最大边际相关性算法
        
        Args:
            documents: 文档列表
            lambda_param: 相关性权重参数
            top_k: 返回的文档数量
            
        Returns:
            DiversityResult: 多样性控制结果
        """
        if not documents:
            return DiversityResult(
                diverse_documents=[],
                diversity_scores=[],
                method="maximal_marginal_relevance",
                parameters={"lambda": lambda_param},
                metadata={"num_documents": 0}
            )
        
        # 计算相似度矩阵
        similarity_matrix = self._calculate_similarity_matrix(documents)
        
        # MMR算法
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        # 选择第一个文档（相关性最高的）
        first_idx = max(remaining_indices, key=lambda i: documents[i].get('fusion_score', 0.0))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # 贪心选择后续文档
        while remaining_indices and (top_k is None or len(selected_indices) < top_k):
            best_idx = None
            best_score = -float('inf')
            
            for idx in remaining_indices:
                # 计算相关性分数
                relevance_score = documents[idx].get('fusion_score', 0.0)
                
                # 计算多样性分数（与已选文档的最小相似度）
                if selected_indices:
                    similarities = [similarity_matrix[idx][i] for i in selected_indices]
                    diversity_score = 1.0 - max(similarities)
                else:
                    diversity_score = 1.0
                
                # MMR分数
                mmr_score = lambda_param * relevance_score + (1 - lambda_param) * diversity_score
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        # 构建结果
        diverse_documents = [documents[i] for i in selected_indices]
        diversity_scores = [documents[i].get('fusion_score', 0.0) for i in selected_indices]
        
        return DiversityResult(
            diverse_documents=diverse_documents,
            diversity_scores=diversity_scores,
            method="maximal_marginal_relevance",
            parameters={"lambda": lambda_param, "top_k": top_k},
            metadata={
                "num_documents": len(diverse_documents),
                "score_range": (min(diversity_scores), max(diversity_scores)) if diversity_scores else (0, 0)
            }
        )
    
    def _calculate_similarity_matrix(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """
        计算文档相似度矩阵
        
        Args:
            documents: 文档列表
            
        Returns:
            np.ndarray: 相似度矩阵
        """
        n = len(documents)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._calculate_text_similarity(
                    documents[i].get('content', ''),
                    documents[j].get('content', '')
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        return similarity_matrix
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 相似度分数
        """
        if not text1 or not text2:
            return 0.0
        
        # 简单的Jaccard相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def control_diversity(self, documents: List[Dict[str, Any]], **kwargs) -> DiversityResult:
        """
        执行多样性控制
        
        Args:
            documents: 文档列表
            **kwargs: 控制参数
            
        Returns:
            DiversityResult: 多样性控制结果
        """
        if self.method == "maximal_marginal_relevance":
            lambda_param = kwargs.get('lambda_param', 0.5)
            top_k = kwargs.get('top_k', None)
            return self.maximal_marginal_relevance(documents, lambda_param, top_k)
        else:
            logger.warning(f"Unknown diversity control method: {self.method}, using maximal_marginal_relevance")
            return self.maximal_marginal_relevance(documents)


# 便捷函数
def rerank_documents(documents: List[Dict[str, Any]], 
                    method: str = "fusion_score",
                    **kwargs) -> RerankingResult:
    """
    对文档进行重排序
    
    Args:
        documents: 文档列表
        method: 重排序方法
        **kwargs: 重排序参数
        
    Returns:
        RerankingResult: 重排序结果
    """
    reranker = DocumentReranker(method)
    return reranker.rerank(documents, **kwargs)


def assess_document_quality(documents: List[Dict[str, Any]], 
                           method: str = "comprehensive",
                           **kwargs) -> QualityAssessmentResult:
    """
    对文档进行质量评估
    
    Args:
        documents: 文档列表
        method: 评估方法
        **kwargs: 评估参数
        
    Returns:
        QualityAssessmentResult: 质量评估结果
    """
    assessor = QualityAssessor(method)
    return assessor.assess_quality(documents, **kwargs)


def control_document_diversity(documents: List[Dict[str, Any]], 
                              method: str = "maximal_marginal_relevance",
                              **kwargs) -> DiversityResult:
    """
    对文档进行多样性控制
    
    Args:
        documents: 文档列表
        method: 控制方法
        **kwargs: 控制参数
        
    Returns:
        DiversityResult: 多样性控制结果
    """
    controller = DiversityController(method)
    return controller.control_diversity(documents, **kwargs) 