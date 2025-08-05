"""
文档去重与质量过滤模块

本模块提供完整的文档去重和质量过滤功能，包括：
- 文档去重算法（贪心、聚类）
- 相似度计算（Jaccard、余弦、编辑距离）
- 质量过滤（长度、分数、来源可靠性）
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """去重结果数据类"""
    deduplicated_documents: List[Dict[str, Any]]
    removed_documents: List[Dict[str, Any]]
    similarity_matrix: Optional[np.ndarray]
    method: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class QualityFilterResult:
    """质量过滤结果数据类"""
    filtered_documents: List[Dict[str, Any]]
    removed_documents: List[Dict[str, Any]]
    filter_criteria: Dict[str, Any]
    method: str
    metadata: Dict[str, Any]


class SimilarityCalculator:
    """相似度计算器"""
    
    def __init__(self, method: str = "jaccard"):
        """
        初始化相似度计算器
        
        Args:
            method: 相似度计算方法，支持 "jaccard", "cosine", "levenshtein"
        """
        self.method = method
        
    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        计算Jaccard相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: Jaccard相似度分数
            
        Example:
            >>> calc = SimilarityCalculator()
            >>> calc.calculate_jaccard_similarity("hello world", "hello python")
            0.25
        """
        if not text1 or not text2:
            return 0.0
        
        # 分词
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # 计算交集和并集
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        计算余弦相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 余弦相似度分数
            
        Example:
            >>> calc = SimilarityCalculator()
            >>> calc.calculate_cosine_similarity("hello world", "hello python")
            0.5
        """
        if not text1 or not text2:
            return 0.0
        
        # 分词
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        if not words1 or not words2:
            return 0.0
        
        # 创建词汇表
        vocab = set(words1 + words2)
        
        # 计算词频向量
        vec1 = [words1.count(word) for word in vocab]
        vec2 = [words2.count(word) for word in vocab]
        
        # 计算余弦相似度
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def calculate_levenshtein_similarity(self, text1: str, text2: str) -> float:
        """
        计算编辑距离相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 编辑距离相似度分数
            
        Example:
            >>> calc = SimilarityCalculator()
            >>> calc.calculate_levenshtein_similarity("hello", "helo")
            0.8
        """
        if not text1 or not text2:
            return 0.0
        
        def levenshtein_distance(s1, s2):
            """计算编辑距离"""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        根据指定方法计算相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 相似度分数
        """
        if self.method == "jaccard":
            return self.calculate_jaccard_similarity(text1, text2)
        elif self.method == "cosine":
            return self.calculate_cosine_similarity(text1, text2)
        elif self.method == "levenshtein":
            return self.calculate_levenshtein_similarity(text1, text2)
        else:
            logger.warning(f"Unknown similarity method: {self.method}, using jaccard")
            return self.calculate_jaccard_similarity(text1, text2)
    
    def calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        计算文本列表的相似度矩阵
        
        Args:
            texts: 文本列表
            
        Returns:
            np.ndarray: 相似度矩阵
        """
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.calculate_similarity(texts[i], texts[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        return similarity_matrix


class DocumentDeduplicator:
    """文档去重器"""
    
    def __init__(self, similarity_threshold: float = 0.85, method: str = "greedy"):
        """
        初始化去重器
        
        Args:
            similarity_threshold: 相似度阈值，超过此值认为重复
            method: 去重方法，支持 "greedy", "clustering"
        """
        self.similarity_threshold = similarity_threshold
        self.method = method
        self.similarity_calculator = SimilarityCalculator()
        
    def deduplicate_greedy(self, documents: List[Dict[str, Any]], 
                          similarity_matrix: Optional[np.ndarray] = None) -> DeduplicationResult:
        """
        贪心去重算法
        
        Args:
            documents: 文档列表
            similarity_matrix: 预计算的相似度矩阵
            
        Returns:
            DeduplicationResult: 去重结果
        """
        if not documents:
            return DeduplicationResult(
                deduplicated_documents=[],
                removed_documents=[],
                similarity_matrix=None,
                method="greedy",
                parameters={"similarity_threshold": self.similarity_threshold},
                metadata={"num_original": 0, "num_kept": 0, "num_removed": 0}
            )
        
        # 计算相似度矩阵
        if similarity_matrix is None:
            texts = [doc.get('content', '') for doc in documents]
            similarity_matrix = self.similarity_calculator.calculate_similarity_matrix(texts)
        
        # 贪心去重
        kept_indices = []
        removed_indices = []
        
        for i in range(len(documents)):
            is_duplicate = False
            
            # 检查是否与已保留的文档重复
            for j in kept_indices:
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                removed_indices.append(i)
            else:
                kept_indices.append(i)
        
        # 构建结果
        deduplicated_documents = [documents[i] for i in kept_indices]
        removed_documents = [documents[i] for i in removed_indices]
        
        return DeduplicationResult(
            deduplicated_documents=deduplicated_documents,
            removed_documents=removed_documents,
            similarity_matrix=similarity_matrix,
            method="greedy",
            parameters={"similarity_threshold": self.similarity_threshold},
            metadata={
                "num_original": len(documents),
                "num_kept": len(deduplicated_documents),
                "num_removed": len(removed_documents),
                "removal_ratio": len(removed_documents) / len(documents) if documents else 0.0
            }
        )
    
    def deduplicate_clustering(self, documents: List[Dict[str, Any]], 
                              similarity_matrix: Optional[np.ndarray] = None) -> DeduplicationResult:
        """
        聚类去重算法
        
        Args:
            documents: 文档列表
            similarity_matrix: 预计算的相似度矩阵
            
        Returns:
            DeduplicationResult: 去重结果
        """
        if not documents:
            return DeduplicationResult(
                deduplicated_documents=[],
                removed_documents=[],
                similarity_matrix=None,
                method="clustering",
                parameters={"similarity_threshold": self.similarity_threshold},
                metadata={"num_original": 0, "num_kept": 0, "num_removed": 0}
            )
        
        # 计算相似度矩阵
        if similarity_matrix is None:
            texts = [doc.get('content', '') for doc in documents]
            similarity_matrix = self.similarity_calculator.calculate_similarity_matrix(texts)
        
        # 找到连通分量
        components = self._find_connected_components(similarity_matrix)
        
        # 从每个组件中选择一个代表文档
        kept_indices = []
        removed_indices = []
        
        for component in components:
            # 选择组件中第一个文档作为代表
            kept_indices.append(component[0])
            removed_indices.extend(component[1:])
        
        # 构建结果
        deduplicated_documents = [documents[i] for i in kept_indices]
        removed_documents = [documents[i] for i in removed_indices]
        
        return DeduplicationResult(
            deduplicated_documents=deduplicated_documents,
            removed_documents=removed_documents,
            similarity_matrix=similarity_matrix,
            method="clustering",
            parameters={"similarity_threshold": self.similarity_threshold},
            metadata={
                "num_original": len(documents),
                "num_kept": len(deduplicated_documents),
                "num_removed": len(removed_documents),
                "removal_ratio": len(removed_documents) / len(documents) if documents else 0.0,
                "num_clusters": len(components)
            }
        )
    
    def _find_connected_components(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """
        找到相似度矩阵中的连通分量
        
        Args:
            similarity_matrix: 相似度矩阵
            
        Returns:
            List[List[int]]: 连通分量列表
        """
        n = len(similarity_matrix)
        visited = [False] * n
        components = []
        
        def dfs(node: int, component: List[int]):
            """深度优先搜索"""
            visited[node] = True
            component.append(node)
            
            for neighbor in range(n):
                if (not visited[neighbor] and 
                    similarity_matrix[node][neighbor] >= self.similarity_threshold):
                    dfs(neighbor, component)
        
        # 找到所有连通分量
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, component)
                components.append(component)
        
        return components
    
    def deduplicate(self, documents: List[Dict[str, Any]], 
                   similarity_matrix: Optional[np.ndarray] = None) -> DeduplicationResult:
        """
        执行文档去重
        
        Args:
            documents: 文档列表
            similarity_matrix: 预计算的相似度矩阵
            
        Returns:
            DeduplicationResult: 去重结果
        """
        if self.method == "greedy":
            return self.deduplicate_greedy(documents, similarity_matrix)
        elif self.method == "clustering":
            return self.deduplicate_clustering(documents, similarity_matrix)
        else:
            raise ValueError(f"不支持的去重方法: {self.method}")
    
    def calculate_similarity(self, doc1: dict, doc2: dict) -> float:
        """
        计算两个文档的相似度
        
        Args:
            doc1: 第一个文档
            doc2: 第二个文档
            
        Returns:
            相似度分数 (0-1)
        """
        content1 = doc1.get('content', '')
        content2 = doc2.get('content', '')
        return self.similarity_calculator.calculate_similarity(content1, content2)


class QualityFilter:
    """质量过滤器"""
    
    def __init__(self, method: str = "comprehensive"):
        """
        初始化质量过滤器
        
        Args:
            method: 质量过滤方法，支持 "comprehensive", "score"
        """
        self.method = method
        
    def filter_comprehensive(self, documents: List[Dict[str, Any]], 
                           min_length: int = 10,
                           min_score: float = 0.3,
                           min_source_reliability: float = 0.5) -> QualityFilterResult:
        """
        综合质量过滤
        
        Args:
            documents: 文档列表
            min_length: 最小内容长度
            min_score: 最小融合分数
            min_source_reliability: 最小来源可靠性
            
        Returns:
            QualityFilterResult: 质量过滤结果
        """
        if not documents:
            return QualityFilterResult(
                filtered_documents=[],
                removed_documents=[],
                filter_criteria={
                    'min_length': min_length,
                    'min_score': min_score,
                    'min_source_reliability': min_source_reliability
                },
                method="comprehensive",
                metadata={"num_original": 0, "num_filtered": 0, "num_removed": 0}
            )
        
        filtered_documents = []
        removed_documents = []
        
        for doc in documents:
            # 检查内容长度
            content = doc.get('content', '')
            if len(content) < min_length:
                removed_documents.append({**doc, 'filter_reason': 'content_too_short'})
                continue
            
            # 检查融合分数
            fusion_score = doc.get('fusion_score', 0.0)
            if fusion_score < min_score:
                removed_documents.append({**doc, 'filter_reason': 'score_too_low'})
                continue
            
            # 检查来源可靠性
            source = doc.get('metadata', {}).get('source', 'unknown')
            reliability_score = self._get_source_reliability(source)
            if reliability_score < min_source_reliability:
                removed_documents.append({**doc, 'filter_reason': 'source_unreliable'})
                continue
            
            # 通过所有过滤条件
            filtered_documents.append(doc)
        
        return QualityFilterResult(
            filtered_documents=filtered_documents,
            removed_documents=removed_documents,
            filter_criteria={
                'min_length': min_length,
                'min_score': min_score,
                'min_source_reliability': min_source_reliability
            },
            method="comprehensive",
            metadata={
                "num_original": len(documents),
                "num_filtered": len(filtered_documents),
                "num_removed": len(removed_documents),
                "filter_ratio": len(removed_documents) / len(documents) if documents else 0.0
            }
        )
    
    def filter_by_score(self, documents: List[Dict[str, Any]], 
                       min_score: float = 0.5) -> QualityFilterResult:
        """
        基于分数过滤
        
        Args:
            documents: 文档列表
            min_score: 最小分数阈值
            
        Returns:
            QualityFilterResult: 质量过滤结果
        """
        if not documents:
            return QualityFilterResult(
                filtered_documents=[],
                removed_documents=[],
                filter_criteria={'min_score': min_score},
                method="score",
                metadata={"num_original": 0, "num_filtered": 0, "num_removed": 0}
            )
        
        filtered_documents = []
        removed_documents = []
        
        for doc in documents:
            score = doc.get('fusion_score', 0.0)
            if score >= min_score:
                filtered_documents.append(doc)
            else:
                removed_documents.append({**doc, 'filter_reason': 'score_too_low'})
        
        return QualityFilterResult(
            filtered_documents=filtered_documents,
            removed_documents=removed_documents,
            filter_criteria={'min_score': min_score},
            method="score",
            metadata={
                "num_original": len(documents),
                "num_filtered": len(filtered_documents),
                "num_removed": len(removed_documents),
                "filter_ratio": len(removed_documents) / len(documents) if documents else 0.0
            }
        )
    
    def _get_source_reliability(self, source: str) -> float:
        """
        获取来源可靠性分数
        
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
    
    def filter_documents(self, documents: List[Dict[str, Any]], **kwargs) -> QualityFilterResult:
        """
        执行质量过滤
        
        Args:
            documents: 文档列表
            **kwargs: 过滤参数
            
        Returns:
            QualityFilterResult: 质量过滤结果
        """
        if self.method == "comprehensive":
            return self.filter_comprehensive(documents, **kwargs)
        elif self.method == "score":
            return self.filter_by_score(documents, **kwargs)
        else:
            logger.warning(f"Unknown filter method: {self.method}, using comprehensive")
            return self.filter_comprehensive(documents, **kwargs)


# 便捷函数
def calculate_text_similarity(text1: str, text2: str, method: str = "jaccard") -> float:
    """
    计算两个文本的相似度
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
        method: 相似度计算方法
        
    Returns:
        float: 相似度分数
    """
    calculator = SimilarityCalculator(method)
    return calculator.calculate_similarity(text1, text2)


def deduplicate_documents(documents: List[Dict[str, Any]], 
                         similarity_threshold: float = 0.85,
                         method: str = "greedy") -> DeduplicationResult:
    """
    对文档进行去重
    
    Args:
        documents: 文档列表
        similarity_threshold: 相似度阈值
        method: 去重方法
        
    Returns:
        DeduplicationResult: 去重结果
    """
    deduplicator = DocumentDeduplicator(similarity_threshold, method)
    return deduplicator.deduplicate(documents)


def filter_documents_by_quality(documents: List[Dict[str, Any]], 
                               method: str = "comprehensive",
                               **kwargs) -> QualityFilterResult:
    """
    对文档进行质量过滤
    
    Args:
        documents: 文档列表
        method: 过滤方法
        **kwargs: 过滤参数
        
    Returns:
        QualityFilterResult: 质量过滤结果
    """
    filter_obj = QualityFilter(method)
    return filter_obj.filter_documents(documents, **kwargs) 