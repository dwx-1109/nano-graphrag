"""
结果标准化模块

实现不同检索源分数的归一化方法，确保不同来源的分数可以公平比较和融合。
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """归一化结果数据类"""
    original_scores: List[float]
    normalized_scores: List[float]
    method: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]


class ScoreNormalizer:
    """分数标准化器"""
    
    def __init__(self, method: str = "minmax"):
        """
        初始化标准化器
        
        Args:
            method: 归一化方法，支持 "minmax", "zscore", "robust"
        """
        self.method = method
        self.parameters = {}
        
    def minmax_normalize(self, scores: List[float], 
                        min_val: Optional[float] = None, 
                        max_val: Optional[float] = None) -> NormalizationResult:
        """
        Min-Max归一化：s_norm = (s - s_min) / (s_max - s_min)
        
        Args:
            scores: 原始分数列表
            min_val: 最小值，如果为None则自动计算
            max_val: 最大值，如果为None则自动计算
            
        Returns:
            NormalizationResult: 归一化结果
        """
        scores_array = np.array(scores)
        
        if min_val is None:
            min_val = np.min(scores_array)
        if max_val is None:
            max_val = np.max(scores_array)
            
        # 避免除零错误
        denominator = max_val - min_val
        if denominator == 0:
            normalized_scores = np.ones_like(scores_array)
        else:
            normalized_scores = (scores_array - min_val) / denominator
            
        self.parameters = {
            "min_val": min_val,
            "max_val": max_val,
            "denominator": denominator
        }
        
        return NormalizationResult(
            original_scores=scores,
            normalized_scores=normalized_scores.tolist(),
            method="minmax",
            parameters=self.parameters,
            metadata={
                "original_range": (min_val, max_val),
                "normalized_range": (0, 1)
            }
        )
    
    def zscore_normalize(self, scores: List[float], 
                        mean_val: Optional[float] = None, 
                        std_val: Optional[float] = None) -> NormalizationResult:
        """
        Z-score归一化：s_norm = (s - μ) / σ
        
        Args:
            scores: 原始分数列表
            mean_val: 均值，如果为None则自动计算
            std_val: 标准差，如果为None则自动计算
            
        Returns:
            NormalizationResult: 归一化结果
        """
        scores_array = np.array(scores)
        
        if mean_val is None:
            mean_val = np.mean(scores_array)
        if std_val is None:
            std_val = np.std(scores_array)
            
        # 避免除零错误
        if std_val == 0:
            normalized_scores = np.zeros_like(scores_array)
        else:
            normalized_scores = (scores_array - mean_val) / std_val
            
        self.parameters = {
            "mean_val": mean_val,
            "std_val": std_val
        }
        
        return NormalizationResult(
            original_scores=scores,
            normalized_scores=normalized_scores.tolist(),
            method="zscore",
            parameters=self.parameters,
            metadata={
                "original_stats": (mean_val, std_val),
                "normalized_stats": (0, 1)
            }
        )
    
    def robust_normalize(self, scores: List[float]) -> NormalizationResult:
        """
        鲁棒归一化：使用中位数和四分位距
        
        Args:
            scores: 原始分数列表
            
        Returns:
            NormalizationResult: 归一化结果
        """
        scores_array = np.array(scores)
        
        median_val = np.median(scores_array)
        q75, q25 = np.percentile(scores_array, [75, 25])
        iqr = q75 - q25
        
        # 避免除零错误
        if iqr == 0:
            normalized_scores = np.zeros_like(scores_array)
        else:
            normalized_scores = (scores_array - median_val) / iqr
            
        self.parameters = {
            "median_val": median_val,
            "iqr": iqr,
            "q25": q25,
            "q75": q75
        }
        
        return NormalizationResult(
            original_scores=scores,
            normalized_scores=normalized_scores.tolist(),
            method="robust",
            parameters=self.parameters,
            metadata={
                "original_stats": (median_val, iqr),
                "normalized_stats": (0, 1)
            }
        )
    
    def normalize(self, scores: List[float], **kwargs) -> NormalizationResult:
        """
        执行归一化
        
        Args:
            scores: 原始分数列表
            **kwargs: 其他参数
            
        Returns:
            NormalizationResult: 归一化结果
        """
        if self.method == "minmax":
            return self.minmax_normalize(scores, **kwargs)
        elif self.method == "zscore":
            return self.zscore_normalize(scores, **kwargs)
        elif self.method == "robust":
            return self.robust_normalize(scores, **kwargs)
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")


class NormalizationAnalyzer:
    """归一化效果分析器"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_distribution(self, original_scores: List[float], 
                           normalized_scores: List[float]) -> Dict[str, Any]:
        """
        分析归一化前后的分布变化
        
        Args:
            original_scores: 原始分数
            normalized_scores: 归一化后分数
            
        Returns:
            Dict: 分析结果
        """
        original_array = np.array(original_scores)
        normalized_array = np.array(normalized_scores)
        
        analysis = {
            "original": {
                "mean": float(np.mean(original_array)),
                "std": float(np.std(original_array)),
                "min": float(np.min(original_array)),
                "max": float(np.max(original_array)),
                "median": float(np.median(original_array)),
                "skewness": float(self._calculate_skewness(original_array)),
                "kurtosis": float(self._calculate_kurtosis(original_array))
            },
            "normalized": {
                "mean": float(np.mean(normalized_array)),
                "std": float(np.std(normalized_array)),
                "min": float(np.min(normalized_array)),
                "max": float(np.max(normalized_array)),
                "median": float(np.median(normalized_array)),
                "skewness": float(self._calculate_skewness(normalized_array)),
                "kurtosis": float(self._calculate_kurtosis(normalized_array))
            }
        }
        
        return analysis
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def compare_methods(self, scores: List[float]) -> Dict[str, Any]:
        """
        比较不同归一化方法的效果
        
        Args:
            scores: 原始分数列表
            
        Returns:
            Dict: 比较结果
        """
        methods = ["minmax", "zscore", "robust"]
        results = {}
        
        for method in methods:
            normalizer = ScoreNormalizer(method=method)
            result = normalizer.normalize(scores)
            analysis = self.analyze_distribution(scores, result.normalized_scores)
            
            results[method] = {
                "result": result,
                "analysis": analysis
            }
            
        return results
    
    def recommend_method(self, scores: List[float]) -> str:
        """
        根据数据特征推荐归一化方法
        
        Args:
            scores: 原始分数列表
            
        Returns:
            str: 推荐的方法
        """
        scores_array = np.array(scores)
        
        # 检查是否有异常值
        q75, q25 = np.percentile(scores_array, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = np.sum((scores_array < lower_bound) | (scores_array > upper_bound))
        outlier_ratio = outliers / len(scores_array)
        
        # 检查分布是否接近正态
        skewness = self._calculate_skewness(scores_array)
        kurtosis = self._calculate_kurtosis(scores_array)
        
        # 推荐逻辑
        if outlier_ratio > 0.1:  # 异常值较多
            return "robust"
        elif abs(skewness) < 0.5 and abs(kurtosis) < 1.0:  # 接近正态分布
            return "zscore"
        else:  # 其他情况
            return "minmax"


def normalize_retrieval_scores(retrieval_results: List[Dict[str, Any]], 
                              method: str = "minmax") -> List[Dict[str, Any]]:
    """
    对检索结果进行分数归一化
    
    Args:
        retrieval_results: 检索结果列表，每个元素包含 'score' 字段
        method: 归一化方法
        
    Returns:
        List[Dict]: 归一化后的检索结果
    """
    if not retrieval_results:
        return retrieval_results
    
    # 提取分数
    scores = [result.get('score', 0.0) for result in retrieval_results]
    
    # 执行归一化
    normalizer = ScoreNormalizer(method=method)
    normalization_result = normalizer.normalize(scores)
    
    # 更新结果
    normalized_results = []
    for i, result in enumerate(retrieval_results):
        normalized_result = result.copy()
        normalized_result['score'] = normalization_result.normalized_scores[i]
        normalized_result['original_score'] = result.get('score', 0.0)
        normalized_results.append(normalized_result)
    
    return normalized_results 