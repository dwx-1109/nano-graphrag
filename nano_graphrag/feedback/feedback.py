"""
反馈处理与收集模块

本模块提供完整的用户反馈处理和分析功能，包括：
- 反馈收集（显式反馈、隐式反馈）
- 反馈处理和分析
- 特征提取和相似度计算
- 权重预测和反馈分析
"""

import os
import json
import uuid
import time
import logging
import random
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from datetime import datetime

import numpy as np

from .storage import FeedbackStorage


class FeedbackCollector:
    """反馈收集器"""
    
    def __init__(
        self, 
        storage: Optional[FeedbackStorage] = None,
        feature_extractors: Optional[Dict[str, Callable]] = None
    ):
        """
        初始化反馈收集器
        
        Args:
            storage: 反馈存储器，如果为None则创建默认存储器
            feature_extractors: 特征提取函数字典
        """
        self.storage = storage or FeedbackStorage()
        
        # 特征提取函数
        self.feature_extractors = feature_extractors or {}
        
        # 设置默认特征提取器
        self._set_default_extractors()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 当前会话
        self.session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now()
    
    def _set_default_extractors(self):
        """设置默认特征提取器"""
        default_extractors = {
            'query_length': self._extract_query_length,
            'query_type': self._extract_query_type,
            'query_complexity': self._extract_query_complexity,
            'timestamp': lambda data: datetime.now().isoformat()
        }
        
        # 添加默认特征提取器（如果未提供）
        for name, extractor in default_extractors.items():
            if name not in self.feature_extractors:
                self.feature_extractors[name] = extractor
    
    def _extract_query_length(self, data: Dict[str, Any]) -> int:
        """提取查询长度特征"""
        query = data.get('query', '')
        return len(query.split())
    
    def _extract_query_type(self, data: Dict[str, Any]) -> str:
        """提取查询类型特征"""
        query = data.get('query', '').lower()
        
        if any(w in query for w in ['what', 'who', 'where', 'which']):
            return 'factoid'
        elif any(w in query for w in ['how', 'why']):
            return 'explanation'
        elif any(w in query for w in ['list', 'enumerate', 'examples']):
            return 'listing'
        else:
            return 'unknown'
    
    def _extract_query_complexity(self, data: Dict[str, Any]) -> str:
        """提取查询复杂度特征"""
        # 如果已提供复杂度信息，直接使用
        if 'complexity' in data:
            return data['complexity']
            
        # 尝试从复杂度分类器获取
        try:
            complexity = data.get('complexity_class', None)
            if complexity:
                return complexity
                
            # 尝试使用置信度路由器预测复杂度
            router = data.get('router', None)
            if router:
                if hasattr(router, 'predict_complexity'):
                    import asyncio
                    loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(loop)
                        complexity = loop.run_until_complete(router.predict_complexity(data.get('query', '')))
                        return complexity
                    finally:
                        loop.close()
        except Exception as e:
            self.logger.warning(f"无法提取查询复杂度: {e}")
        
        # 默认基于查询长度判断
        query_length = len(data.get('query', '').split())
        if query_length < 5:
            return 'zero_hop'
        elif query_length < 10:
            return 'one_hop'
        else:
            return 'multi_hop'
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取特征
        
        Args:
            data: 输入数据
            
        Returns:
            特征字典
        """
        features = {}
        
        # 使用所有特征提取器
        for name, extractor in self.feature_extractors.items():
            try:
                features[name] = extractor(data)
            except Exception as e:
                self.logger.warning(f"特征提取器 {name} 失败: {e}")
                features[name] = None
        
        return features
    
    def collect_feedback(
        self,
        query_id: str,
        query: str,
        user_rating: float,
        retrieval_results: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        fusion_result: Optional[List[Dict[str, Any]]] = None,
        fusion_id: str = "default",
        weights: Optional[Dict[str, float]] = None,
        execution_time: Optional[Dict[str, float]] = None,
        user_comments: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        收集显式反馈
        
        Args:
            query_id: 查询ID
            query: 查询文本
            user_rating: 用户评分
            retrieval_results: 检索结果
            fusion_result: 融合结果
            fusion_id: 融合方法ID
            weights: 权重信息
            execution_time: 执行时间
            user_comments: 用户评论
            metadata: 元数据
            
        Returns:
            反馈数据字典
        """
        # 构建基础数据
        data = {
            'query': query,
            'user_rating': user_rating,
            'retrieval_results': retrieval_results or {},
            'fusion_result': fusion_result or [],
            'fusion_id': fusion_id,
            'weights': weights or {},
            'execution_time': execution_time or {},
            'user_comments': user_comments,
            'metadata': metadata or {}
        }
        
        # 提取特征
        features = self.extract_features(data)
        
        # 构建反馈数据
        feedback_data = {
            'query_id': query_id,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'feedback_type': 'explicit',
            'query': query,
            'user_rating': user_rating,
            'features': features,
            'retrieval_results': retrieval_results,
            'fusion_result': fusion_result,
            'fusion_id': fusion_id,
            'weights': weights,
            'execution_time': execution_time,
            'user_comments': user_comments,
            'metadata': metadata
        }
        
        # 存储反馈
        success = self.storage.store(query_id, feedback_data)
        
        if success:
            self.logger.info(f"成功收集反馈: {query_id}")
        else:
            self.logger.error(f"收集反馈失败: {query_id}")
        
        return feedback_data
    
    def collect_implicit_feedback(
        self,
        query_id: str,
        query: str,
        clicks: List[str],
        dwell_times: Dict[str, float],
        retrieval_results: Dict[str, List[Dict[str, Any]]],
        fusion_result: List[Dict[str, Any]],
        fusion_id: str = "default",
        weights: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        收集隐式反馈
        
        Args:
            query_id: 查询ID
            query: 查询文本
            clicks: 点击的文档ID列表
            dwell_times: 停留时间字典
            retrieval_results: 检索结果
            fusion_result: 融合结果
            fusion_id: 融合方法ID
            weights: 权重信息
            metadata: 元数据
            
        Returns:
            反馈数据字典
        """
        # 构建基础数据
        data = {
            'query': query,
            'clicks': clicks,
            'dwell_times': dwell_times,
            'retrieval_results': retrieval_results,
            'fusion_result': fusion_result,
            'fusion_id': fusion_id,
            'weights': weights or {},
            'metadata': metadata or {}
        }
        
        # 提取特征
        features = self.extract_features(data)
        
        # 计算隐式评分
        implicit_rating = self._calculate_implicit_rating(clicks, dwell_times, fusion_result)
        
        # 构建反馈数据
        feedback_data = {
            'query_id': query_id,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'feedback_type': 'implicit',
            'query': query,
            'implicit_rating': implicit_rating,
            'clicks': clicks,
            'dwell_times': dwell_times,
            'features': features,
            'retrieval_results': retrieval_results,
            'fusion_result': fusion_result,
            'fusion_id': fusion_id,
            'weights': weights,
            'metadata': metadata
        }
        
        # 存储反馈
        success = self.storage.store(query_id, feedback_data)
        
        if success:
            self.logger.info(f"成功收集隐式反馈: {query_id}")
        else:
            self.logger.error(f"收集隐式反馈失败: {query_id}")
        
        return feedback_data
    
    def _calculate_implicit_rating(self, clicks: List[str], dwell_times: Dict[str, float], 
                                 fusion_result: List[Dict[str, Any]]) -> float:
        """
        计算隐式评分
        
        Args:
            clicks: 点击的文档ID列表
            dwell_times: 停留时间字典
            fusion_result: 融合结果
            
        Returns:
            隐式评分
        """
        if not fusion_result:
            return 0.0
        
        # 基于点击和停留时间计算评分
        total_score = 0.0
        max_possible_score = len(fusion_result)
        
        for i, doc in enumerate(fusion_result):
            doc_id = doc.get('id', str(i))
            
            # 点击分数
            click_score = 1.0 if doc_id in clicks else 0.0
            
            # 停留时间分数
            dwell_time = dwell_times.get(doc_id, 0.0)
            dwell_score = min(dwell_time / 30.0, 1.0)  # 30秒为满分
            
            # 位置分数（越靠前分数越高）
            position_score = 1.0 - (i / len(fusion_result))
            
            # 综合分数
            doc_score = (click_score * 0.5 + dwell_score * 0.3 + position_score * 0.2)
            total_score += doc_score
        
        return total_score / max_possible_score if max_possible_score > 0 else 0.0
    
    def start_new_session(self):
        """开始新会话"""
        self.session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now()
        self.logger.info(f"开始新会话: {self.session_id}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息"""
        return {
            'session_id': self.session_id,
            'session_start_time': self.session_start_time.isoformat(),
            'duration': (datetime.now() - self.session_start_time).total_seconds()
        }
    
    def save_feedback(self, feedback_data: dict) -> bool:
        """
        保存反馈数据
        
        Args:
            feedback_data: 反馈数据
            
        Returns:
            是否成功保存
        """
        try:
            # 添加会话信息
            feedback_data['session_info'] = self.get_session_info()
            
            # 提取特征
            features = self.extract_features(feedback_data)
            feedback_data['features'] = features
            
            # 保存到存储
            if self.storage:
                self.storage.save_feedback(feedback_data)
            
            self.logger.info(f"已保存反馈数据: {feedback_data.get('query_id', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存反馈数据失败: {e}")
            return False


class FeedbackProcessor:
    """反馈处理器"""
    
    def __init__(
        self,
        feedback_file: str = None,
        storage: Optional[FeedbackStorage] = None,
        similarity_threshold: float = 0.7
    ):
        """
        初始化反馈处理器
        
        Args:
            feedback_file: 反馈数据文件路径
            storage: 反馈存储器实例
            similarity_threshold: 相似度阈值
        """
        self.feedback_data = {}
        self.feedback_file = feedback_file
        self.storage = storage or FeedbackStorage()
        self.similarity_threshold = similarity_threshold
        
        # 缓存
        self.weights_cache = {}
        self.similarity_cache = {}
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 加载反馈数据
        if feedback_file and os.path.exists(feedback_file):
            self.load_feedback(feedback_file)
    
    def extract_features(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从查询数据中提取特征
        
        Args:
            query_data: 查询数据
            
        Returns:
            特征字典
        """
        features = {}
        
        # 提取查询文本
        query = query_data.get('query', '')
        
        # 特征1: 查询长度
        features['query_length'] = len(query.split())
        
        # 特征2: 查询复杂度
        features['complexity'] = query_data.get('complexity', self._extract_complexity(query))
        
        # 特征3: 置信度
        features['confidence'] = query_data.get('confidence')
        
        # 特征4: 查询类型
        features['query_type'] = self._extract_query_type(query)
        
        return features
    
    def _extract_complexity(self, query: str) -> str:
        """
        提取查询复杂度
        
        Args:
            query: 查询文本
            
        Returns:
            复杂度类别
        """
        query_length = len(query.split())
        
        if query_length < 5:
            return 'zero_hop'
        elif query_length < 10:
            return 'one_hop'
        else:
            return 'multi_hop'
    
    def _extract_query_type(self, query: str) -> str:
        """
        提取查询类型
        
        Args:
            query: 查询文本
            
        Returns:
            查询类型
        """
        query_lower = query.lower()
        
        if any(w in query_lower for w in ['what', 'who', 'where', 'which']):
            return 'factoid'
        elif any(w in query_lower for w in ['how', 'why']):
            return 'explanation'
        elif any(w in query_lower for w in ['list', 'enumerate', 'examples']):
            return 'listing'
        else:
            return 'unknown'
    
    def add_feedback(self, query_id: str, feedback_data: Dict[str, Any]) -> None:
        """
        添加反馈数据
        
        Args:
            query_id: 查询ID
            feedback_data: 反馈数据
        """
        if query_id not in self.feedback_data:
            self.feedback_data[query_id] = []
        
        self.feedback_data[query_id].append(feedback_data)
        
        # 清除相关缓存
        self._clear_cache_for_query(query_id)
        
        self.logger.info(f"添加反馈数据: {query_id}")
    
    def get_feedback(self, query_id: str) -> List[Dict[str, Any]]:
        """
        获取指定查询的反馈
        
        Args:
            query_id: 查询ID
            
        Returns:
            反馈数据列表
        """
        return self.feedback_data.get(query_id, [])
    
    def get_all_feedback(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取所有反馈数据
        
        Returns:
            所有反馈数据
        """
        return self.feedback_data.copy()
    
    def save_feedback(self, file_path: str) -> None:
        """
        保存反馈数据到文件
        
        Args:
            file_path: 文件路径
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"反馈数据已保存到: {file_path}")
        except Exception as e:
            self.logger.error(f"保存反馈数据失败: {e}")
    
    def load_feedback(self, file_path: str) -> None:
        """
        从文件加载反馈数据
        
        Args:
            file_path: 文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.feedback_data = json.load(f)
            self.logger.info(f"反馈数据已从文件加载: {file_path}")
        except Exception as e:
            self.logger.error(f"加载反馈数据失败: {e}")
    
    def find_similar_feedback(self, query_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        查找相似反馈
        
        Args:
            query_data: 查询数据
            
        Returns:
            相似反馈列表
        """
        query_features = self.extract_features(query_data)
        similar_feedback = []
        
        for query_id, feedback_list in self.feedback_data.items():
            for feedback in feedback_list:
                feedback_features = feedback.get('features', {})
                similarity = self._compute_similarity(query_features, feedback_features)
                
                if similarity >= self.similarity_threshold:
                    similar_feedback.append({
                        'query_id': query_id,
                        'feedback': feedback,
                        'similarity': similarity
                    })
        
        # 按相似度排序
        similar_feedback.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_feedback
    
    def _compute_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        计算特征相似度
        
        Args:
            features1: 特征1
            features2: 特征2
            
        Returns:
            相似度分数
        """
        if not features1 or not features2:
            return 0.0
        
        # 计算数值特征的相似度
        numerical_features = ['query_length', 'confidence']
        numerical_similarity = 0.0
        numerical_count = 0
        
        for feature in numerical_features:
            if feature in features1 and feature in features2:
                val1 = features1[feature]
                val2 = features2[feature]
                
                if val1 is not None and val2 is not None:
                    # 归一化数值差异
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        diff = abs(val1 - val2) / max_val
                        numerical_similarity += 1.0 - diff
                        numerical_count += 1
        
        numerical_similarity = numerical_similarity / numerical_count if numerical_count > 0 else 0.0
        
        # 计算分类特征的相似度
        categorical_features = ['complexity', 'query_type']
        categorical_similarity = 0.0
        categorical_count = 0
        
        for feature in categorical_features:
            if feature in features1 and feature in features2:
                val1 = features1[feature]
                val2 = features2[feature]
                
                if val1 is not None and val2 is not None:
                    categorical_similarity += 1.0 if val1 == val2 else 0.0
                    categorical_count += 1
        
        categorical_similarity = categorical_similarity / categorical_count if categorical_count > 0 else 0.0
        
        # 综合相似度
        total_similarity = (numerical_similarity * 0.4 + categorical_similarity * 0.6)
        
        return total_similarity
    
    def predict_weights(self, query_data: Dict[str, Any], retriever_ids: List[str]) -> Dict[str, float]:
        """
        预测权重
        
        Args:
            query_data: 查询数据
            retriever_ids: 检索器ID列表
            
        Returns:
            权重字典
        """
        # 检查缓存
        cache_key = f"{query_data.get('query', '')}_{','.join(retriever_ids)}"
        if cache_key in self.weights_cache:
            return self.weights_cache[cache_key]
        
        # 查找相似反馈
        similar_feedback = self.find_similar_feedback(query_data)
        
        if not similar_feedback:
            # 如果没有相似反馈，返回均匀权重
            weights = {retriever_id: 1.0 / len(retriever_ids) for retriever_id in retriever_ids}
            self.weights_cache[cache_key] = weights
            return weights
        
        # 基于相似反馈计算权重
        total_weight = 0.0
        retriever_scores = {retriever_id: 0.0 for retriever_id in retriever_ids}
        
        for item in similar_feedback:
            feedback = item['feedback']
            similarity = item['similarity']
            
            # 获取反馈中的权重信息
            feedback_weights = feedback.get('weights', {})
            user_rating = feedback.get('user_rating', 0.5)
            
            # 计算加权分数
            weight_score = similarity * user_rating
            
            for retriever_id in retriever_ids:
                if retriever_id in feedback_weights:
                    retriever_scores[retriever_id] += feedback_weights[retriever_id] * weight_score
            
            total_weight += weight_score
        
        # 归一化权重
        if total_weight > 0:
            weights = {
                retriever_id: score / total_weight 
                for retriever_id, score in retriever_scores.items()
            }
        else:
            weights = {retriever_id: 1.0 / len(retriever_ids) for retriever_id in retriever_ids}
        
        # 缓存结果
        self.weights_cache[cache_key] = weights
        
        return weights
    
    def analyze_feedback(self) -> Dict[str, Any]:
        """
        分析反馈数据
        
        Returns:
            分析结果
        """
        if not self.feedback_data:
            return {"message": "没有反馈数据"}
        
        analysis = {
            "total_queries": len(self.feedback_data),
            "total_feedback": sum(len(feedback_list) for feedback_list in self.feedback_data.values()),
            "avg_rating": 0.0,
            "rating_distribution": {},
            "complexity_distribution": {},
            "query_type_distribution": {},
            "top_rated_queries": [],
            "low_rated_queries": []
        }
        
        all_ratings = []
        complexity_counts = {}
        query_type_counts = {}
        
        for query_id, feedback_list in self.feedback_data.items():
            query_ratings = []
            
            for feedback in feedback_list:
                # 收集评分
                rating = feedback.get('user_rating', 0.0)
                if rating > 0:
                    query_ratings.append(rating)
                    all_ratings.append(rating)
                
                # 统计复杂度分布
                complexity = feedback.get('features', {}).get('complexity', 'unknown')
                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
                
                # 统计查询类型分布
                query_type = feedback.get('features', {}).get('query_type', 'unknown')
                query_type_counts[query_type] = query_type_counts.get(query_type, 0) + 1
            
            # 计算查询平均评分
            if query_ratings:
                avg_query_rating = np.mean(query_ratings)
                analysis["top_rated_queries"].append({
                    "query_id": query_id,
                    "avg_rating": avg_query_rating,
                    "feedback_count": len(query_ratings)
                })
        
        # 计算总体平均评分
        if all_ratings:
            analysis["avg_rating"] = np.mean(all_ratings)
        
        # 评分分布
        if all_ratings:
            for rating in all_ratings:
                rating_bucket = f"{int(rating * 2) / 2:.1f}-{int(rating * 2 + 1) / 2:.1f}"
                analysis["rating_distribution"][rating_bucket] = analysis["rating_distribution"].get(rating_bucket, 0) + 1
        
        # 复杂度分布
        analysis["complexity_distribution"] = complexity_counts
        
        # 查询类型分布
        analysis["query_type_distribution"] = query_type_counts
        
        # 排序最高评分查询
        analysis["top_rated_queries"].sort(key=lambda x: x["avg_rating"], reverse=True)
        analysis["top_rated_queries"] = analysis["top_rated_queries"][:10]
        
        return analysis
    
    def _clear_cache_for_query(self, query_id: str) -> None:
        """
        清除指定查询的缓存
        
        Args:
            query_id: 查询ID
        """
        # 清除权重缓存中相关的条目
        keys_to_remove = []
        for cache_key in self.weights_cache.keys():
            if query_id in cache_key:
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            del self.weights_cache[key]
    
    def clear_cache(self) -> None:
        """清除所有缓存"""
        self.weights_cache.clear()
        self.similarity_cache.clear()
        self.logger.info("已清除所有缓存")
    
    def process_feedback(self, feedback_data: dict) -> dict:
        """
        处理反馈数据
        
        Args:
            feedback_data: 反馈数据
            
        Returns:
            处理后的反馈数据
        """
        processed_feedback = {
            'timestamp': datetime.now().isoformat(),
            'original_data': feedback_data,
            'processed_features': self.extract_features(feedback_data),
            'similar_feedback': self.find_similar_feedback(feedback_data)
        }
        
        # 添加到反馈数据中
        query_id = feedback_data.get('query_id', str(uuid.uuid4()))
        if query_id not in self.feedback_data:
            self.feedback_data[query_id] = []
        
        self.feedback_data[query_id].append(processed_feedback)
        
        # 清除相关缓存
        self._clear_cache_for_query(query_id)
        
        self.logger.info(f"已处理反馈数据: {query_id}")
        return processed_feedback
    
    def get_feedback_stats(self) -> dict:
        """
        获取反馈统计信息
        
        Returns:
            反馈统计字典
        """
        total_feedback = sum(len(feedbacks) for feedbacks in self.feedback_data.values())
        unique_queries = len(self.feedback_data)
        
        # 计算平均评分
        total_rating = 0.0
        rating_count = 0
        
        for query_feedbacks in self.feedback_data.values():
            for feedback in query_feedbacks:
                original_data = feedback.get('original_data', {})
                if 'user_rating' in original_data:
                    total_rating += original_data['user_rating']
                    rating_count += 1
        
        avg_rating = total_rating / rating_count if rating_count > 0 else 0.0
        
        # 计算反馈类型分布
        feedback_types = {}
        for query_feedbacks in self.feedback_data.values():
            for feedback in query_feedbacks:
                original_data = feedback.get('original_data', {})
                feedback_type = original_data.get('feedback_type', 'unknown')
                feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1
        
        return {
            'total_feedback_count': total_feedback,
            'unique_queries': unique_queries,
            'average_rating': avg_rating,
            'feedback_type_distribution': feedback_types,
            'cache_size': len(self.weights_cache) + len(self.similarity_cache),
            'last_updated': datetime.now().isoformat()
        }


# 便捷函数
def create_feedback_collector(storage: Optional[FeedbackStorage] = None) -> FeedbackCollector:
    """
    创建反馈收集器
    
    Args:
        storage: 反馈存储器
        
    Returns:
        FeedbackCollector: 反馈收集器实例
    """
    return FeedbackCollector(storage=storage)


def create_feedback_processor(storage: Optional[FeedbackStorage] = None, 
                            similarity_threshold: float = 0.7) -> FeedbackProcessor:
    """
    创建反馈处理器
    
    Args:
        storage: 反馈存储器
        similarity_threshold: 相似度阈值
        
    Returns:
        FeedbackProcessor: 反馈处理器实例
    """
    return FeedbackProcessor(storage=storage, similarity_threshold=similarity_threshold) 