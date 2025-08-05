"""
多臂赌博机优化模块

提供基于多臂赌博机的权重优化功能
"""

import os
import json
import math
import time
import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from abc import abstractmethod

from .learning import WeightOptimizer


class BanditOptimizer(WeightOptimizer):
    """多臂赌博机优化器基类"""
    
    def __init__(
        self,
        retriever_ids: List[str],
        exploration_rate: float = 0.1,
        min_weight: float = 0.01,
        window_size: int = 100,
        decay_factor: float = 0.95
    ):
        """
        初始化多臂赌博机优化器
        
        Args:
            retriever_ids: 检索器ID列表
            exploration_rate: 探索率
            min_weight: 最小权重
            window_size: 滑动窗口大小
            decay_factor: 衰减因子
        """
        super().__init__(retriever_ids)
        
        self.exploration_rate = exploration_rate
        self.min_weight = min_weight
        self.window_size = window_size
        self.decay_factor = decay_factor
        
        # 初始化计数器和奖励
        self.counts = {r_id: 0 for r_id in retriever_ids}
        self.values = {r_id: 0.0 for r_id in retriever_ids}
        
        # 滑动窗口
        self.rewards_window = {r_id: [] for r_id in retriever_ids}
        
        # 特征相关奖励
        self.feature_rewards = {}
        
        # 当前选择的检索器
        self.current_choice = None
        
        # 累计奖励
        self.total_reward = 0.0
        self.num_trials = 0
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取优化器状态
        
        Returns:
            状态字典
        """
        return {
            'type': self.__class__.__name__,
            'retriever_ids': self.retriever_ids,
            'exploration_rate': self.exploration_rate,
            'min_weight': self.min_weight,
            'window_size': self.window_size,
            'decay_factor': self.decay_factor,
            'counts': self.counts,
            'values': self.values,
            'rewards_window': self.rewards_window,
            'feature_rewards': self.feature_rewards,
            'total_reward': self.total_reward,
            'num_trials': self.num_trials
        }
    
    def load(self, file_path: str) -> bool:
        """
        加载优化器状态
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('type') != self.__class__.__name__:
                self.logger.error("文件类型不匹配")
                return False
            
            self.counts = data.get('counts', {})
            self.values = data.get('values', {})
            self.rewards_window = data.get('rewards_window', {})
            self.feature_rewards = data.get('feature_rewards', {})
            self.total_reward = data.get('total_reward', 0.0)
            self.num_trials = data.get('num_trials', 0)
            
            self.logger.info(f"优化器状态已从 {file_path} 加载")
            return True
        except Exception as e:
            self.logger.error(f"加载优化器状态失败: {e}")
            return False
    
    def optimize(self, query_data: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        优化权重
        
        Args:
            query_data: 查询数据
            feedback: 反馈数据
            
        Returns:
            优化后的权重
        """
        # 提取特征
        features = self._extract_features(query_data)
        
        # 选择权重
        weights = self._select_weights(features)
        
        # 如果有反馈，更新模型
        if feedback:
            self._update_model(query_data, feedback)
        
        return self.constrain_weights(weights, self.min_weight)
    
    def _update_model(self, query_data: Dict[str, Any], feedback: Dict[str, Any]) -> None:
        """
        更新模型
        
        Args:
            query_data: 查询数据
            feedback: 反馈数据
        """
        # 提取特征
        features = self._extract_features(query_data)
        
        # 提取奖励
        reward = self._extract_reward(feedback)
        
        # 更新特征奖励
        self._update_feature_rewards(features, reward)
        
        # 更新统计
        self.total_reward += reward
        self.num_trials += 1
    
    def _extract_features(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取特征
        
        Args:
            query_data: 查询数据
            
        Returns:
            特征字典
        """
        features = {}
        
        # 查询长度
        query = query_data.get('query', '')
        features['query_length'] = len(query.split())
        
        # 查询类型
        features['query_type'] = self._extract_query_type(query)
        
        # 复杂度
        features['complexity'] = query_data.get('complexity', 'unknown')
        
        # 置信度
        features['confidence'] = query_data.get('confidence', 0.5)
        
        return features
    
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
    
    def _extract_reward(self, feedback: Dict[str, Any]) -> float:
        """
        提取奖励
        
        Args:
            feedback: 反馈数据
            
        Returns:
            奖励值
        """
        # 基于用户评分
        user_rating = feedback.get('user_rating', 0.5)
        
        # 基于检索质量
        retrieval_quality = feedback.get('retrieval_quality', 0.5)
        
        # 综合奖励
        reward = 0.7 * user_rating + 0.3 * retrieval_quality
        
        return max(0.0, min(1.0, reward))
    
    def _update_feature_rewards(self, features: Dict[str, Any], reward: float) -> None:
        """
        更新特征奖励
        
        Args:
            features: 特征字典
            reward: 奖励值
        """
        for feature_name, feature_value in features.items():
            feature_key = f"{feature_name}_{feature_value}"
            
            if feature_key not in self.feature_rewards:
                self.feature_rewards[feature_key] = {
                    'total_reward': 0.0,
                    'count': 0,
                    'avg_reward': 0.0
                }
            
            # 更新统计
            stats = self.feature_rewards[feature_key]
            stats['total_reward'] += reward
            stats['count'] += 1
            stats['avg_reward'] = stats['total_reward'] / stats['count']
    
    @abstractmethod
    def _select_weights(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        选择权重（抽象方法）
        
        Args:
            features: 特征字典
            
        Returns:
            权重字典
        """
        pass


class ThompsonSamplingOptimizer(BanditOptimizer):
    """Thompson采样优化器"""
    
    def __init__(
        self,
        retriever_ids: List[str],
        min_weight: float = 0.01,
        window_size: int = 100,
        decay_factor: float = 0.95,
        alpha: float = 1.0,
        beta: float = 1.0
    ):
        """
        初始化Thompson采样优化器
        
        Args:
            retriever_ids: 检索器ID列表
            min_weight: 最小权重
            window_size: 滑动窗口大小
            decay_factor: 衰减因子
            alpha: Beta分布参数alpha
            beta: Beta分布参数beta
        """
        super().__init__(retriever_ids, min_weight=min_weight, window_size=window_size, decay_factor=decay_factor)
        
        self.alpha = alpha
        self.beta = beta
        
        # 初始化Beta分布参数
        self.alpha_params = {r_id: alpha for r_id in retriever_ids}
        self.beta_params = {r_id: beta for r_id in retriever_ids}
    
    def get_state(self) -> Dict[str, Any]:
        """获取优化器状态"""
        state = super().get_state()
        state.update({
            'alpha': self.alpha,
            'beta': self.beta,
            'alpha_params': self.alpha_params,
            'beta_params': self.beta_params
        })
        return state
    
    def load(self, file_path: str) -> bool:
        """加载优化器状态"""
        if not super().load(file_path):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.alpha_params = data.get('alpha_params', {})
            self.beta_params = data.get('beta_params', {})
            
            return True
        except Exception as e:
            self.logger.error(f"加载Thompson采样参数失败: {e}")
            return False
    
    def _update_model(self, query_data: Dict[str, Any], feedback: Dict[str, Any]) -> None:
        """更新模型"""
        super()._update_model(query_data, feedback)
        
        # 更新Beta分布参数
        reward = self._extract_reward(feedback)
        
        for r_id in self.retriever_ids:
            if f'{r_id}_used' in feedback and feedback[f'{r_id}_used']:
                self.alpha_params[r_id] += reward
                self.beta_params[r_id] += (1.0 - reward)
    
    def _select_weights(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Thompson采样选择权重"""
        weights = {}
        
        for r_id in self.retriever_ids:
            # 从Beta分布采样
            alpha = self.alpha_params[r_id]
            beta = self.beta_params[r_id]
            
            # 采样期望值
            sample = np.random.beta(alpha, beta)
            weights[r_id] = sample
        
        return self.normalize_weights(weights)


class UCBOptimizer(BanditOptimizer):
    """UCB优化器"""
    
    def __init__(
        self,
        retriever_ids: List[str],
        confidence: float = 1.0,
        min_weight: float = 0.01,
        window_size: int = 100,
        decay_factor: float = 0.95
    ):
        """
        初始化UCB优化器
        
        Args:
            retriever_ids: 检索器ID列表
            confidence: 置信度参数
            min_weight: 最小权重
            window_size: 滑动窗口大小
            decay_factor: 衰减因子
        """
        super().__init__(retriever_ids, min_weight=min_weight, window_size=window_size, decay_factor=decay_factor)
        
        self.confidence = confidence
    
    def get_state(self) -> Dict[str, Any]:
        """获取优化器状态"""
        state = super().get_state()
        state['confidence'] = self.confidence
        return state
    
    def _update_model(self, query_data: Dict[str, Any], feedback: Dict[str, Any]) -> None:
        """更新模型"""
        super()._update_model(query_data, feedback)
        
        # 更新统计
        reward = self._extract_reward(feedback)
        
        for r_id in self.retriever_ids:
            if f'{r_id}_used' in feedback and feedback[f'{r_id}_used']:
                self.counts[r_id] += 1
                
                # 更新平均值
                current_value = self.values[r_id]
                current_count = self.counts[r_id]
                self.values[r_id] = (current_value * (current_count - 1) + reward) / current_count
                
                # 更新滑动窗口
                self.rewards_window[r_id].append(reward)
                if len(self.rewards_window[r_id]) > self.window_size:
                    self.rewards_window[r_id].pop(0)
    
    def _select_weights(self, features: Dict[str, Any]) -> Dict[str, float]:
        """UCB选择权重"""
        weights = {}
        
        for r_id in self.retriever_ids:
            if self.counts[r_id] == 0:
                # 如果从未选择过，给予高权重
                weights[r_id] = float('inf')
            else:
                # UCB公式
                exploration_bonus = self.confidence * math.sqrt(math.log(self.num_trials) / self.counts[r_id])
                ucb_value = self.values[r_id] + exploration_bonus
                weights[r_id] = ucb_value
        
        return self.normalize_weights(weights) 