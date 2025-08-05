"""
简单但有效的优化器
基于现有优化器模块，添加一些轻量级但有效的优化策略
适合本科生项目的实际需求
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
from collections import defaultdict, deque
import random

@dataclass
class SimpleOptimizerConfig:
    """简单优化器配置"""
    
    # 基础参数
    num_sources: int = 4  # naive, bm25, local, global
    learning_rate: float = 0.01
    momentum: float = 0.9
    
    # 滑动窗口参数
    window_size: int = 100  # 保留最近100次查询的反馈
    min_samples: int = 10   # 最少需要10个样本才开始优化
    
    # 置信度参数
    confidence_threshold: float = 0.6
    uncertainty_penalty: float = 0.1
    
    # 多臂老虎机参数
    epsilon: float = 0.1     # ε-贪心策略
    epsilon_decay: float = 0.99
    ucb_c: float = 1.414     # UCB参数
    
    # 简单规则参数
    enable_simple_rules: bool = True
    query_length_threshold: int = 10  # 长查询阈值

class SimpleMultiArmedBandit:
    """简单的多臂老虎机优化器"""
    
    def __init__(self, config: SimpleOptimizerConfig):
        self.config = config
        self.num_arms = config.num_sources
        
        # 统计信息
        self.counts = np.zeros(self.num_arms)      # 每个臂被选择的次数
        self.rewards = np.zeros(self.num_arms)     # 每个臂的累积奖励
        self.recent_rewards = [deque(maxlen=config.window_size) for _ in range(self.num_arms)]
        
        # ε-贪心参数
        self.epsilon = config.epsilon
        
        # 臂名称映射
        self.arm_names = ["naive", "bm25", "local", "global"]
    
    def select_weights(self, 
                      query_features: Optional[Dict[str, Any]] = None,
                      method: str = "epsilon_greedy") -> np.ndarray:
        """
        选择权重
        
        Args:
            query_features: 查询特征（可选）
            method: 选择方法 ("epsilon_greedy", "ucb", "thompson")
        """
        if method == "epsilon_greedy":
            return self._epsilon_greedy()
        elif method == "ucb":
            return self._upper_confidence_bound()
        elif method == "thompson":
            return self._thompson_sampling()
        else:
            # 默认均匀权重
            return np.ones(self.num_arms) / self.num_arms
    
    def _epsilon_greedy(self) -> np.ndarray:
        """ε-贪心策略"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            weights = np.zeros(self.num_arms)
            weights[np.random.randint(self.num_arms)] = 1.0
        else:
            # 利用：选择最佳臂
            if np.all(self.counts == 0):
                # 如果没有历史数据，均匀分配
                weights = np.ones(self.num_arms) / self.num_arms
            else:
                avg_rewards = np.divide(self.rewards, self.counts, 
                                      out=np.zeros_like(self.rewards), 
                                      where=self.counts!=0)
                best_arm = np.argmax(avg_rewards)
                weights = np.zeros(self.num_arms)
                weights[best_arm] = 1.0
        
        return weights
    
    def _upper_confidence_bound(self) -> np.ndarray:
        """上置信界算法"""
        total_counts = np.sum(self.counts)
        
        if total_counts == 0:
            return np.ones(self.num_arms) / self.num_arms
        
        # 计算UCB值
        avg_rewards = np.divide(self.rewards, self.counts, 
                              out=np.zeros_like(self.rewards), 
                              where=self.counts!=0)
        
        confidence = np.sqrt(np.divide(2 * np.log(total_counts), self.counts,
                                     out=np.full_like(self.counts, float('inf')),
                                     where=self.counts!=0))
        
        ucb_values = avg_rewards + self.config.ucb_c * confidence
        
        # 选择UCB值最高的臂
        best_arm = np.argmax(ucb_values)
        weights = np.zeros(self.num_arms)
        weights[best_arm] = 1.0
        
        return weights
    
    def _thompson_sampling(self) -> np.ndarray:
        """Thompson采样"""
        # 简化的Beta分布采样
        sampled_values = []
        
        for i in range(self.num_arms):
            if len(self.recent_rewards[i]) == 0:
                sampled_values.append(0.5)  # 默认值
            else:
                rewards = list(self.recent_rewards[i])
                alpha = sum(rewards) + 1
                beta = len(rewards) - sum(rewards) + 1
                sampled_values.append(np.random.beta(alpha, beta))
        
        # 选择采样值最高的臂
        best_arm = np.argmax(sampled_values)
        weights = np.zeros(self.num_arms)
        weights[best_arm] = 1.0
        
        return weights
    
    def update(self, weights: np.ndarray, reward: float):
        """
        更新统计信息
        
        Args:
            weights: 使用的权重
            reward: 获得的奖励
        """
        # 找到主要使用的臂（权重最大的）
        main_arm = np.argmax(weights)
        
        # 更新统计
        self.counts[main_arm] += 1
        self.rewards[main_arm] += reward
        self.recent_rewards[main_arm].append(reward)
        
        # 衰减epsilon
        self.epsilon *= self.config.epsilon_decay
        self.epsilon = max(self.epsilon, 0.01)  # 最小值

class SimpleRuleBasedOptimizer:
    """基于简单规则的优化器"""
    
    def __init__(self, config: SimpleOptimizerConfig):
        self.config = config
        
        # 规则权重映射
        self.rule_weights = {
            "short_query": np.array([0.4, 0.4, 0.1, 0.1]),  # 短查询偏向关键词和向量检索
            "long_query": np.array([0.2, 0.2, 0.3, 0.3]),   # 长查询偏向图推理
            "factual_query": np.array([0.3, 0.5, 0.1, 0.1]), # 事实性查询偏向BM25
            "analytical_query": np.array([0.1, 0.1, 0.4, 0.4]), # 分析性查询偏向图推理
            "default": np.array([0.25, 0.25, 0.25, 0.25])    # 默认均匀分配
        }
    
    def select_weights(self, query_features: Dict[str, Any]) -> np.ndarray:
        """基于规则选择权重"""
        if not self.config.enable_simple_rules:
            return self.rule_weights["default"]
        
        # 提取查询特征
        query_length = query_features.get("length", 0)
        query_type = query_features.get("type", "unknown")
        complexity = query_features.get("complexity", "one_hop")
        
        # 应用规则
        if query_length < self.config.query_length_threshold:
            weights = self.rule_weights["short_query"]
        else:
            weights = self.rule_weights["long_query"]
        
        # 根据查询类型调整
        if query_type in ["factual", "definition"]:
            weights = 0.7 * weights + 0.3 * self.rule_weights["factual_query"]
        elif query_type in ["analytical", "comparison"]:
            weights = 0.7 * weights + 0.3 * self.rule_weights["analytical_query"]
        
        # 根据复杂度调整
        if complexity == "zero_hop":
            # 零跳查询：更偏向直接检索
            weights[0] += 0.1  # naive
            weights[1] += 0.1  # bm25
            weights[2] -= 0.1  # local
            weights[3] -= 0.1  # global
        elif complexity == "multi_hop":
            # 多跳查询：更偏向图推理
            weights[0] -= 0.1  # naive
            weights[1] -= 0.1  # bm25
            weights[2] += 0.1  # local
            weights[3] += 0.1  # global
        
        # 确保权重为正且和为1
        weights = np.maximum(weights, 0.01)
        weights = weights / np.sum(weights)
        
        return weights

class SimpleGradientOptimizer:
    """简单的梯度优化器"""
    
    def __init__(self, config: SimpleOptimizerConfig):
        self.config = config
        
        # 权重参数
        self.weights = np.ones(config.num_sources) / config.num_sources
        self.momentum = np.zeros(config.num_sources)
        
        # 历史记录
        self.history = deque(maxlen=config.window_size)
    
    def select_weights(self, query_features: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """选择当前权重"""
        return self.weights.copy()
    
    def update(self, query_features: Dict[str, Any], 
               source_scores: Dict[str, float], 
               user_feedback: float):
        """
        更新权重
        
        Args:
            query_features: 查询特征
            source_scores: 各源的得分
            user_feedback: 用户反馈 (0-1)
        """
        # 记录历史
        self.history.append({
            'features': query_features,
            'scores': source_scores,
            'feedback': user_feedback,
            'weights': self.weights.copy()
        })
        
        if len(self.history) < self.config.min_samples:
            return
        
        # 计算梯度
        gradients = self._compute_gradients()
        
        # 更新权重
        self.momentum = (self.config.momentum * self.momentum + 
                        self.config.learning_rate * gradients)
        
        self.weights += self.momentum
        
        # 确保权重约束
        self.weights = np.maximum(self.weights, 0.01)
        self.weights = self.weights / np.sum(self.weights)
    
    def _compute_gradients(self) -> np.ndarray:
        """计算简单的梯度"""
        gradients = np.zeros(self.config.num_sources)
        
        # 基于最近的反馈计算梯度
        recent_samples = list(self.history)[-min(20, len(self.history)):]
        
        for sample in recent_samples:
            feedback = sample['feedback']
            weights = sample['weights']
            scores = sample['scores']
            
            # 简单的梯度：高反馈时增加当前权重，低反馈时减少
            feedback_signal = feedback - 0.5  # 中心化到[-0.5, 0.5]
            
            for i, source in enumerate(["naive", "bm25", "local", "global"]):
                source_score = scores.get(source, 0.5)
                # 梯度 = 反馈信号 * 源得分 * 当前权重
                gradients[i] += feedback_signal * source_score * weights[i]
        
        # 归一化梯度
        if len(recent_samples) > 0:
            gradients /= len(recent_samples)
        
        return gradients

class SimpleOptimizer:
    """简单集成优化器"""
    
    def __init__(self, config: Optional[SimpleOptimizerConfig] = None):
        self.config = config or SimpleOptimizerConfig()
        
        # 子优化器
        self.bandit_optimizer = SimpleMultiArmedBandit(self.config)
        self.rule_optimizer = SimpleRuleBasedOptimizer(self.config)
        self.gradient_optimizer = SimpleGradientOptimizer(self.config)
        
        # 优化器选择权重
        self.optimizer_weights = np.array([0.4, 0.4, 0.2])  # bandit, rule, gradient
        
        # 性能追踪
        self.performance_history = deque(maxlen=self.config.window_size)
    
    def get_optimal_weights(self, 
                           query_features: Dict[str, Any],
                           method: str = "ensemble") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        获取最优权重
        
        Args:
            query_features: 查询特征
            method: 优化方法 ("bandit", "rule", "gradient", "ensemble")
        """
        info = {"method": method}
        
        if method == "bandit":
            weights = self.bandit_optimizer.select_weights(query_features)
            info["optimizer"] = "bandit"
        elif method == "rule":
            weights = self.rule_optimizer.select_weights(query_features)
            info["optimizer"] = "rule"
        elif method == "gradient":
            weights = self.gradient_optimizer.select_weights(query_features)
            info["optimizer"] = "gradient"
        else:  # ensemble
            # 集成多个优化器的结果
            bandit_weights = self.bandit_optimizer.select_weights(query_features)
            rule_weights = self.rule_optimizer.select_weights(query_features)
            gradient_weights = self.gradient_optimizer.select_weights(query_features)
            
            # 加权组合
            weights = (self.optimizer_weights[0] * bandit_weights + 
                      self.optimizer_weights[1] * rule_weights + 
                      self.optimizer_weights[2] * gradient_weights)
            
            info.update({
                "optimizer": "ensemble",
                "bandit_weights": bandit_weights,
                "rule_weights": rule_weights,
                "gradient_weights": gradient_weights,
                "ensemble_weights": self.optimizer_weights
            })
        
        # 确保权重约束
        weights = np.maximum(weights, 0.01)
        weights = weights / np.sum(weights)
        
        return weights, info
    
    def update_feedback(self, 
                       query_features: Dict[str, Any],
                       weights_used: np.ndarray,
                       source_scores: Dict[str, float],
                       user_feedback: float,
                       performance_metrics: Optional[Dict[str, float]] = None):
        """
        更新反馈
        
        Args:
            query_features: 查询特征
            weights_used: 使用的权重
            source_scores: 各源得分
            user_feedback: 用户反馈
            performance_metrics: 性能指标
        """
        # 更新各个优化器
        self.bandit_optimizer.update(weights_used, user_feedback)
        self.gradient_optimizer.update(query_features, source_scores, user_feedback)
        
        # 记录性能历史
        self.performance_history.append({
            'query_features': query_features,
            'weights_used': weights_used,
            'source_scores': source_scores,
            'user_feedback': user_feedback,
            'performance_metrics': performance_metrics or {}
        })
        
        # 自适应调整优化器权重
        self._adjust_optimizer_weights()
    
    def _adjust_optimizer_weights(self):
        """自适应调整优化器权重"""
        if len(self.performance_history) < 20:
            return
        
        # 计算最近的性能
        recent_performance = list(self.performance_history)[-10:]
        avg_feedback = np.mean([p['user_feedback'] for p in recent_performance])
        
        # 简单的自适应策略
        if avg_feedback > 0.7:
            # 性能好，稍微减少exploration
            self.optimizer_weights[0] -= 0.01  # 减少bandit
            self.optimizer_weights[2] += 0.01  # 增加gradient
        elif avg_feedback < 0.4:
            # 性能差，增加exploration
            self.optimizer_weights[0] += 0.01  # 增加bandit
            self.optimizer_weights[2] -= 0.01  # 减少gradient
        
        # 确保权重约束
        self.optimizer_weights = np.maximum(self.optimizer_weights, 0.1)
        self.optimizer_weights = self.optimizer_weights / np.sum(self.optimizer_weights)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        stats = {
            "total_queries": len(self.performance_history),
            "optimizer_weights": self.optimizer_weights.tolist(),
            "bandit_stats": {
                "arm_counts": self.bandit_optimizer.counts.tolist(),
                "arm_rewards": self.bandit_optimizer.rewards.tolist(),
                "epsilon": self.bandit_optimizer.epsilon
            }
        }
        
        if len(self.performance_history) > 0:
            recent_feedback = [p['user_feedback'] for p in self.performance_history]
            stats["performance"] = {
                "avg_feedback": np.mean(recent_feedback),
                "std_feedback": np.std(recent_feedback),
                "trend": np.mean(recent_feedback[-10:]) - np.mean(recent_feedback[:10]) 
                        if len(recent_feedback) >= 20 else 0
            }
        
        return stats

def create_simple_optimizer(config: Optional[SimpleOptimizerConfig] = None) -> SimpleOptimizer:
    """创建简单优化器"""
    return SimpleOptimizer(config)

def create_simple_optimizer_config(**kwargs) -> SimpleOptimizerConfig:
    """创建简单优化器配置"""
    return SimpleOptimizerConfig(**kwargs) 