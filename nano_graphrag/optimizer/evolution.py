"""
进化算法优化模块

提供基于进化算法的权重优化功能
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

from .learning import WeightOptimizer


class EvolutionaryOptimizer(WeightOptimizer):
    """进化算法优化器"""
    
    def __init__(
        self,
        retriever_ids: List[str],
        population_size: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        generations: int = 5,
        min_weight: float = 0.01,
        history_size: int = 50
    ):
        """
        初始化进化算法优化器
        
        Args:
            retriever_ids: 检索器ID列表
            population_size: 种群大小
            mutation_rate: 变异率
            crossover_rate: 交叉率
            tournament_size: 锦标赛大小
            generations: 迭代代数
            min_weight: 最小权重
            history_size: 历史记录大小
        """
        super().__init__(retriever_ids)
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.generations = generations
        self.min_weight = min_weight
        self.history_size = history_size
        
        # 初始化种群
        self.population = self._initialize_population()
        
        # 最佳个体
        self.best_individual = self.default_weights.copy()
        self.best_fitness = 0.0
        
        # 记录适应度历史
        self.fitness_history = []
        
        # 查询历史
        self.query_history = []
        
        # 特征类型映射
        self.feature_fitness = {}
        
        # 当前个体
        self.current_individual = None
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取优化器状态
        
        Returns:
            状态字典
        """
        return {
            'type': 'EvolutionaryOptimizer',
            'retriever_ids': self.retriever_ids,
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'tournament_size': self.tournament_size,
            'generations': self.generations,
            'min_weight': self.min_weight,
            'population': self.population,
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'query_history': self.query_history,
            'feature_fitness': self.feature_fitness
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
            
            if data.get('type') != 'EvolutionaryOptimizer':
                self.logger.error("文件类型不匹配")
                return False
            
            self.population = data.get('population', [])
            self.best_individual = data.get('best_individual', {})
            self.best_fitness = data.get('best_fitness', 0.0)
            self.fitness_history = data.get('fitness_history', [])
            self.query_history = data.get('query_history', [])
            self.feature_fitness = data.get('feature_fitness', {})
            
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
        # 如果有反馈，更新模型
        if feedback:
            self._update_population(query_data, feedback)
        
        # 选择最佳个体
        weights = self._select_individual()
        
        return self.constrain_weights(weights, self.min_weight)
    
    def _initialize_population(self) -> List[Dict[str, float]]:
        """
        初始化种群
        
        Returns:
            初始种群
        """
        population = []
        
        for _ in range(self.population_size):
            # 随机生成权重
            weights = {}
            for r_id in self.retriever_ids:
                weights[r_id] = random.uniform(0.1, 1.0)
            
            # 归一化
            weights = self.normalize_weights(weights)
            population.append(weights)
        
        return population
    
    def _update_population(self, query_data: Dict[str, Any], feedback: Dict[str, Any]) -> None:
        """
        更新种群
        
        Args:
            query_data: 查询数据
            feedback: 反馈数据
        """
        # 提取适应度
        fitness = self._extract_fitness(feedback)
        
        if fitness is None:
            return
        
        # 添加到历史记录
        self.query_history.append({
            'query_data': query_data,
            'fitness': fitness,
            'timestamp': datetime.now().isoformat()
        })
        
        # 限制历史记录大小
        if len(self.query_history) > self.history_size:
            self.query_history = self.query_history[-self.history_size:]
        
        # 更新最佳个体
        if fitness > self.best_fitness:
            self.best_individual = self.default_weights.copy()  # 当前权重
            self.best_fitness = fitness
        
        # 进化
        self._evolve()
    
    def _extract_fitness(self, feedback: Dict[str, Any]) -> Optional[float]:
        """
        提取适应度
        
        Args:
            feedback: 反馈数据
            
        Returns:
            适应度值
        """
        # 基于用户评分
        user_rating = feedback.get('user_rating', 0.5)
        
        # 基于检索质量
        retrieval_quality = feedback.get('retrieval_quality', 0.5)
        
        # 综合适应度
        fitness = 0.7 * user_rating + 0.3 * retrieval_quality
        
        return max(0.0, min(1.0, fitness))
    
    def _evolve(self) -> None:
        """进化过程"""
        # 计算适应度
        fitness_scores = self._calculate_fitness()
        
        # 生成新种群
        new_population = []
        
        while len(new_population) < self.population_size:
            # 锦标赛选择
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)
            
            # 交叉
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # 变异
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
    
    def _calculate_fitness(self) -> Dict[Dict[str, float], float]:
        """
        计算适应度
        
        Returns:
            适应度字典
        """
        fitness_scores = {}
        
        for individual in self.population:
            # 基于历史数据计算适应度
            total_fitness = 0.0
            count = 0
            
            for record in self.query_history:
                # 简单的相似度计算
                similarity = 0.5  # 简化处理
                fitness = record['fitness'] * similarity
                total_fitness += fitness
                count += 1
            
            if count > 0:
                avg_fitness = total_fitness / count
            else:
                avg_fitness = 0.5  # 默认适应度
            
            fitness_scores[tuple(individual.items())] = avg_fitness
        
        return fitness_scores
    
    def _tournament_selection(self, fitness_scores: Dict[Dict[str, float], float]) -> Dict[str, float]:
        """
        锦标赛选择
        
        Args:
            fitness_scores: 适应度分数
            
        Returns:
            选择的个体
        """
        tournament = random.sample(list(fitness_scores.keys()), min(self.tournament_size, len(fitness_scores)))
        
        best_individual = None
        best_fitness = -1.0
        
        for individual_tuple in tournament:
            individual = dict(individual_tuple)
            fitness = fitness_scores[individual_tuple]
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual
        
        return best_individual or self.default_weights.copy()
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """
        交叉操作
        
        Args:
            parent1: 父代1
            parent2: 父代2
            
        Returns:
            子代
        """
        child = {}
        
        for r_id in self.retriever_ids:
            # 加权平均
            weight = random.random()
            child[r_id] = weight * parent1[r_id] + (1 - weight) * parent2[r_id]
        
        return self.normalize_weights(child)
    
    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """
        变异操作
        
        Args:
            individual: 个体
            
        Returns:
            变异后的个体
        """
        mutated = individual.copy()
        
        for r_id in self.retriever_ids:
            if random.random() < 0.3:  # 变异概率
                # 添加随机噪声
                noise = random.uniform(-0.1, 0.1)
                mutated[r_id] = max(0.0, mutated[r_id] + noise)
        
        return self.normalize_weights(mutated)
    
    def _select_individual(self) -> Dict[str, float]:
        """
        选择个体
        
        Returns:
            选择的个体
        """
        if not self.population:
            return self.default_weights.copy()
        
        # 选择最佳个体
        return self.best_individual.copy() 