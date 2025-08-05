"""
复杂度感知路由器模块
基于查询复杂度自动选择最佳检索策略
"""

import asyncio
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .classifier import ComplexityClassifier, ComplexityClassifierConfig
from .._utils import logger

class BaseRouter:
    """路由器基类 - 提供基本的路由接口"""
    
    def __init__(self):
        pass
    
    async def route(self, query: str, available_modes: List[str] = None) -> str:
        """路由查询到检索模式"""
        raise NotImplementedError
    
    def create_query_param(self, query: str, available_modes: List[str] = None, **kwargs) -> Any:
        """创建查询参数"""
        raise NotImplementedError

@dataclass
class ComplexityAwareRouter(BaseRouter):
    """复杂度感知路由器
    
    基于ModernBERT复杂度分类器，将查询路由到最适合的检索模式。
    """
    
    model_path: str = "nano_graphrag/models/modernbert_complexity_classifier"
    confidence_threshold: float = 0.6
    enable_fallback: bool = True
    use_modernbert: bool = True
    
    def __post_init__(self):
        """初始化路由器"""
        super().__init__()
        
        # 复杂度到检索模式的映射
        self._complexity_to_candidate = {
            "zero_hop": ["llm_only"],              # 无检索：直接LLM回答
            "one_hop": ["naive", "bm25"],          # 单跳检索：向量检索或关键词检索
            "multi_hop": ["local", "global"],      # 多跳检索：图推理
        }
        
        # 初始化分类器
        self.classifier = None
        if self.use_modernbert:
            try:
                config = ComplexityClassifierConfig(
                    model_path=self.model_path,
                    confidence_threshold=self.confidence_threshold
                )
                self.classifier = ComplexityClassifier(config)
                logger.info(f"复杂度分类器初始化成功: {self.model_path}")
            except Exception as e:
                logger.warning(f"复杂度分类器初始化失败: {e}")
                self.classifier = None
        
        # 统计信息
        self._complexity_stats = {
            "zero_hop": 0,
            "one_hop": 0,
            "multi_hop": 0,
            "fallback": 0
        }
    
    async def predict_complexity_detailed(self, query: str) -> Dict[str, Any]:
        """预测查询复杂度（详细版本）"""
        try:
            if self.classifier and self.classifier.is_available():
                # 使用ModernBERT分类器
                complexity, confidence, probabilities = self.classifier.predict_with_confidence(query)
                
                # 更新统计
                self._complexity_stats[complexity] += 1
                
                # 获取候选模式
                candidate_modes = self._complexity_to_candidate.get(complexity, ["naive"])
                
                return {
                    "complexity": complexity,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "candidate_modes": candidate_modes,
                    "method": "modernbert"
                }
            else:
                # 回退到规则分类
                return await self._rule_based_complexity(query)
                
        except Exception as e:
            logger.error(f"复杂度预测失败: {e}")
            return await self._rule_based_complexity(query)
    
    async def predict_complexity(self, query: str) -> str:
        """预测查询复杂度（简化版本）"""
        result = await self.predict_complexity_detailed(query)
        return result["complexity"]
    
    async def _rule_based_complexity(self, query: str) -> Dict[str, Any]:
        """基于规则的复杂度分类"""
        # 简单的规则分类
        query_lower = query.lower()
        
        # zero-hop 规则
        if any(word in query_lower for word in ["what is", "define", "explain", "describe"]):
            if len(query.split()) <= 5:
                complexity = "zero_hop"
            else:
                complexity = "one_hop"
        # multi-hop 规则
        elif any(word in query_lower for word in ["compare", "relationship", "difference", "similarity", "how does", "why does"]):
            complexity = "multi_hop"
        else:
            complexity = "one_hop"
        
        self._complexity_stats[complexity] += 1
        self._complexity_stats["fallback"] += 1
        
        candidate_modes = self._complexity_to_candidate.get(complexity, ["naive"])
        
        return {
            "complexity": complexity,
            "confidence": 0.5,  # 规则分类的置信度较低
            "probabilities": {},
            "candidate_modes": candidate_modes,
            "method": "rule_based"
        }
    
    async def route(self, query: str, available_modes: List[str] = None) -> str:
        """路由查询到最佳检索模式"""
        if not available_modes:
            available_modes = ["llm_only", "naive", "bm25", "local", "global"]
        
        # 预测复杂度
        complexity_result = await self.predict_complexity_detailed(query)
        complexity = complexity_result["complexity"]
        confidence = complexity_result["confidence"]
        
        # 获取候选模式
        candidate_modes = complexity_result["candidate_modes"]
        
        # 从候选模式中选择可用的模式
        available_candidates = [mode for mode in candidate_modes if mode in available_modes]
        
        if not available_candidates:
            # 如果没有可用的候选模式，使用第一个可用模式
            logger.warning(f"复杂度 {complexity} 的候选模式 {candidate_modes} 都不可用，使用 {available_modes[0]}")
            return available_modes[0]
        
        # 如果置信度低于阈值且启用了回退，使用规则分类
        if confidence < self.confidence_threshold and self.enable_fallback:
            logger.info(f"置信度 {confidence:.3f} 低于阈值 {self.confidence_threshold}，使用规则回退")
            fallback_result = await self._rule_based_complexity(query)
            fallback_candidates = [mode for mode in fallback_result["candidate_modes"] if mode in available_modes]
            if fallback_candidates:
                return fallback_candidates[0]
        
        # 返回第一个可用的候选模式
        selected_mode = available_candidates[0]
        logger.info(f"查询复杂度: {complexity}, 置信度: {confidence:.3f}, 选择模式: {selected_mode}")
        
        return selected_mode
    
    def create_query_param(self, query: str, available_modes: List[str] = None, **kwargs) -> Any:
        """创建查询参数"""
        from ..base import QueryParam
        
        # 同步路由 - 优先使用训练好的分类器
        try:
            # 首先尝试使用训练好的ModernBERT分类器
            if self.classifier and self.classifier.is_available():
                try:
                    # 使用同步方式调用分类器
                    complexity, confidence, probabilities = self.classifier.predict_with_confidence(query)
                    
                    # 更新统计
                    self._complexity_stats[complexity] += 1
                    
                    # 获取候选模式
                    candidate_modes = self._complexity_to_candidate.get(complexity, ["naive"])
                    
                    logger.info(f"ModernBERT分类器预测 - 复杂度: {complexity}, 置信度: {confidence:.3f}")
                    
                except Exception as e:
                    logger.warning(f"ModernBERT分类器预测失败，使用规则分类: {e}")
                    complexity_result = self._rule_based_complexity_sync(query)
                    complexity = complexity_result["complexity"]
                    candidate_modes = complexity_result["candidate_modes"]
                    confidence = complexity_result["confidence"]
            else:
                # 使用规则分类
                complexity_result = self._rule_based_complexity_sync(query)
                complexity = complexity_result["complexity"]
                candidate_modes = complexity_result["candidate_modes"]
                confidence = complexity_result["confidence"]
            
            # 从候选模式中选择可用的模式
            if not available_modes:
                available_modes = ["llm_only", "naive", "bm25", "local", "global"]
            
            available_candidates = [mode for mode in candidate_modes if mode in available_modes]
            
            if not available_candidates:
                # 如果没有可用的候选模式，使用第一个可用模式
                logger.warning(f"复杂度 {complexity} 的候选模式 {candidate_modes} 都不可用，使用 {available_modes[0]}")
                selected_mode = available_modes[0]
            else:
                selected_mode = available_candidates[0]
                
            logger.info(f"同步路由 - 查询复杂度: {complexity}, 置信度: {confidence:.3f}, 选择模式: {selected_mode}")
            
        except Exception as e:
            logger.warning(f"同步路由失败，使用默认模式: {e}")
            selected_mode = "naive" if not available_modes else available_modes[0]
        
        return QueryParam(
            mode=selected_mode,
            **kwargs
        )
    
    def _rule_based_complexity_sync(self, query: str) -> Dict[str, Any]:
        """基于规则的复杂度分类（同步版本）"""
        # 简单的规则分类
        query_lower = query.lower()
        
        # zero-hop 规则
        if any(word in query_lower for word in ["what is", "define", "explain", "describe"]):
            if len(query.split()) <= 5:
                complexity = "zero_hop"
            else:
                complexity = "one_hop"
        # multi-hop 规则
        elif any(word in query_lower for word in ["compare", "relationship", "difference", "similarity", "how does", "why does"]):
            complexity = "multi_hop"
        else:
            complexity = "one_hop"
        
        self._complexity_stats[complexity] += 1
        self._complexity_stats["fallback"] += 1
        
        candidate_modes = self._complexity_to_candidate.get(complexity, ["naive"])
        
        return {
            "complexity": complexity,
            "confidence": 0.5,  # 规则分类的置信度较低
            "probabilities": {},
            "candidate_modes": candidate_modes,
            "method": "rule_based_sync"
        }
    
    def get_complexity_stats(self) -> Dict[str, int]:
        """获取复杂度统计信息"""
        return self._complexity_stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self._complexity_stats = {
            "zero_hop": 0,
            "one_hop": 0,
            "multi_hop": 0,
            "fallback": 0
        } 