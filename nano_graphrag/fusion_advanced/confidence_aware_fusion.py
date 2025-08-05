"""
置信度感知的Multi-Head Attention Fusion模块

基于现有的MultiHeadAttentionFusion，增加置信度权重调节功能。
这是精简方案的第三个核心创新点。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from .multihead_attention_fusion import (
    MultiHeadAttentionFusion, 
    MultiHeadAttentionFusionConfig,
    create_multihead_attention_fusion_config
)

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceAwareFusionConfig(MultiHeadAttentionFusionConfig):
    """置信度感知融合配置（继承基础配置）"""
    
    # 置信度权重调节参数
    enable_confidence_weighting: bool = True
    confidence_temperature: float = 2.0       # 置信度温度参数
    confidence_min_weight: float = 0.1        # 最小置信度权重
    confidence_max_weight: float = 2.0        # 最大置信度权重
    
    # 动态权重调节
    enable_dynamic_weighting: bool = True
    dynamic_weight_scale: float = 1.5         # 动态权重缩放因子
    
    # 不确定性门控
    enable_uncertainty_gating: bool = True
    uncertainty_threshold: float = 0.5        # 不确定性阈值
    uncertainty_gate_dim: int = 64            # 不确定性门控维度


class ConfidenceWeightingModule(nn.Module):
    """置信度权重调节模块"""
    
    def __init__(self, config: ConfidenceAwareFusionConfig):
        super().__init__()
        self.config = config
        
        # 置信度转换网络（可选）
        if config.enable_dynamic_weighting:
            self.confidence_transform = nn.Sequential(
                nn.Linear(1, config.d_model // 4),
                nn.ReLU(),
                nn.Linear(config.d_model // 4, 1),
                nn.Sigmoid()
            )
        
        # 不确定性门控网络
        if config.enable_uncertainty_gating:
            self.uncertainty_gate = nn.Sequential(
                nn.Linear(1, config.uncertainty_gate_dim),
                nn.ReLU(),
                nn.Linear(config.uncertainty_gate_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, 
                attention_scores: torch.Tensor,
                confidence_scores: Dict[str, float],
                source_names: List[str],
                uncertainty_score: Optional[float] = None) -> torch.Tensor:
        """
        应用置信度权重调节
        
        Args:
            attention_scores: 原始注意力分数 [batch_size, num_heads, seq_len, seq_len] 或 [batch_size, seq_len]
            confidence_scores: 各源的置信度分数 {"bgem3": 0.8, "bm25": 0.6, ...}
            source_names: 源名称列表
            uncertainty_score: 整体不确定性分数
            
        Returns:
            调节后的注意力分数
        """
        # 处理不同维度的注意力分数
        if attention_scores.dim() == 4:
            batch_size, num_heads, query_len, key_len = attention_scores.shape
        elif attention_scores.dim() == 3:
            batch_size, query_len, key_len = attention_scores.shape
            num_heads = 1
            attention_scores = attention_scores.unsqueeze(1)  # 添加head维度
        elif attention_scores.dim() == 2:
            query_len, key_len = attention_scores.shape
            batch_size = 1
            num_heads = 1
            attention_scores = attention_scores.unsqueeze(0).unsqueeze(0)  # 添加batch和head维度
        else:
            raise ValueError(f"Unsupported attention_scores shape: {attention_scores.shape}")
        
        # 1. 构建置信度权重向量
        confidence_weights = self._build_confidence_weights(confidence_scores, source_names, key_len)
        confidence_weights = confidence_weights.to(attention_scores.device)
        
        # 2. 动态权重调节（可选）
        if self.config.enable_dynamic_weighting and hasattr(self, 'confidence_transform'):
            avg_confidence = torch.tensor([sum(confidence_scores.values()) / len(confidence_scores)], 
                                        dtype=torch.float32, device=attention_scores.device)
            dynamic_factor = self.confidence_transform(avg_confidence.unsqueeze(0))
            confidence_weights = confidence_weights * (dynamic_factor.item() * self.config.dynamic_weight_scale)
        
        # 3. 应用置信度权重（广播到所有头和batch）
        confidence_weights = confidence_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, key_len]
        confidence_weights = confidence_weights.expand(batch_size, num_heads, -1)  # [batch, heads, key_len]
        confidence_weights = confidence_weights.unsqueeze(2)  # [batch, heads, 1, key_len]
        
        # 调节注意力分数
        adjusted_scores = attention_scores * confidence_weights
        
        # 4. 不确定性门控（可选）
        if (self.config.enable_uncertainty_gating and 
            hasattr(self, 'uncertainty_gate') and 
            uncertainty_score is not None):
            
            uncertainty_tensor = torch.tensor([uncertainty_score], 
                                            dtype=torch.float32, device=attention_scores.device)
            gate_value = self.uncertainty_gate(uncertainty_tensor.unsqueeze(0))
            
            # 高不确定性时降低调节强度
            if uncertainty_score > self.config.uncertainty_threshold:
                gate_factor = gate_value.item()
                adjusted_scores = attention_scores + (adjusted_scores - attention_scores) * gate_factor
        
        return adjusted_scores
    
    def _build_confidence_weights(self, 
                                 confidence_scores: Dict[str, float], 
                                 source_names: List[str], 
                                 seq_len: int) -> torch.Tensor:
        """
        构建置信度权重向量
        
        Args:
            confidence_scores: 置信度分数字典
            source_names: 源名称列表
            seq_len: 序列长度
            
        Returns:
            置信度权重向量 [seq_len]
        """
        # 简化版本：假设每个文档对应一个源，权重均匀分布
        default_confidence = 0.5
        weights = []
        
        # 计算平均置信度
        if confidence_scores:
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        else:
            avg_confidence = default_confidence
        
        # 应用温度缩放和限制
        scaled_confidence = torch.sigmoid(torch.tensor(avg_confidence) / self.config.confidence_temperature)
        
        # 限制权重范围
        weight = scaled_confidence.item() * (self.config.confidence_max_weight - self.config.confidence_min_weight) + self.config.confidence_min_weight
        
        # 为所有位置分配相同权重（简化版本）
        weights = torch.full((seq_len,), weight, dtype=torch.float32)
        
        return weights


class ConfidenceAwareFusion(MultiHeadAttentionFusion):
    """
    置信度感知的Multi-Head Attention Fusion
    
    继承基础的MultiHeadAttentionFusion，增加置信度权重调节功能
    """
    
    def __init__(self, config: ConfidenceAwareFusionConfig):
        # 先初始化基类
        super().__init__(config)
        self.confidence_config = config
        
        # 添加置信度权重调节模块
        if config.enable_confidence_weighting:
            self.confidence_weighting = ConfidenceWeightingModule(config)
        
        logger.info("ConfidenceAwareFusion初始化完成")
    
    def forward(self, 
                query_embedding: torch.Tensor,
                document_embeddings: torch.Tensor,
                source_names: List[str],
                confidence_scores: Optional[Dict[str, float]] = None,
                uncertainty_score: Optional[float] = None,
                document_ranks: Optional[torch.Tensor] = None,
                document_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        置信度感知的前向传播
        
        Args:
            query_embedding: 查询嵌入
            document_embeddings: 文档嵌入
            source_names: 源名称列表
            confidence_scores: 置信度分数字典（新增）
            uncertainty_score: 不确定性分数（新增）
            document_ranks: 文档排名
            document_mask: 文档掩码
            return_attention: 是否返回注意力权重
            
        Returns:
            融合结果字典，包含置信度调节信息
        """
        # 1. 调用基类的forward方法获取基础注意力结果
        base_results = super().forward(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
            source_names=source_names,
            document_ranks=document_ranks,
            document_mask=document_mask,
            return_attention=True  # 强制返回注意力权重用于调节
        )
        
        # 2. 如果启用置信度权重调节且有置信度分数
        if (self.confidence_config.enable_confidence_weighting and 
            hasattr(self, 'confidence_weighting') and 
            confidence_scores is not None):
            
            # 调节交叉注意力权重
            if 'cross_attention_weights' in base_results:
                original_cross_attention = base_results['cross_attention_weights']
                adjusted_cross_attention = self.confidence_weighting(
                    attention_scores=original_cross_attention,
                    confidence_scores=confidence_scores,
                    source_names=source_names,
                    uncertainty_score=uncertainty_score
                )
                base_results['adjusted_cross_attention_weights'] = adjusted_cross_attention
                
                # 记录调节统计信息
                base_results['confidence_adjustment_stats'] = {
                    'applied_confidence_scores': confidence_scores,
                    'uncertainty_score': uncertainty_score,
                    'adjustment_strength': self._compute_adjustment_strength(
                        original_cross_attention, adjusted_cross_attention
                    )
                }
            
            # 调节自注意力权重
            if 'self_attention_weights' in base_results:
                original_self_attention = base_results['self_attention_weights']
                if isinstance(original_self_attention, list) and len(original_self_attention) > 0:
                    # 只调节最后一层的自注意力
                    last_layer_attention = original_self_attention[-1]
                    adjusted_self_attention = self.confidence_weighting(
                        attention_scores=last_layer_attention,
                        confidence_scores=confidence_scores,
                        source_names=source_names,
                        uncertainty_score=uncertainty_score
                    )
                    base_results['adjusted_self_attention_weights'] = adjusted_self_attention
        
        # 3. 决定返回哪些注意力权重
        if not return_attention:
            # 移除注意力权重（但保留调节后的版本供内部使用）
            attention_keys = ['cross_attention_weights', 'self_attention_weights']
            for key in attention_keys:
                if key in base_results:
                    del base_results[key]
        
        return base_results
    
    def _compute_adjustment_strength(self, 
                                   original_attention: torch.Tensor,
                                   adjusted_attention: torch.Tensor) -> float:
        """
        计算置信度调节强度
        
        Args:
            original_attention: 原始注意力权重
            adjusted_attention: 调节后的注意力权重
            
        Returns:
            调节强度（L2距离）
        """
        try:
            diff = original_attention - adjusted_attention
            adjustment_strength = torch.norm(diff).item()
            return adjustment_strength
        except Exception as e:
            logger.warning(f"计算调节强度失败: {e}")
            return 0.0
    
    def get_confidence_impact_analysis(self,
                                     confidence_scores: Dict[str, float],
                                     base_attention: torch.Tensor,
                                     adjusted_attention: torch.Tensor) -> Dict[str, Any]:
        """
        分析置信度对注意力的影响
        
        Args:
            confidence_scores: 置信度分数
            base_attention: 基础注意力权重
            adjusted_attention: 调节后的注意力权重
            
        Returns:
            影响分析结果
        """
        try:
            # 计算注意力分布变化
            attention_shift = torch.abs(adjusted_attention - base_attention).mean().item()
            
            # 计算权重方差变化
            base_variance = torch.var(base_attention).item()
            adjusted_variance = torch.var(adjusted_attention).item()
            variance_change = adjusted_variance - base_variance
            
            # 高低置信度影响分析
            high_confidence_sources = [k for k, v in confidence_scores.items() if v > 0.7]
            low_confidence_sources = [k for k, v in confidence_scores.items() if v < 0.3]
            
            analysis = {
                'attention_shift': attention_shift,
                'variance_change': variance_change,
                'high_confidence_sources': high_confidence_sources,
                'low_confidence_sources': low_confidence_sources,
                'average_confidence': sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0,
                'confidence_range': max(confidence_scores.values()) - min(confidence_scores.values()) if confidence_scores else 0.0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"置信度影响分析失败: {e}")
            return {}


def create_confidence_aware_fusion(config: Optional[ConfidenceAwareFusionConfig] = None) -> ConfidenceAwareFusion:
    """创建置信度感知融合器的工厂函数"""
    if config is None:
        # 使用默认配置
        base_config = create_multihead_attention_fusion_config()
        config = ConfidenceAwareFusionConfig(**base_config.__dict__)
    
    return ConfidenceAwareFusion(config)


def create_confidence_aware_fusion_config(**kwargs) -> ConfidenceAwareFusionConfig:
    """创建置信度感知融合配置的工厂函数"""
    # 先创建基础配置
    base_config = create_multihead_attention_fusion_config()
    
    # 合并基础配置和新参数
    config_dict = base_config.__dict__.copy()
    config_dict.update(kwargs)
    
    return ConfidenceAwareFusionConfig(**config_dict)


# 导出所有公共接口
__all__ = [
    'ConfidenceAwareFusionConfig',
    'ConfidenceWeightingModule', 
    'ConfidenceAwareFusion',
    'create_confidence_aware_fusion',
    'create_confidence_aware_fusion_config'
] 