"""
复杂度感知图神经网络层

这是项目的核心创新算法，将查询复杂度信息融入GraphSAGE中，
实现复杂度感知的图表示学习。

创新点:
1. 复杂度嵌入：将zero-hop/one-hop/multi-hop复杂度编码为嵌入向量
2. 复杂度调制：使用复杂度信息调制图神经网络的聚合过程
3. 自适应权重：根据复杂度动态调整邻居重要性
4. 门控机制：复杂度门控的特征融合

科研贡献:
- 首创查询复杂度感知的图神经网络架构
- 解决传统GNN无法处理不同复杂度查询的问题
- 提供可解释的复杂度驱动推理机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ComplexityType(Enum):
    """查询复杂度类型枚举"""
    ZERO_HOP = "zero_hop"    # 无需检索的简单查询
    ONE_HOP = "one_hop"      # 单跳检索查询
    MULTI_HOP = "multi_hop"  # 多跳推理查询


@dataclass
class ComplexityAwareConfig:
    """复杂度感知配置"""
    
    # 复杂度嵌入参数
    complexity_embedding_dim: int = 64  # 复杂度嵌入维度
    num_complexity_types: int = 3       # 复杂度类型数量
    
    # 调制参数
    modulation_type: str = "gating"     # gating, attention, scaling
    gate_activation: str = "sigmoid"    # sigmoid, tanh
    
    # 自适应权重参数
    enable_adaptive_weights: bool = True
    adaptive_weight_dim: int = 32
    
    # 复杂度预测参数  
    enable_complexity_prediction: bool = True
    prediction_hidden_dim: int = 128
    
    # 门控融合参数
    enable_gated_fusion: bool = True
    fusion_dropout: float = 0.1
    
    # 正则化参数
    complexity_regularization: float = 0.01
    orthogonality_regularization: float = 0.001


class ComplexityEmbedding(nn.Module):
    """复杂度嵌入层"""
    
    def __init__(self, config: ComplexityAwareConfig):
        super().__init__()
        self.config = config
        
        # 复杂度类型嵌入
        self.complexity_embeddings = nn.Embedding(
            num_embeddings=config.num_complexity_types,
            embedding_dim=config.complexity_embedding_dim
        )
        
        # 复杂度特征投影
        self.complexity_projection = nn.Sequential(
            nn.Linear(config.complexity_embedding_dim, config.complexity_embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.complexity_embedding_dim, config.complexity_embedding_dim)
        )
        
        # 初始化嵌入
        nn.init.xavier_uniform_(self.complexity_embeddings.weight)
        
        # 复杂度到ID的映射
        self.complexity_to_id = {
            ComplexityType.ZERO_HOP.value: 0,
            ComplexityType.ONE_HOP.value: 1,
            ComplexityType.MULTI_HOP.value: 2
        }
    
    def forward(self, 
                complexity_types: Union[List[str], torch.Tensor],
                batch_size: Optional[int] = None) -> torch.Tensor:
        """
        生成复杂度嵌入
        
        Args:
            complexity_types: 复杂度类型列表或张量
            batch_size: 批次大小
            
        Returns:
            复杂度嵌入 [batch_size, complexity_embedding_dim]
        """
        if isinstance(complexity_types, list):
            # 转换字符串列表为ID
            complexity_ids = [self.complexity_to_id.get(ct, 1) for ct in complexity_types]
            complexity_tensor = torch.tensor(complexity_ids, dtype=torch.long)
            if batch_size is not None and len(complexity_ids) == 1:
                complexity_tensor = complexity_tensor.repeat(batch_size)
        else:
            complexity_tensor = complexity_types
            
        # 获取嵌入
        complexity_emb = self.complexity_embeddings(complexity_tensor)
        
        # 投影和归一化
        complexity_emb = self.complexity_projection(complexity_emb)
        complexity_emb = F.normalize(complexity_emb, p=2, dim=-1)
        
        return complexity_emb


class ComplexityModulator(nn.Module):
    """复杂度调制器"""
    
    def __init__(self, feature_dim: int, config: ComplexityAwareConfig):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        
        if config.modulation_type == "gating":
            self.modulator = self._build_gating_modulator()
        elif config.modulation_type == "attention":
            self.modulator = self._build_attention_modulator()
        elif config.modulation_type == "scaling":
            self.modulator = self._build_scaling_modulator()
        else:
            raise ValueError(f"Unknown modulation type: {config.modulation_type}")
    
    def _build_gating_modulator(self):
        """构建门控调制器"""
        return nn.Sequential(
            nn.Linear(
                self.config.complexity_embedding_dim + self.feature_dim,
                self.feature_dim
            ),
            nn.Sigmoid() if self.config.gate_activation == "sigmoid" else nn.Tanh()
        )
    
    def _build_attention_modulator(self):
        """构建注意力调制器"""
        return nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            dropout=self.config.fusion_dropout,
            batch_first=True
        )
    
    def _build_scaling_modulator(self):
        """构建缩放调制器"""
        return nn.Sequential(
            nn.Linear(self.config.complexity_embedding_dim, self.feature_dim),
            nn.Softplus()  # 确保输出为正
        )
    
    def forward(self, 
                features: torch.Tensor,
                complexity_emb: torch.Tensor) -> torch.Tensor:
        """
        复杂度调制特征
        
        Args:
            features: 输入特征 [batch_size, feature_dim] or [batch_size, seq_len, feature_dim]
            complexity_emb: 复杂度嵌入 [batch_size, complexity_embedding_dim]
            
        Returns:
            调制后的特征
        """
        if self.config.modulation_type == "gating":
            return self._gating_modulation(features, complexity_emb)
        elif self.config.modulation_type == "attention":
            return self._attention_modulation(features, complexity_emb)
        elif self.config.modulation_type == "scaling":
            return self._scaling_modulation(features, complexity_emb)
    
    def _gating_modulation(self, features: torch.Tensor, complexity_emb: torch.Tensor) -> torch.Tensor:
        """门控调制实现"""
        batch_size = features.size(0)
        
        # 处理不同维度的features
        if features.dim() == 3:  # [batch, seq_len, feature_dim]
            seq_len = features.size(1)
            complexity_emb_expanded = complexity_emb.unsqueeze(1).expand(-1, seq_len, -1)
            features_flat = features.view(-1, self.feature_dim)
            complexity_flat = complexity_emb_expanded.view(-1, self.config.complexity_embedding_dim)
            
            # 拼接并生成门控
            combined = torch.cat([features_flat, complexity_flat], dim=-1)
            gates = self.modulator(combined)
            gates = gates.view(batch_size, seq_len, -1)
            
            return features * gates
        else:  # [batch, feature_dim]
            # 拼接特征和复杂度嵌入
            combined = torch.cat([features, complexity_emb], dim=-1)
            gates = self.modulator(combined)
            
            return features * gates
    
    def _attention_modulation(self, features: torch.Tensor, complexity_emb: torch.Tensor) -> torch.Tensor:
        """注意力调制实现"""
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [batch, 1, feature_dim]
        
        # 使用复杂度嵌入作为query
        complexity_query = complexity_emb.unsqueeze(1)  # [batch, 1, complexity_dim]
        
        # 投影到相同维度
        if complexity_query.size(-1) != features.size(-1):
            complexity_query = F.linear(complexity_query, 
                                      torch.randn(features.size(-1), complexity_query.size(-1)))
        
        # 注意力计算
        attended_features, _ = self.modulator(complexity_query, features, features)
        
        return attended_features.squeeze(1)
    
    def _scaling_modulation(self, features: torch.Tensor, complexity_emb: torch.Tensor) -> torch.Tensor:
        """缩放调制实现"""
        # 生成缩放因子
        scaling_factors = self.modulator(complexity_emb)
        
        if features.dim() == 3:
            scaling_factors = scaling_factors.unsqueeze(1)
        
        return features * scaling_factors


class AdaptiveWeightGenerator(nn.Module):
    """自适应权重生成器"""
    
    def __init__(self, input_dim: int, config: ComplexityAwareConfig):
        super().__init__()
        self.config = config
        
        # 权重生成网络
        self.weight_generator = nn.Sequential(
            nn.Linear(input_dim + config.complexity_embedding_dim, config.adaptive_weight_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.adaptive_weight_dim, 1),
            nn.Sigmoid()
        )
        
        # 复杂度特异性权重
        self.complexity_specific_weights = nn.Parameter(
            torch.randn(config.num_complexity_types, input_dim)
        )
        nn.init.xavier_uniform_(self.complexity_specific_weights)
    
    def forward(self, 
                neighbor_features: torch.Tensor,
                complexity_emb: torch.Tensor,
                complexity_ids: torch.Tensor) -> torch.Tensor:
        """
        生成自适应邻居权重
        
        Args:
            neighbor_features: 邻居特征 [batch, num_neighbors, feature_dim]
            complexity_emb: 复杂度嵌入 [batch, complexity_dim]
            complexity_ids: 复杂度ID [batch]
            
        Returns:
            邻居权重 [batch, num_neighbors, 1]
        """
        batch_size, num_neighbors, feature_dim = neighbor_features.shape
        
        # 扩展复杂度嵌入到所有邻居
        complexity_expanded = complexity_emb.unsqueeze(1).expand(-1, num_neighbors, -1)
        
        # 拼接邻居特征和复杂度嵌入
        combined_features = torch.cat([neighbor_features, complexity_expanded], dim=-1)
        
        # 生成基础权重
        base_weights = self.weight_generator(combined_features)  # [batch, num_neighbors, 1]
        
        # 获取复杂度特异性权重
        complexity_weights = self.complexity_specific_weights[complexity_ids]  # [batch, feature_dim]
        
        # 计算特异性得分
        specificity_scores = torch.bmm(
            neighbor_features, complexity_weights.unsqueeze(-1)
        )  # [batch, num_neighbors, 1]
        specificity_scores = torch.sigmoid(specificity_scores)
        
        # 融合基础权重和特异性权重
        final_weights = base_weights * specificity_scores
        
        # 归一化
        final_weights = F.softmax(final_weights, dim=1)
        
        return final_weights


class ComplexityAwareLayer(nn.Module):
    """
    复杂度感知图神经网络层
    
    这是项目的核心创新模块，将查询复杂度信息融入图神经网络，
    实现复杂度感知的图表示学习。
    """
    
    def __init__(self, input_dim: int, output_dim: int, config: ComplexityAwareConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 复杂度嵌入
        self.complexity_embedding = ComplexityEmbedding(config)
        
        # 特征变换
        self.self_transform = nn.Linear(input_dim, output_dim)
        self.neighbor_transform = nn.Linear(input_dim, output_dim)
        
        # 复杂度调制器
        self.self_modulator = ComplexityModulator(output_dim, config)
        self.neighbor_modulator = ComplexityModulator(output_dim, config)
        
        # 自适应权重生成器
        if config.enable_adaptive_weights:
            self.weight_generator = AdaptiveWeightGenerator(input_dim, config)
        
        # 门控融合
        if config.enable_gated_fusion:
            self.fusion_gate = nn.Sequential(
                nn.Linear(output_dim * 2 + config.complexity_embedding_dim, output_dim),
                nn.Sigmoid()
            )
        
        # 复杂度预测头（用于辅助训练）
        if config.enable_complexity_prediction:
            self.complexity_predictor = nn.Sequential(
                nn.Linear(output_dim, config.prediction_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.fusion_dropout),
                nn.Linear(config.prediction_hidden_dim, config.num_complexity_types)
            )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout)
        )
    
    def forward(self, 
                self_features: torch.Tensor,
                neighbor_features: torch.Tensor,
                complexity_types: Union[List[str], torch.Tensor],
                neighbor_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        复杂度感知前向传播
        
        Args:
            self_features: 自身节点特征 [batch_size, input_dim]
            neighbor_features: 邻居特征 [batch_size, max_neighbors, input_dim]
            complexity_types: 查询复杂度类型
            neighbor_mask: 邻居掩码 [batch_size, max_neighbors]
            
        Returns:
            包含输出特征和辅助信息的字典
        """
        batch_size = self_features.size(0)
        
        # 1. 生成复杂度嵌入
        complexity_emb = self.complexity_embedding(complexity_types, batch_size)
        
        # 2. 特征变换
        self_transformed = self.self_transform(self_features)  # [batch, output_dim]
        neighbor_transformed = self.neighbor_transform(neighbor_features)  # [batch, neighbors, output_dim]
        
        # 3. 复杂度调制
        modulated_self = self.self_modulator(self_transformed, complexity_emb)
        modulated_neighbors = self.neighbor_modulator(neighbor_transformed, complexity_emb)
        
        # 4. 自适应邻居聚合
        if self.config.enable_adaptive_weights and hasattr(self, 'weight_generator'):
            # 生成复杂度ID
            if isinstance(complexity_types, list):
                complexity_ids = torch.tensor([
                    self.complexity_embedding.complexity_to_id.get(ct, 1) for ct in complexity_types
                ], dtype=torch.long, device=self_features.device)
            else:
                complexity_ids = complexity_types
            
            # 生成自适应权重
            adaptive_weights = self.weight_generator(
                neighbor_features, complexity_emb, complexity_ids
            )  # [batch, neighbors, 1]
            
            # 加权聚合
            if neighbor_mask is not None:
                adaptive_weights = adaptive_weights * neighbor_mask.unsqueeze(-1).float()
            
            weighted_neighbors = modulated_neighbors * adaptive_weights
            aggregated_neighbors = weighted_neighbors.sum(dim=1)  # [batch, output_dim]
        else:
            # 标准均值聚合
            if neighbor_mask is not None:
                masked_neighbors = modulated_neighbors * neighbor_mask.unsqueeze(-1).float()
                neighbor_count = neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
                aggregated_neighbors = masked_neighbors.sum(dim=1) / neighbor_count
            else:
                aggregated_neighbors = modulated_neighbors.mean(dim=1)
        
        # 5. 门控融合
        if self.config.enable_gated_fusion and hasattr(self, 'fusion_gate'):
            # 拼接所有信息
            fusion_input = torch.cat([modulated_self, aggregated_neighbors, complexity_emb], dim=-1)
            gate = self.fusion_gate(fusion_input)
            
            # 门控融合
            output = gate * modulated_self + (1 - gate) * aggregated_neighbors
        else:
            # 简单拼接
            output = modulated_self + aggregated_neighbors
        
        # 6. 输出投影
        output = self.output_projection(output)
        
        # 7. 复杂度预测（辅助任务）
        result = {"output": output, "complexity_embedding": complexity_emb}
        
        if self.config.enable_complexity_prediction and hasattr(self, 'complexity_predictor'):
            complexity_pred = self.complexity_predictor(output)
            result["complexity_prediction"] = complexity_pred
        
        if self.config.enable_adaptive_weights and 'adaptive_weights' in locals():
            result["adaptive_weights"] = adaptive_weights
        
        return result
    
    def compute_complexity_regularization(self) -> torch.Tensor:
        """计算复杂度正则化损失"""
        reg_loss = 0.0
        
        # 复杂度嵌入正交性正则化
        if hasattr(self.complexity_embedding, 'complexity_embeddings'):
            embeddings = self.complexity_embedding.complexity_embeddings.weight
            gram = torch.mm(embeddings, embeddings.t())
            identity = torch.eye(embeddings.size(0), device=embeddings.device)
            orthogonality_loss = F.mse_loss(gram, identity)
            reg_loss += self.config.orthogonality_regularization * orthogonality_loss
        
        return reg_loss


# 工厂函数
def create_complexity_aware_layer(input_dim: int, 
                                  output_dim: int, 
                                  config: Optional[ComplexityAwareConfig] = None) -> ComplexityAwareLayer:
    """创建复杂度感知层"""
    if config is None:
        config = ComplexityAwareConfig()
    return ComplexityAwareLayer(input_dim, output_dim, config)


def create_complexity_aware_config(**kwargs) -> ComplexityAwareConfig:
    """创建复杂度感知配置"""
    return ComplexityAwareConfig(**kwargs) 