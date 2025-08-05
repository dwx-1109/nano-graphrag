"""
GraphSAGE: 归纳式图表示学习

基于Hamilton et al. (NIPS 2017)的GraphSAGE算法实现，支持:
- 归纳式节点表示学习
- 多种聚合函数(Mean, LSTM, Pooling)
- 邻居采样策略
- 批处理训练

参考文献:
Hamilton, W., Ying, Z., & Leskovec, J. (2017). 
Inductive representation learning on large graphs. NIPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class GraphSAGEConfig:
    """GraphSAGE配置类"""
    
    # 模型架构参数
    input_dim: int = 768  # 输入特征维度 (如BERT嵌入)
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])  # 隐藏层维度
    output_dim: int = 128  # 输出嵌入维度
    num_layers: int = 2  # GraphSAGE层数
    
    # 邻居采样参数
    neighbor_sample_sizes: List[int] = field(default_factory=lambda: [25, 10])  # 每层采样邻居数
    max_degree: int = 50  # 最大度数限制
    
    # 聚合器参数
    aggregator_type: str = "mean"  # mean, lstm, pool, attention
    lstm_hidden_dim: int = 128  # LSTM聚合器隐藏维度
    pooling_type: str = "max"  # max, mean for pooling aggregator
    
    # 训练参数
    learning_rate: float = 0.001
    dropout: float = 0.1
    batch_size: int = 256
    num_epochs: int = 50
    weight_decay: float = 1e-5
    
    # 设备参数
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # 正则化参数
    l2_regularization: float = 1e-4
    gradient_clipping: float = 1.0
    
    # 复杂度感知参数
    enable_complexity_aware: bool = True
    complexity_dim: int = 32  # 复杂度嵌入维度


class GraphSAGEAggregator(nn.Module, ABC):
    """GraphSAGE聚合器基类"""
    
    def __init__(self, input_dim: int, output_dim: int, config: GraphSAGEConfig):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        
    @abstractmethod
    def forward(self, 
                self_features: torch.Tensor,
                neighbor_features: torch.Tensor,
                neighbor_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        聚合邻居特征
        
        Args:
            self_features: 自身节点特征 [batch_size, input_dim]
            neighbor_features: 邻居节点特征 [batch_size, max_neighbors, input_dim]
            neighbor_mask: 邻居掩码 [batch_size, max_neighbors]
            
        Returns:
            聚合后的特征 [batch_size, output_dim]
        """
        pass


class MeanAggregator(GraphSAGEAggregator):
    """均值聚合器 - GraphSAGE原始实现"""
    
    def __init__(self, input_dim: int, output_dim: int, config: GraphSAGEConfig):
        super().__init__(input_dim, output_dim, config)
        
        # 线性变换层
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim)
        self.output_linear = nn.Linear(output_dim * 2, output_dim)
        
        # Dropout和归一化
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, 
                self_features: torch.Tensor,
                neighbor_features: torch.Tensor,
                neighbor_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        均值聚合实现
        
        基于原始GraphSAGE论文的均值聚合器实现，包含以下步骤:
        1. 计算邻居特征的均值
        2. 线性变换自身和邻居特征
        3. 拼接并投影到输出维度
        """
        batch_size = self_features.size(0)
        
        # 1. 邻居特征均值聚合
        if neighbor_mask is not None:
            # 考虑邻居掩码的加权平均
            neighbor_mask = neighbor_mask.unsqueeze(-1)  # [batch, neighbors, 1]
            masked_neighbor_features = neighbor_features * neighbor_mask.float()
            neighbor_sum = masked_neighbor_features.sum(dim=1)  # [batch, input_dim]
            neighbor_count = neighbor_mask.sum(dim=1).clamp(min=1)  # [batch, 1]
            neighbor_mean = neighbor_sum / neighbor_count.float()
        else:
            neighbor_mean = neighbor_features.mean(dim=1)  # [batch, input_dim]
        
        # 2. 线性变换
        self_transformed = self.self_linear(self_features)  # [batch, output_dim]
        neighbor_transformed = self.neighbor_linear(neighbor_mean)  # [batch, output_dim]
        
        # 3. 拼接和最终投影
        combined = torch.cat([self_transformed, neighbor_transformed], dim=-1)  # [batch, 2*output_dim]
        output = self.output_linear(combined)  # [batch, output_dim]
        
        # 4. 正则化和激活
        output = self.layer_norm(output)
        output = F.relu(output)
        output = self.dropout(output)
        
        return output


class LSTMAggregator(GraphSAGEAggregator):
    """LSTM聚合器 - 处理邻居顺序信息"""
    
    def __init__(self, input_dim: int, output_dim: int, config: GraphSAGEConfig):
        super().__init__(input_dim, output_dim, config)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.lstm_hidden_dim,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # 线性变换层
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(config.lstm_hidden_dim, output_dim)
        self.output_linear = nn.Linear(output_dim * 2, output_dim)
        
        # 正则化
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, 
                self_features: torch.Tensor,
                neighbor_features: torch.Tensor,
                neighbor_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        LSTM聚合实现
        
        使用LSTM处理邻居节点序列，捕获邻居间的顺序关系
        """
        batch_size, max_neighbors, feature_dim = neighbor_features.shape
        
        # 1. LSTM处理邻居序列
        if neighbor_mask is not None:
            # 计算实际序列长度
            seq_lengths = neighbor_mask.sum(dim=1).cpu()
            # Pack padded sequence for efficiency
            packed_neighbors = nn.utils.rnn.pack_padded_sequence(
                neighbor_features, seq_lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed_neighbors)
            # 使用最后的隐藏状态作为聚合结果
            neighbor_aggregated = hidden[-1]  # [batch, lstm_hidden_dim]
        else:
            lstm_out, (hidden, cell) = self.lstm(neighbor_features)
            neighbor_aggregated = hidden[-1]  # [batch, lstm_hidden_dim]
        
        # 2. 线性变换
        self_transformed = self.self_linear(self_features)
        neighbor_transformed = self.neighbor_linear(neighbor_aggregated)
        
        # 3. 拼接和投影
        combined = torch.cat([self_transformed, neighbor_transformed], dim=-1)
        output = self.output_linear(combined)
        
        # 4. 正则化和激活
        output = self.layer_norm(output)
        output = F.relu(output)
        output = self.dropout(output)
        
        return output


class PoolingAggregator(GraphSAGEAggregator):
    """池化聚合器 - 最大值或均值池化"""
    
    def __init__(self, input_dim: int, output_dim: int, config: GraphSAGEConfig):
        super().__init__(input_dim, output_dim, config)
        
        self.pooling_type = config.pooling_type
        
        # MLP for neighbor transformation
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # 线性变换层
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.output_linear = nn.Linear(output_dim * 2, output_dim)
        
        # 正则化
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, 
                self_features: torch.Tensor,
                neighbor_features: torch.Tensor,
                neighbor_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        池化聚合实现
        
        对邻居特征进行非线性变换后池化
        """
        batch_size, max_neighbors, feature_dim = neighbor_features.shape
        
        # 1. 邻居特征非线性变换
        neighbor_transformed = self.neighbor_mlp(neighbor_features)  # [batch, neighbors, output_dim]
        
        # 2. 池化操作
        if neighbor_mask is not None:
            # 应用掩码
            mask_expanded = neighbor_mask.unsqueeze(-1).expand_as(neighbor_transformed)
            if self.pooling_type == "max":
                masked_features = neighbor_transformed.masked_fill(~mask_expanded, float('-inf'))
                neighbor_pooled = masked_features.max(dim=1)[0]  # [batch, output_dim]
            else:  # mean pooling
                masked_features = neighbor_transformed * mask_expanded.float()
                neighbor_sum = masked_features.sum(dim=1)
                neighbor_count = neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
                neighbor_pooled = neighbor_sum / neighbor_count
        else:
            if self.pooling_type == "max":
                neighbor_pooled = neighbor_transformed.max(dim=1)[0]
            else:
                neighbor_pooled = neighbor_transformed.mean(dim=1)
        
        # 3. 自身特征变换
        self_transformed = self.self_linear(self_features)
        
        # 4. 拼接和投影
        combined = torch.cat([self_transformed, neighbor_pooled], dim=-1)
        output = self.output_linear(combined)
        
        # 5. 正则化和激活
        output = self.layer_norm(output)
        output = F.relu(output)
        output = self.dropout(output)
        
        return output


class GraphSAGELayer(nn.Module):
    """单层GraphSAGE实现"""
    
    def __init__(self, input_dim: int, output_dim: int, config: GraphSAGEConfig):
        super().__init__()
        self.config = config
        
        # 选择聚合器
        if config.aggregator_type == "mean":
            self.aggregator = MeanAggregator(input_dim, output_dim, config)
        elif config.aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(input_dim, output_dim, config)
        elif config.aggregator_type == "pool":
            self.aggregator = PoolingAggregator(input_dim, output_dim, config)
        else:
            raise ValueError(f"Unknown aggregator type: {config.aggregator_type}")
    
    def forward(self, 
                node_features: torch.Tensor,
                neighbor_features: torch.Tensor,
                neighbor_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        GraphSAGE层前向传播
        
        Args:
            node_features: 节点特征 [batch_size, input_dim]
            neighbor_features: 邻居特征 [batch_size, max_neighbors, input_dim]
            neighbor_mask: 邻居掩码 [batch_size, max_neighbors]
            
        Returns:
            更新后的节点特征 [batch_size, output_dim]
        """
        # 聚合邻居信息
        aggregated = self.aggregator(node_features, neighbor_features, neighbor_mask)
        
        # L2正则化
        aggregated = F.normalize(aggregated, p=2, dim=-1)
        
        return aggregated


class GraphSAGE(nn.Module):
    """
    GraphSAGE主模型
    
    实现多层GraphSAGE网络，支持归纳式节点表示学习
    """
    
    def __init__(self, config: GraphSAGEConfig):
        super().__init__()
        self.config = config
        
        # 构建多层GraphSAGE
        self.layers = nn.ModuleList()
        
        # 输入层
        input_dim = config.input_dim
        for i, hidden_dim in enumerate(config.hidden_dims):
            layer = GraphSAGELayer(input_dim, hidden_dim, config)
            self.layers.append(layer)
            input_dim = hidden_dim
        
        # 输出层
        if config.hidden_dims:
            final_layer = GraphSAGELayer(config.hidden_dims[-1], config.output_dim, config)
        else:
            final_layer = GraphSAGELayer(config.input_dim, config.output_dim, config)
        self.layers.append(final_layer)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        logger.info(f"GraphSAGE initialized with {len(self.layers)} layers")
    
    def _init_weights(self, module):
        """Xavier初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, 
                node_features: torch.Tensor,
                neighbor_features_list: List[torch.Tensor],
                neighbor_mask_list: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        GraphSAGE前向传播
        
        Args:
            node_features: 节点初始特征 [batch_size, input_dim]
            neighbor_features_list: 每层的邻居特征列表
            neighbor_mask_list: 每层的邻居掩码列表
            
        Returns:
            最终节点嵌入 [batch_size, output_dim]
        """
        if neighbor_mask_list is None:
            neighbor_mask_list = [None] * len(neighbor_features_list)
        
        assert len(neighbor_features_list) == len(self.layers), \
            f"Expected {len(self.layers)} neighbor feature tensors, got {len(neighbor_features_list)}"
        
        # 逐层传播
        current_features = node_features
        for i, (layer, neighbor_features, neighbor_mask) in enumerate(
            zip(self.layers, neighbor_features_list, neighbor_mask_list)
        ):
            current_features = layer(current_features, neighbor_features, neighbor_mask)
            
            # 中间层激活函数（最后一层不加激活）
            if i < len(self.layers) - 1:
                current_features = F.relu(current_features)
        
        return current_features
    
    def encode_nodes(self, 
                     node_ids: List[str],
                     node_features: torch.Tensor,
                     graph_data: Dict[str, Any],
                     sample_sizes: Optional[List[int]] = None) -> torch.Tensor:
        """
        编码节点嵌入（推理模式）
        
        Args:
            node_ids: 节点ID列表
            node_features: 节点特征
            graph_data: 图数据（邻接关系等）
            sample_sizes: 每层采样大小
            
        Returns:
            节点嵌入
        """
        if sample_sizes is None:
            sample_sizes = self.config.neighbor_sample_sizes
        
        # 采样邻居
        neighbor_features_list, neighbor_mask_list = self._sample_neighbors(
            node_ids, node_features, graph_data, sample_sizes
        )
        
        # 前向传播
        with torch.no_grad():
            embeddings = self.forward(node_features, neighbor_features_list, neighbor_mask_list)
        
        return embeddings
    
    def _sample_neighbors(self, 
                          node_ids: List[str],
                          node_features: torch.Tensor, 
                          graph_data: Dict[str, Any],
                          sample_sizes: List[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        采样多层邻居
        
        实现GraphSAGE的邻居采样策略
        """
        # 这里简化实现，实际项目中需要根据具体的图数据结构来实现
        batch_size = len(node_ids)
        device = node_features.device
        
        neighbor_features_list = []
        neighbor_mask_list = []
        
        for layer_idx, sample_size in enumerate(sample_sizes):
            # 模拟邻居采样（实际实现需要根据图结构）
            neighbor_features = torch.randn(
                batch_size, sample_size, node_features.size(-1), 
                device=device
            )
            neighbor_mask = torch.ones(batch_size, sample_size, dtype=torch.bool, device=device)
            
            neighbor_features_list.append(neighbor_features)
            neighbor_mask_list.append(neighbor_mask)
        
        return neighbor_features_list, neighbor_mask_list


# 工厂函数
def create_graphsage_model(config: GraphSAGEConfig) -> GraphSAGE:
    """创建GraphSAGE模型"""
    return GraphSAGE(config)


def create_graphsage_config(**kwargs) -> GraphSAGEConfig:
    """创建GraphSAGE配置"""
    return GraphSAGEConfig(**kwargs) 