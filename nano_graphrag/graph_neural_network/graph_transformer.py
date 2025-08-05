"""
基于最新Graph Transformer的图神经网络实现
参考: Graph Transformer Networks (GTN) 和 Graphormer
适配nano_graphrag项目的实际需求
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

from .complexity_aware import ComplexityAwareConfig, ComplexityEmbedding, ComplexityModulator

@dataclass
class GraphTransformerConfig:
    """Graph Transformer配置类"""
    
    # 模型架构参数
    input_dim: int = 768  # 输入特征维度 (如BERT嵌入)
    hidden_dim: int = 256  # 隐藏层维度
    output_dim: int = 128  # 输出嵌入维度
    num_layers: int = 3  # Transformer层数
    num_heads: int = 8  # 注意力头数
    
    # 位置编码参数
    max_path_distance: int = 10  # 最大路径距离
    use_edge_features: bool = True  # 是否使用边特征
    edge_dim: int = 32  # 边特征维度
    
    # 注意力参数
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # 图结构编码参数
    use_centrality_encoding: bool = True  # 使用中心性编码
    use_spatial_encoding: bool = True     # 使用空间编码
    spatial_encoding_dim: int = 32
    
    # 训练参数
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clipping: float = 1.0
    
    # 复杂度感知参数
    enable_complexity_aware: bool = True
    complexity_config: ComplexityAwareConfig = field(default_factory=ComplexityAwareConfig)
    
    # 设备参数
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class GraphStructuralEncoding(nn.Module):
    """图结构编码层，结合空间位置和中心性特征"""
    
    def __init__(self, config: GraphTransformerConfig):
        super().__init__()
        self.config = config
        
        # 空间位置编码（基于最短路径距离）
        if config.use_spatial_encoding:
            self.spatial_embedding = nn.Embedding(
                config.max_path_distance + 1, 
                config.spatial_encoding_dim
            )
        
        # 中心性编码
        if config.use_centrality_encoding:
            self.degree_embedding = nn.Embedding(100, config.spatial_encoding_dim)  # 支持度数0-99
            self.centrality_projection = nn.Linear(config.spatial_encoding_dim * 2, config.hidden_dim)
        
        # 边特征处理
        if config.use_edge_features:
            self.edge_projection = nn.Linear(config.edge_dim, config.hidden_dim)
    
    def forward(self, 
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None,
                node_degrees: Optional[torch.Tensor] = None,
                spatial_distances: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_features: [N, input_dim] 节点特征
            edge_index: [2, E] 边索引
            edge_features: [E, edge_dim] 边特征
            node_degrees: [N] 节点度数
            spatial_distances: [N, N] 节点间最短路径距离
        """
        batch_size, seq_len = node_features.shape[:2]
        encodings = {}
        
        # 空间位置编码
        if self.config.use_spatial_encoding and spatial_distances is not None:
            # 限制距离在最大范围内
            clipped_distances = torch.clamp(spatial_distances, 0, self.config.max_path_distance)
            spatial_encoding = self.spatial_embedding(clipped_distances.long())
            encodings['spatial'] = spatial_encoding  # [N, N, spatial_encoding_dim]
        
        # 中心性编码
        if self.config.use_centrality_encoding and node_degrees is not None:
            degree_encoding = self.degree_embedding(torch.clamp(node_degrees, 0, 99).long())
            encodings['degree'] = degree_encoding  # [N, spatial_encoding_dim]
        
        # 边特征编码
        if self.config.use_edge_features and edge_features is not None:
            edge_encoding = self.edge_projection(edge_features)
            encodings['edge'] = edge_encoding  # [E, hidden_dim]
        
        return encodings

class GraphMultiHeadAttention(nn.Module):
    """图结构感知的多头注意力机制"""
    
    def __init__(self, config: GraphTransformerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        # 线性投影层
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # 结构偏置
        if config.use_spatial_encoding:
            self.spatial_bias = nn.Linear(config.spatial_encoding_dim, config.num_heads)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                spatial_encoding: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len, seq_len] 注意力掩码
            spatial_encoding: [seq_len, seq_len, spatial_encoding_dim] 空间编码
            return_attention: 是否返回注意力权重
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 线性投影
        q = self.q_proj(hidden_states)  # [batch_size, seq_len, hidden_dim]
        k = self.k_proj(hidden_states)  # [batch_size, seq_len, hidden_dim]
        v = self.v_proj(hidden_states)  # [batch_size, seq_len, hidden_dim]
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 添加结构偏置
        if self.config.use_spatial_encoding and spatial_encoding is not None:
            spatial_bias = self.spatial_bias(spatial_encoding)  # [seq_len, seq_len, num_heads]
            spatial_bias = spatial_bias.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, seq_len, seq_len]
            attention_scores = attention_scores + spatial_bias
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # 计算注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 应用注意力权重
        context = torch.matmul(attention_probs, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重塑输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = self.out_proj(context)
        
        if return_attention:
            return output, attention_probs.mean(dim=1)  # 平均所有头的注意力
        return output, None

class GraphTransformerLayer(nn.Module):
    """Graph Transformer层"""
    
    def __init__(self, config: GraphTransformerConfig):
        super().__init__()
        self.config = config
        
        # 多头注意力
        self.attention = GraphMultiHeadAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.hidden_dropout)
        )
        self.ff_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # 复杂度调制器（如果启用）
        if config.enable_complexity_aware:
            self.complexity_modulator = ComplexityModulator(config.hidden_dim, config.complexity_config)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                spatial_encoding: Optional[torch.Tensor] = None,
                complexity_emb: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # 多头注意力
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        attention_output, attention_weights = self.attention(
            hidden_states, 
            attention_mask=attention_mask,
            spatial_encoding=spatial_encoding,
            return_attention=return_attention
        )
        hidden_states = residual + attention_output
        
        # 前馈网络
        residual = hidden_states
        hidden_states = self.ff_norm(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + ff_output
        
        # 复杂度调制
        if self.config.enable_complexity_aware and complexity_emb is not None:
            hidden_states = self.complexity_modulator(hidden_states, complexity_emb)
        
        return hidden_states, attention_weights

class GraphTransformer(nn.Module):
    """
    Graph Transformer模型
    基于Graphormer和最新Graph Transformer研究
    """
    
    def __init__(self, config: GraphTransformerConfig):
        super().__init__()
        self.config = config
        
        # 输入投影
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # 结构编码
        self.structural_encoding = GraphStructuralEncoding(config)
        
        # Transformer层
        self.layers = nn.ModuleList([
            GraphTransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # 输出投影
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)
        self.output_norm = nn.LayerNorm(config.output_dim, eps=config.layer_norm_eps)
        
        # 复杂度嵌入（如果启用）
        if config.enable_complexity_aware:
            self.complexity_embedding = ComplexityEmbedding(config.complexity_config)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, 
                node_features: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                edge_features: Optional[torch.Tensor] = None,
                node_degrees: Optional[torch.Tensor] = None,
                spatial_distances: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                complexity_types: Optional[Union[List[str], torch.Tensor]] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_features: [batch_size, seq_len, input_dim] 节点特征
            edge_index: [2, E] 边索引
            edge_features: [E, edge_dim] 边特征
            node_degrees: [batch_size, seq_len] 节点度数
            spatial_distances: [batch_size, seq_len, seq_len] 空间距离
            attention_mask: [batch_size, seq_len] 注意力掩码
            complexity_types: 查询复杂度类型
            return_attention: 是否返回注意力权重
        """
        batch_size, seq_len = node_features.shape[:2]
        
        # 输入投影
        hidden_states = self.input_projection(node_features)
        
        # 结构编码
        structure_encodings = self.structural_encoding(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_degrees=node_degrees,
            spatial_distances=spatial_distances
        )
        
        # 复杂度嵌入
        complexity_emb = None
        if self.config.enable_complexity_aware and complexity_types is not None:
            complexity_emb = self.complexity_embedding(complexity_types, batch_size)
        
        # 通过Transformer层
        all_attention_weights = []
        for layer in self.layers:
            hidden_states, attention_weights = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                spatial_encoding=structure_encodings.get('spatial'),
                complexity_emb=complexity_emb,
                return_attention=return_attention
            )
            if return_attention and attention_weights is not None:
                all_attention_weights.append(attention_weights)
        
        # 输出投影
        output = self.output_projection(hidden_states)
        output = self.output_norm(output)
        
        results = {
            'node_embeddings': output,
            'hidden_states': hidden_states,
            'structure_encodings': structure_encodings
        }
        
        if return_attention:
            results['attention_weights'] = all_attention_weights
        
        return results
    
    def encode_graph(self, 
                     graph_data: Dict[str, Any],
                     complexity_type: str = "one_hop") -> torch.Tensor:
        """
        编码整个图
        
        Args:
            graph_data: 图数据字典，包含节点特征、边信息等
            complexity_type: 查询复杂度类型
        
        Returns:
            图嵌入表示
        """
        # 提取图数据
        node_features = graph_data.get('node_features')  # [num_nodes, input_dim]
        edge_index = graph_data.get('edge_index')        # [2, num_edges]
        edge_features = graph_data.get('edge_features')  # [num_edges, edge_dim]
        node_degrees = graph_data.get('node_degrees')    # [num_nodes]
        spatial_distances = graph_data.get('spatial_distances')  # [num_nodes, num_nodes]
        
        # 添加batch维度
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        if spatial_distances is not None and spatial_distances.dim() == 2:
            spatial_distances = spatial_distances.unsqueeze(0)
        if node_degrees is not None and node_degrees.dim() == 1:
            node_degrees = node_degrees.unsqueeze(0)
        
        # 前向传播
        results = self.forward(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_degrees=node_degrees,
            spatial_distances=spatial_distances,
            complexity_types=[complexity_type],
            return_attention=False
        )
        
        # 返回图级别的嵌入（平均池化）
        node_embeddings = results['node_embeddings']  # [1, num_nodes, output_dim]
        graph_embedding = torch.mean(node_embeddings, dim=1)  # [1, output_dim]
        
        return graph_embedding.squeeze(0)  # [output_dim]

def create_graph_transformer(config: Optional[GraphTransformerConfig] = None) -> GraphTransformer:
    """创建Graph Transformer模型"""
    if config is None:
        config = GraphTransformerConfig()
    return GraphTransformer(config)

def create_graph_transformer_config(**kwargs) -> GraphTransformerConfig:
    """创建Graph Transformer配置"""
    return GraphTransformerConfig(**kwargs) 