"""
图嵌入生成模块
提供图结构的嵌入表示生成功能
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class GraphEmbeddingConfig:
    """图嵌入配置"""
    embedding_dim: int = 768
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    use_attention: bool = True
    aggregation_method: str = "mean"  # mean, sum, max, attention


class GraphEmbeddingGenerator(nn.Module):
    """图嵌入生成器"""
    
    def __init__(self, config: GraphEmbeddingConfig):
        super().__init__()
        self.config = config
        
        # 图神经网络层
        self.gnn_layers = nn.ModuleList([
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        ])
        
        # 注意力机制
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.embedding_dim,
                num_heads=8,
                dropout=config.dropout
            )
        
        # 输出层
        self.output_layer = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
    
    def forward(self, node_features: torch.Tensor, 
                edge_index: torch.Tensor,
                batch_size: Optional[int] = None) -> torch.Tensor:
        """
        生成图嵌入
        
        Args:
            node_features: 节点特征 [num_nodes, embedding_dim]
            edge_index: 边索引 [2, num_edges]
            batch_size: 批次大小
            
        Returns:
            图嵌入表示 [batch_size, embedding_dim]
        """
        x = node_features
        
        # 图神经网络处理
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x)
            if i < len(self.gnn_layers) - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        
        # 注意力机制
        if self.config.use_attention:
            x = x.unsqueeze(0)  # [1, num_nodes, embedding_dim]
            attn_output, _ = self.attention(x, x, x)
            x = attn_output.squeeze(0)  # [num_nodes, embedding_dim]
        
        # 聚合节点特征
        if self.config.aggregation_method == "mean":
            graph_embedding = torch.mean(x, dim=0)
        elif self.config.aggregation_method == "sum":
            graph_embedding = torch.sum(x, dim=0)
        elif self.config.aggregation_method == "max":
            graph_embedding = torch.max(x, dim=0)[0]
        else:
            graph_embedding = torch.mean(x, dim=0)
        
        # 输出处理
        graph_embedding = self.output_layer(graph_embedding)
        graph_embedding = self.layer_norm(graph_embedding)
        
        return graph_embedding
    
    def generate_embeddings(self, graph_data: Dict[str, Any]) -> torch.Tensor:
        """
        从图数据生成嵌入
        
        Args:
            graph_data: 包含节点和边信息的字典
            
        Returns:
            图嵌入向量
        """
        node_features = graph_data.get('node_features', torch.randn(10, self.config.embedding_dim))
        edge_index = graph_data.get('edge_index', torch.tensor([[0, 1], [1, 0]]))
        
        return self.forward(node_features, edge_index)


class GraphEmbeddingManager:
    """图嵌入管理器"""
    
    def __init__(self, config: GraphEmbeddingConfig):
        self.config = config
        self.generator = GraphEmbeddingGenerator(config)
        self.embeddings_cache = {}
    
    def get_embedding(self, graph_id: str, graph_data: Dict[str, Any]) -> torch.Tensor:
        """
        获取图嵌入，支持缓存
        
        Args:
            graph_id: 图标识符
            graph_data: 图数据
            
        Returns:
            图嵌入向量
        """
        if graph_id in self.embeddings_cache:
            return self.embeddings_cache[graph_id]
        
        embedding = self.generator.generate_embeddings(graph_data)
        self.embeddings_cache[graph_id] = embedding
        
        return embedding
    
    def clear_cache(self):
        """清空缓存"""
        self.embeddings_cache.clear()
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self.embeddings_cache) 