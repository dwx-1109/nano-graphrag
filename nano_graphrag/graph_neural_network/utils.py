"""
图神经网络工具模块
提供图神经网络相关的工具函数和辅助功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import networkx as nx
from dataclasses import dataclass


@dataclass
class GraphUtilsConfig:
    """图工具配置"""
    max_nodes: int = 1000
    max_edges: int = 5000
    embedding_dim: int = 768
    use_sparse: bool = False
    normalize_features: bool = True


class GraphUtils:
    """图工具类"""
    
    def __init__(self, config: GraphUtilsConfig):
        self.config = config
    
    def networkx_to_torch(self, G: nx.Graph) -> Dict[str, torch.Tensor]:
        """
        将NetworkX图转换为PyTorch张量格式
        
        Args:
            G: NetworkX图对象
            
        Returns:
            包含节点特征、边索引等的字典
        """
        # 节点特征
        node_features = []
        node_mapping = {}
        
        for i, node in enumerate(G.nodes()):
            node_mapping[node] = i
            # 提取节点特征，如果没有则使用默认特征
            if 'features' in G.nodes[node]:
                features = G.nodes[node]['features']
            else:
                features = np.random.randn(self.config.embedding_dim)
            node_features.append(features)
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # 边索引
        edge_index = []
        edge_features = []
        
        for u, v, data in G.edges(data=True):
            if u in node_mapping and v in node_mapping:
                edge_index.append([node_mapping[u], node_mapping[v]])
                # 提取边特征，如果没有则使用默认特征
                if 'features' in data:
                    features = data['features']
                else:
                    features = np.random.randn(self.config.embedding_dim)
                edge_features.append(features)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_features = torch.tensor(edge_features, dtype=torch.float32) if edge_features else None
        
        # 归一化特征
        if self.config.normalize_features:
            node_features = F.normalize(node_features, p=2, dim=1)
            if edge_features is not None:
                edge_features = F.normalize(edge_features, p=2, dim=1)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'node_mapping': node_mapping,
            'num_nodes': len(G.nodes()),
            'num_edges': len(G.edges())
        }
    
    def torch_to_networkx(self, node_features: torch.Tensor,
                         edge_index: torch.Tensor,
                         edge_features: Optional[torch.Tensor] = None) -> nx.Graph:
        """
        将PyTorch张量格式转换为NetworkX图
        
        Args:
            node_features: 节点特征 [num_nodes, embedding_dim]
            edge_index: 边索引 [2, num_edges]
            edge_features: 边特征 [num_edges, embedding_dim]
            
        Returns:
            NetworkX图对象
        """
        G = nx.Graph()
        
        # 添加节点
        for i in range(node_features.size(0)):
            G.add_node(i, features=node_features[i].numpy())
        
        # 添加边
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if edge_features is not None:
                G.add_edge(u, v, features=edge_features[i].numpy())
            else:
                G.add_edge(u, v)
        
        return G
    
    def create_subgraph(self, node_features: torch.Tensor,
                       edge_index: torch.Tensor,
                       node_indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        创建子图
        
        Args:
            node_features: 原始节点特征
            edge_index: 原始边索引
            node_indices: 要包含的节点索引
            
        Returns:
            子图的张量表示
        """
        # 创建节点映射
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # 提取子图节点特征
        sub_node_features = node_features[node_indices]
        
        # 提取子图边
        sub_edge_index = []
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u in node_mapping and v in node_mapping:
                sub_edge_index.append([node_mapping[u], node_mapping[v]])
        
        sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.long).t() if sub_edge_index else torch.empty((2, 0), dtype=torch.long)
        
        return {
            'node_features': sub_node_features,
            'edge_index': sub_edge_index,
            'num_nodes': len(node_indices),
            'num_edges': sub_edge_index.size(1)
        }
    
    def compute_graph_statistics(self, node_features: torch.Tensor,
                               edge_index: torch.Tensor) -> Dict[str, float]:
        """
        计算图统计信息
        
        Args:
            node_features: 节点特征
            edge_index: 边索引
            
        Returns:
            图统计信息字典
        """
        num_nodes = node_features.size(0)
        num_edges = edge_index.size(1)
        
        # 计算度分布
        degrees = torch.zeros(num_nodes)
        for i in range(num_edges):
            u, v = edge_index[0, i], edge_index[1, i]
            degrees[u] += 1
            degrees[v] += 1
        
        # 计算聚类系数
        clustering_coeff = self._compute_clustering_coefficient(edge_index, num_nodes)
        
        # 计算特征统计
        feature_mean = torch.mean(node_features, dim=0)
        feature_std = torch.std(node_features, dim=0)
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': torch.mean(degrees).item(),
            'max_degree': torch.max(degrees).item(),
            'min_degree': torch.min(degrees).item(),
            'clustering_coefficient': clustering_coeff,
            'feature_mean': feature_mean.mean().item(),
            'feature_std': feature_std.mean().item(),
            'density': (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        }
    
    def _compute_clustering_coefficient(self, edge_index: torch.Tensor,
                                      num_nodes: int) -> float:
        """计算聚类系数"""
        # 构建邻接矩阵
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i], edge_index[1, i]
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
        
        # 计算三角形数量
        triangles = torch.trace(torch.matrix_power(adj_matrix, 3)) / 6
        
        # 计算三元组数量
        triples = torch.sum(torch.matrix_power(adj_matrix, 2)) / 2
        
        return (triangles / triples).item() if triples > 0 else 0.0
    
    def normalize_features(self, features: torch.Tensor,
                          method: str = 'l2') -> torch.Tensor:
        """
        归一化特征
        
        Args:
            features: 输入特征
            method: 归一化方法 ('l2', 'minmax', 'zscore')
            
        Returns:
            归一化后的特征
        """
        if method == 'l2':
            return F.normalize(features, p=2, dim=1)
        elif method == 'minmax':
            min_vals = torch.min(features, dim=1, keepdim=True)[0]
            max_vals = torch.max(features, dim=1, keepdim=True)[0]
            return (features - min_vals) / (max_vals - min_vals + 1e-8)
        elif method == 'zscore':
            mean_vals = torch.mean(features, dim=1, keepdim=True)
            std_vals = torch.std(features, dim=1, keepdim=True)
            return (features - mean_vals) / (std_vals + 1e-8)
        else:
            return features
    
    def add_self_loops(self, edge_index: torch.Tensor,
                      num_nodes: int) -> torch.Tensor:
        """
        添加自环
        
        Args:
            edge_index: 边索引
            num_nodes: 节点数量
            
        Returns:
            添加自环后的边索引
        """
        self_loops = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        return torch.cat([edge_index, self_loops], dim=1)
    
    def remove_self_loops(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        移除自环
        
        Args:
            edge_index: 边索引
            
        Returns:
            移除自环后的边索引
        """
        mask = edge_index[0] != edge_index[1]
        return edge_index[:, mask]


class GraphDataLoader:
    """图数据加载器"""
    
    def __init__(self, config: GraphUtilsConfig):
        self.config = config
        self.utils = GraphUtils(config)
    
    def load_graph_data(self, file_path: str) -> Dict[str, torch.Tensor]:
        """
        从文件加载图数据
        
        Args:
            file_path: 图数据文件路径
            
        Returns:
            图数据字典
        """
        # 这里可以根据实际文件格式实现加载逻辑
        # 目前返回示例数据
        num_nodes = min(100, self.config.max_nodes)
        num_edges = min(200, self.config.max_edges)
        
        # 生成随机图数据
        node_features = torch.randn(num_nodes, self.config.embedding_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'num_nodes': num_nodes,
            'num_edges': num_edges
        }
    
    def batch_graphs(self, graph_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        批处理多个图
        
        Args:
            graph_list: 图数据列表
            
        Returns:
            批处理后的图数据
        """
        # 实现图批处理逻辑
        # 这里简化处理，实际应该考虑不同大小的图
        batched_features = torch.cat([g['node_features'] for g in graph_list], dim=0)
        batched_edges = torch.cat([g['edge_index'] for g in graph_list], dim=1)
        
        return {
            'node_features': batched_features,
            'edge_index': batched_edges,
            'batch_size': len(graph_list)
        } 