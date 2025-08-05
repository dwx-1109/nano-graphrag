"""
Multi-Head Attention Fusion: 基于注意力机制的多源融合

基于Transformer的Multi-Head Attention机制，实现多源检索结果的智能融合。
这是项目的第二大创新算法，解决传统融合方法无法捕获源间复杂关系的问题。

核心算法创新:
1. 查询感知注意力：使用查询作为query，检索结果作为key/value
2. 源间交互建模：捕获不同检索源之间的语义关系
3. 位置编码：考虑检索结果在各源中的排名信息
4. 分层注意力：实现文档级和段落级的层次化融合

科研贡献:
- 首创基于注意力机制的多源RAG融合架构
- 解决传统线性/RRF融合的局限性
- 提供可解释的融合权重分析
- 支持动态源权重学习

参考文献:
Vaswani, A., et al. (2017). Attention is all you need. NIPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MultiHeadAttentionFusionConfig:
    """Multi-Head Attention Fusion配置"""
    
    # 注意力机制参数
    d_model: int = 768  # 模型维度 (与BERT嵌入维度对齐)
    num_heads: int = 12  # 注意力头数
    d_ff: int = 2048    # 前馈网络维度
    
    # Dropout参数
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # 位置编码参数
    max_position_embeddings: int = 512
    enable_position_encoding: bool = True
    position_encoding_type: str = "sinusoidal"  # sinusoidal, learned
    
    # 融合策略参数
    fusion_layers: int = 3  # Transformer层数
    enable_cross_attention: bool = True  # 是否启用交叉注意力
    enable_self_attention: bool = True   # 是否启用自注意力
    
    # 查询增强参数
    enable_query_conditioning: bool = True
    query_projection_dim: int = 768
    
    # 排名感知参数
    enable_rank_encoding: bool = True
    max_rank: int = 100
    rank_embedding_dim: int = 64
    
    # 源特异性参数
    enable_source_embedding: bool = True
    num_sources: int = 4  # naive, bm25, local, global
    source_embedding_dim: int = 64
    
    # 温度参数
    temperature_init: float = 1.0
    learnable_temperature: bool = True
    
    # 正则化参数
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 512, encoding_type: str = "sinusoidal"):
        super().__init__()
        self.d_model = d_model
        self.encoding_type = encoding_type
        
        if encoding_type == "sinusoidal":
            # 正弦位置编码
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe.unsqueeze(0))
            
        elif encoding_type == "learned":
            # 可学习位置编码
            self.position_embeddings = nn.Embedding(max_len, d_model)
            nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            positions: 位置索引 [batch_size, seq_len]
            
        Returns:
            添加位置编码后的张量
        """
        if self.encoding_type == "sinusoidal":
            seq_len = x.size(1)
            return x + self.pe[:, :seq_len, :]
        else:
            if positions is None:
                positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
            return x + self.position_embeddings(positions)


class RankEncoding(nn.Module):
    """排名编码模块 - 编码检索结果在源中的排名信息"""
    
    def __init__(self, config: MultiHeadAttentionFusionConfig):
        super().__init__()
        self.config = config
        
        # 排名嵌入
        self.rank_embeddings = nn.Embedding(config.max_rank + 1, config.rank_embedding_dim)
        
        # 排名投影
        self.rank_projection = nn.Linear(config.rank_embedding_dim, config.d_model)
        
        # 初始化
        nn.init.normal_(self.rank_embeddings.weight, std=0.02)
        nn.init.xavier_uniform_(self.rank_projection.weight)
    
    def forward(self, ranks: torch.Tensor) -> torch.Tensor:
        """
        生成排名编码
        
        Args:
            ranks: 排名张量 [batch_size, num_sources, max_docs_per_source]
            
        Returns:
            排名编码 [batch_size, num_sources, max_docs_per_source, d_model]
        """
        # 限制排名范围并转换为整数类型
        ranks_clamped = torch.clamp(ranks, 0, self.config.max_rank).long()
        
        # 获取排名嵌入
        rank_emb = self.rank_embeddings(ranks_clamped)  # [..., rank_embedding_dim]
        
        # 投影到模型维度
        rank_encoding = self.rank_projection(rank_emb)  # [..., d_model]
        
        return rank_encoding


class SourceEncoding(nn.Module):
    """源编码模块 - 编码不同检索源的特征"""
    
    def __init__(self, config: MultiHeadAttentionFusionConfig):
        super().__init__()
        self.config = config
        
        # 源嵌入
        self.source_embeddings = nn.Embedding(config.num_sources, config.source_embedding_dim)
        
        # 源投影
        self.source_projection = nn.Linear(config.source_embedding_dim, config.d_model)
        
        # 源标识映射
        self.source_to_id = {
            "naive": 0,
            "bm25": 1, 
            "local": 2,
            "global": 3
        }
        
        # 初始化
        nn.init.normal_(self.source_embeddings.weight, std=0.02)
        nn.init.xavier_uniform_(self.source_projection.weight)
    
    def forward(self, source_names: List[str], batch_size: int, seq_len: int) -> torch.Tensor:
        """
        生成源编码
        
        Args:
            source_names: 源名称列表
            batch_size: 批次大小
            seq_len: 序列长度
            
        Returns:
            源编码 [batch_size, seq_len, d_model]
        """
        # 转换源名称为ID
        source_ids = [self.source_to_id.get(name, 0) for name in source_names]
        source_tensor = torch.tensor(source_ids, dtype=torch.long)
        
        # 扩展到批次和序列维度
        source_tensor = source_tensor.unsqueeze(0).expand(batch_size, len(source_names))
        
        # 如果序列长度与源数量不匹配，需要调整
        if seq_len != len(source_names):
            # 重复或截断以匹配seq_len
            if seq_len > len(source_names):
                repeat_factor = seq_len // len(source_names) + 1
                source_tensor = source_tensor.repeat(1, repeat_factor)[:, :seq_len]
            else:
                source_tensor = source_tensor[:, :seq_len]
        
        # 获取源嵌入
        source_emb = self.source_embeddings(source_tensor)  # [batch, seq_len, source_embedding_dim]
        
        # 投影到模型维度
        source_encoding = self.source_projection(source_emb)  # [batch, seq_len, d_model]
        
        return source_encoding


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention模块"""
    
    def __init__(self, config: MultiHeadAttentionFusionConfig):
        super().__init__()
        self.config = config
        
        assert config.d_model % config.num_heads == 0
        self.d_k = config.d_model // config.num_heads
        
        # 线性投影层
        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        
        # Dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # 可学习温度参数
        if config.learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(config.temperature_init))
        else:
            self.register_buffer('temperature', torch.tensor(config.temperature_init))
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Multi-Head Attention前向传播
        
        Args:
            query: Query张量 [batch_size, query_len, d_model]
            key: Key张量 [batch_size, key_len, d_model]  
            value: Value张量 [batch_size, value_len, d_model]
            mask: 注意力掩码 [batch_size, query_len, key_len]
            return_attention: 是否返回注意力权重
            
        Returns:
            输出张量和可选的注意力权重
        """
        batch_size, query_len, d_model = query.size()
        key_len = key.size(1)
        
        # 线性投影
        Q = self.w_q(query)  # [batch, query_len, d_model]
        K = self.w_k(key)    # [batch, key_len, d_model]
        V = self.w_v(value)  # [batch, value_len, d_model]
        
        # 重塑为多头形式
        Q = Q.view(batch_size, query_len, self.config.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, key_len, self.config.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, key_len, self.config.num_heads, self.d_k).transpose(1, 2)
        # 现在形状为 [batch, num_heads, seq_len, d_k]
        
        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.d_k) * self.temperature)
        # [batch, num_heads, query_len, key_len]
        
        # 应用掩码
        if mask is not None:
            # 扩展掩码到多头维度
            mask = mask.unsqueeze(1).expand(-1, self.config.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, query_len, key_len]
        attention_weights = self.attention_dropout(attention_weights)
        
        # 应用注意力权重
        attended_values = torch.matmul(attention_weights, V)  # [batch, num_heads, query_len, d_k]
        
        # 拼接多头输出
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, query_len, d_model
        )
        
        # 输出投影
        output = self.w_o(attended_values)
        output = self.output_dropout(output)
        
        if return_attention:
            # 平均多头注意力权重用于可视化
            avg_attention = attention_weights.mean(dim=1)  # [batch, query_len, key_len]
            return output, avg_attention
        
        return output


class TransformerLayer(nn.Module):
    """Transformer层 - 包含自注意力和前馈网络"""
    
    def __init__(self, config: MultiHeadAttentionFusionConfig):
        super().__init__()
        self.config = config
        
        # 自注意力
        self.self_attention = MultiHeadAttention(config)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Layer Norm
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Transformer层前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
            return_attention: 是否返回注意力权重
            
        Returns:
            输出张量和可选的注意力权重
        """
        # 自注意力 + 残差连接
        if return_attention:
            attn_output, attention_weights = self.self_attention(
                x, x, x, mask=mask, return_attention=True
            )
        else:
            attn_output = self.self_attention(x, x, x, mask=mask)
            attention_weights = None
        
        x = self.ln1(x + attn_output)
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.ln2(x + ff_output)
        
        if return_attention:
            return x, attention_weights
        return x


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合 - 使用查询作为query，文档作为key/value"""
    
    def __init__(self, config: MultiHeadAttentionFusionConfig):
        super().__init__()
        self.config = config
        
        # 查询投影
        self.query_projection = nn.Linear(config.query_projection_dim, config.d_model)
        
        # 交叉注意力
        self.cross_attention = MultiHeadAttention(config)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, 
                query_embedding: torch.Tensor,
                document_embeddings: torch.Tensor,
                document_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        交叉注意力融合
        
        Args:
            query_embedding: 查询嵌入 [batch_size, query_dim]
            document_embeddings: 文档嵌入 [batch_size, num_docs, d_model]
            document_mask: 文档掩码 [batch_size, num_docs]
            return_attention: 是否返回注意力权重
            
        Returns:
            融合后的文档表示和可选的注意力权重
        """
        batch_size = query_embedding.size(0)
        num_docs = document_embeddings.size(1)
        
        # 投影查询到模型维度
        query_projected = self.query_projection(query_embedding)  # [batch, d_model]
        query_projected = query_projected.unsqueeze(1)  # [batch, 1, d_model]
        
        # 构建注意力掩码
        if document_mask is not None:
            # 扩展掩码维度：[batch, 1, num_docs]
            attention_mask = document_mask.unsqueeze(1)
        else:
            attention_mask = None
        
        # 交叉注意力：query attend to documents
        if return_attention:
            attended_docs, attention_weights = self.cross_attention(
                query_projected, document_embeddings, document_embeddings,
                mask=attention_mask, return_attention=True
            )
            # attention_weights: [batch, 1, num_docs]
            attention_weights = attention_weights.squeeze(1)  # [batch, num_docs]
        else:
            attended_docs = self.cross_attention(
                query_projected, document_embeddings, document_embeddings,
                mask=attention_mask
            )
            attention_weights = None
        
        # 输出处理
        attended_docs = attended_docs.squeeze(1)  # [batch, d_model]
        output = self.output_layer(attended_docs)
        
        if return_attention:
            return output, attention_weights
        return output


class SelfAttentionFusion(nn.Module):
    """自注意力融合 - 文档间相互注意力"""
    
    def __init__(self, config: MultiHeadAttentionFusionConfig):
        super().__init__()
        self.config = config
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.fusion_layers)
        ])
        
        # 池化层
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # 输出投影
        self.output_projection = nn.Linear(config.d_model, config.d_model)
    
    def forward(self, 
                document_embeddings: torch.Tensor,
                document_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        自注意力融合
        
        Args:
            document_embeddings: 文档嵌入 [batch_size, num_docs, d_model]
            document_mask: 文档掩码 [batch_size, num_docs]
            return_attention: 是否返回注意力权重
            
        Returns:
            融合后的表示和可选的注意力权重列表
        """
        x = document_embeddings
        attention_weights_list = []
        
        # 通过Transformer层
        for layer in self.transformer_layers:
            if return_attention:
                x, attention_weights = layer(x, mask=document_mask, return_attention=True)
                attention_weights_list.append(attention_weights)
            else:
                x = layer(x, mask=document_mask)
        
        # 池化到单一表示
        if document_mask is not None:
            # 掩码池化
            masked_x = x * document_mask.unsqueeze(-1).float()
            doc_count = document_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            pooled = masked_x.sum(dim=1) / doc_count  # [batch, d_model]
        else:
            pooled = x.mean(dim=1)  # [batch, d_model]
        
        # 输出投影
        output = self.output_projection(pooled)
        
        if return_attention:
            return output, attention_weights_list
        return output


class MultiHeadAttentionFusion(nn.Module):
    """
    Multi-Head Attention Fusion主模型
    
    实现基于Transformer注意力机制的多源检索结果融合，
    这是项目的第二大核心创新算法。
    """
    
    def __init__(self, config: MultiHeadAttentionFusionConfig):
        super().__init__()
        self.config = config
        
        # 位置编码
        if config.enable_position_encoding:
            self.position_encoding = PositionalEncoding(
                config.d_model, config.max_position_embeddings, config.position_encoding_type
            )
        
        # 排名编码
        if config.enable_rank_encoding:
            self.rank_encoding = RankEncoding(config)
        
        # 源编码
        if config.enable_source_embedding:
            self.source_encoding = SourceEncoding(config)
        
        # 输入投影层
        self.input_projection = nn.Linear(config.d_model, config.d_model)
        
        # 交叉注意力融合
        if config.enable_cross_attention:
            self.cross_attention_fusion = CrossAttentionFusion(config)
        
        # 自注意力融合
        if config.enable_self_attention:
            self.self_attention_fusion = SelfAttentionFusion(config)
        
        # 最终融合层
        if config.enable_cross_attention and config.enable_self_attention:
            self.final_fusion = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.d_model)
            )
        
        # 初始化权重
        self.apply(self._init_weights)
        
        logger.info(f"MultiHeadAttentionFusion initialized with {config.fusion_layers} layers")
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, 
                query_embedding: torch.Tensor,
                document_embeddings: torch.Tensor,
                source_names: List[str],
                document_ranks: Optional[torch.Tensor] = None,
                document_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Multi-Head Attention Fusion前向传播
        
        Args:
            query_embedding: 查询嵌入 [batch_size, query_dim] 或 [query_dim]
            document_embeddings: 文档嵌入 [batch_size, num_docs, d_model] 或 [num_docs, d_model]
            source_names: 源名称列表
            document_ranks: 文档排名 [batch_size, num_docs] 或 [num_docs]
            document_mask: 文档掩码 [batch_size, num_docs] 或 [num_docs]
            return_attention: 是否返回注意力权重
            
        Returns:
            包含融合结果和注意力权重的字典
        """
        # 处理输入张量维度，支持2D和3D输入
        original_2d = False
        if len(document_embeddings.size()) == 2:
            # 如果是2D张量，添加batch维度
            original_2d = True
            num_docs, embed_dim = document_embeddings.size()
            batch_size = 1
            document_embeddings = document_embeddings.unsqueeze(0)  # [1, num_docs, embed_dim]
            
            # 相应地处理其他输入
            if len(query_embedding.size()) == 1:
                query_embedding = query_embedding.unsqueeze(0)  # [1, query_dim]
            if document_ranks is not None and len(document_ranks.size()) == 1:
                document_ranks = document_ranks.unsqueeze(0)  # [1, num_docs]
            if document_mask is not None and len(document_mask.size()) == 1:
                document_mask = document_mask.unsqueeze(0)  # [1, num_docs]
        else:
            batch_size, num_docs, embed_dim = document_embeddings.size()
        
        # 1. 输入投影
        doc_embeddings = self.input_projection(document_embeddings)
        
        # 2. 添加位置编码
        if self.config.enable_position_encoding and hasattr(self, 'position_encoding'):
            doc_embeddings = self.position_encoding(doc_embeddings)
        
        # 3. 添加排名编码
        if self.config.enable_rank_encoding and hasattr(self, 'rank_encoding') and document_ranks is not None:
            rank_encodings = self.rank_encoding(document_ranks.unsqueeze(1))  # [batch, 1, num_docs, d_model]
            rank_encodings = rank_encodings.squeeze(1)  # [batch, num_docs, d_model]
            doc_embeddings = doc_embeddings + rank_encodings
        
        # 4. 添加源编码
        if self.config.enable_source_embedding and hasattr(self, 'source_encoding'):
            source_encodings = self.source_encoding(source_names, batch_size, num_docs)
            doc_embeddings = doc_embeddings + source_encodings
        
        # 5. 交叉注意力融合
        result = {}
        if self.config.enable_cross_attention and hasattr(self, 'cross_attention_fusion'):
            if return_attention:
                cross_output, cross_attention = self.cross_attention_fusion(
                    query_embedding, doc_embeddings, document_mask, return_attention=True
                )
                result['cross_attention_weights'] = cross_attention
            else:
                cross_output = self.cross_attention_fusion(
                    query_embedding, doc_embeddings, document_mask
                )
            result['cross_attention_output'] = cross_output
        
        # 6. 自注意力融合
        if self.config.enable_self_attention and hasattr(self, 'self_attention_fusion'):
            if return_attention:
                self_output, self_attention = self.self_attention_fusion(
                    doc_embeddings, document_mask, return_attention=True
                )
                result['self_attention_weights'] = self_attention
            else:
                self_output = self.self_attention_fusion(
                    doc_embeddings, document_mask
                )
            result['self_attention_output'] = self_output
        
        # 7. 最终融合
        if self.config.enable_cross_attention and self.config.enable_self_attention:
            combined = torch.cat([cross_output, self_output], dim=-1)
            final_output = self.final_fusion(combined)
        elif self.config.enable_cross_attention:
            final_output = cross_output
        elif self.config.enable_self_attention:
            final_output = self_output
        else:
            raise ValueError("At least one of cross_attention or self_attention must be enabled")
        
        # 如果原始输入是2D张量，则将输出也恢复为相应的维度
        if original_2d:
            final_output = final_output.squeeze(0)  # [output_dim] 或 [num_docs, output_dim]
            # 也需要处理结果字典中的其他张量
            for key, value in result.items():
                if isinstance(value, torch.Tensor) and len(value.size()) > 0:
                    if value.size(0) == 1:  # 如果第一维是batch_size=1
                        result[key] = value.squeeze(0)
        
        result['fused_representation'] = final_output
        
        return result
    
    def get_attention_visualization_data(self, 
                                        query_embedding: torch.Tensor,
                                        document_embeddings: torch.Tensor,
                                        source_names: List[str],
                                        document_texts: List[List[str]],
                                        document_ranks: Optional[torch.Tensor] = None,
                                        document_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        获取注意力可视化数据
        
        Args:
            query_embedding: 查询嵌入
            document_embeddings: 文档嵌入
            source_names: 源名称列表
            document_texts: 文档文本列表
            document_ranks: 文档排名
            document_mask: 文档掩码
            
        Returns:
            可视化数据字典
        """
        with torch.no_grad():
            result = self.forward(
                query_embedding, document_embeddings, source_names,
                document_ranks, document_mask, return_attention=True
            )
        
        visualization_data = {
            'source_names': source_names,
            'document_texts': document_texts,
            'attention_weights': result.get('cross_attention_weights'),
            'self_attention_weights': result.get('self_attention_weights'),
            'document_ranks': document_ranks,
            'fused_representation': result['fused_representation']
        }
        
        return visualization_data


# 工厂函数
def create_multihead_attention_fusion(config: Optional[MultiHeadAttentionFusionConfig] = None) -> MultiHeadAttentionFusion:
    """创建Multi-Head Attention Fusion模型"""
    if config is None:
        config = MultiHeadAttentionFusionConfig()
    return MultiHeadAttentionFusion(config)


def create_multihead_attention_fusion_config(**kwargs) -> MultiHeadAttentionFusionConfig:
    """创建Multi-Head Attention Fusion配置"""
    return MultiHeadAttentionFusionConfig(**kwargs) 