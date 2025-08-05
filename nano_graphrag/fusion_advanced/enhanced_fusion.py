"""
增强融合策略模块
基于现有multihead_attention_fusion.py进行改进
参考最新的融合技术：RankFusion, CombSUM等
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .multihead_attention_fusion import MultiHeadAttentionFusion, MultiHeadAttentionFusionConfig

logger = logging.getLogger(__name__)

@dataclass
class EnhancedFusionConfig:
    """增强融合配置"""
    
    # 基础配置
    base_config: MultiHeadAttentionFusionConfig = field(default_factory=MultiHeadAttentionFusionConfig)
    
    # RankFusion参数
    enable_rank_fusion: bool = True
    rank_fusion_method: str = "rrf"  # rrf, combsum, combmnz
    rrf_k: float = 60.0  # RRF参数
    
    # 自适应权重参数
    enable_adaptive_weights: bool = True
    weight_learning_rate: float = 0.01
    weight_momentum: float = 0.9
    
    # 查询感知融合参数
    enable_query_aware_fusion: bool = True
    query_projection_dim: int = 128
    
    # 动态温度参数
    enable_dynamic_temperature: bool = True
    temperature_min: float = 0.1
    temperature_max: float = 2.0
    
    # 融合质量估计
    enable_fusion_quality_estimation: bool = True
    quality_threshold: float = 0.6

class RankBasedFusion(nn.Module):
    """基于排名的融合方法"""
    
    def __init__(self, config: EnhancedFusionConfig):
        super().__init__()
        self.config = config
        self.method = config.rank_fusion_method
        self.rrf_k = config.rrf_k
    
    def forward(self, 
                scores_list: List[torch.Tensor],
                source_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        融合多个检索源的分数
        
        Args:
            scores_list: 每个源的分数列表 [num_sources, num_docs]
            source_weights: 源权重 [num_sources]
        """
        if self.method == "rrf":
            return self._reciprocal_rank_fusion(scores_list, source_weights)
        elif self.method == "combsum":
            return self._combsum_fusion(scores_list, source_weights)
        elif self.method == "combmnz":
            return self._combmnz_fusion(scores_list, source_weights)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
    
    def _reciprocal_rank_fusion(self, 
                               scores_list: List[torch.Tensor],
                               source_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """倒数排名融合 (RRF)"""
        device = scores_list[0].device
        num_docs = scores_list[0].shape[0]
        
        # 转换分数为排名
        fused_scores = torch.zeros(num_docs, device=device)
        
        for i, scores in enumerate(scores_list):
            # 计算排名（分数越高排名越前）
            ranks = torch.argsort(torch.argsort(scores, descending=True)) + 1
            
            # RRF公式：1 / (k + rank)
            rrf_scores = 1.0 / (self.rrf_k + ranks.float())
            
            # 应用源权重
            if source_weights is not None:
                rrf_scores *= source_weights[i]
            
            fused_scores += rrf_scores
        
        return fused_scores
    
    def _combsum_fusion(self, 
                       scores_list: List[torch.Tensor],
                       source_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """CombSUM融合：归一化分数求和"""
        device = scores_list[0].device
        fused_scores = torch.zeros_like(scores_list[0])
        
        for i, scores in enumerate(scores_list):
            # 归一化分数到[0,1]
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            
            # 应用源权重
            if source_weights is not None:
                normalized_scores *= source_weights[i]
            
            fused_scores += normalized_scores
        
        return fused_scores
    
    def _combmnz_fusion(self, 
                       scores_list: List[torch.Tensor],
                       source_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """CombMNZ融合：考虑非零源数量"""
        # 先计算CombSUM
        combsum_scores = self._combsum_fusion(scores_list, source_weights)
        
        # 计算每个文档的非零源数量
        non_zero_counts = torch.zeros_like(combsum_scores)
        for scores in scores_list:
            non_zero_counts += (scores > 0).float()
        
        # CombMNZ = CombSUM * non_zero_count
        return combsum_scores * non_zero_counts

class AdaptiveWeightLearner(nn.Module):
    """自适应权重学习器"""
    
    def __init__(self, config: EnhancedFusionConfig):
        super().__init__()
        self.config = config
        
        # 权重网络
        self.weight_network = nn.Sequential(
            nn.Linear(config.query_projection_dim + 4, 64),  # query + 4个源的特征
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4个源的权重
            nn.Softmax(dim=-1)
        )
        
        # 历史权重的动量更新
        self.register_buffer('weight_momentum', torch.ones(4) / 4)
        
    def forward(self, 
                query_embedding: torch.Tensor,
                source_features: torch.Tensor,
                historical_performance: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        学习自适应权重
        
        Args:
            query_embedding: 查询嵌入 [query_dim]
            source_features: 源特征 [4] (平均分数、方差、覆盖度、新颖性)
            historical_performance: 历史性能 [4]
        """
        # 拼接特征
        features = torch.cat([query_embedding, source_features], dim=-1)
        
        # 预测权重
        predicted_weights = self.weight_network(features)
        
        # 动量更新
        if self.training:
            self.weight_momentum = (
                self.config.weight_momentum * self.weight_momentum + 
                (1 - self.config.weight_momentum) * predicted_weights.detach()
            )
        
        return predicted_weights

class QueryAwareFusion(nn.Module):
    """查询感知融合模块"""
    
    def __init__(self, config: EnhancedFusionConfig):
        super().__init__()
        self.config = config
        
        # 查询分析网络
        self.query_analyzer = nn.Sequential(
            nn.Linear(config.base_config.d_model, config.query_projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.query_projection_dim, config.query_projection_dim)
        )
        
        # 查询类型分类器
        self.query_type_classifier = nn.Sequential(
            nn.Linear(config.query_projection_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # factual, analytical, creative
        )
        
        # 类型特异性权重
        self.type_specific_weights = nn.Parameter(
            torch.randn(3, 4) * 0.1  # 3个查询类型 x 4个源
        )
    
    def forward(self, 
                query_embedding: torch.Tensor,
                source_scores: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        查询感知融合
        
        Args:
            query_embedding: 查询嵌入
            source_scores: 各源分数列表
        """
        # 分析查询特征
        query_features = self.query_analyzer(query_embedding)
        
        # 预测查询类型
        query_type_logits = self.query_type_classifier(query_features)
        query_type_probs = F.softmax(query_type_logits, dim=-1)
        
        # 基于查询类型调整权重
        type_weights = torch.matmul(query_type_probs, self.type_specific_weights)
        type_weights = F.softmax(type_weights, dim=-1)
        
        # 应用权重融合
        fused_scores = torch.zeros_like(source_scores[0])
        for i, scores in enumerate(source_scores):
            fused_scores += type_weights[i] * scores
        
        info = {
            'query_type_probs': query_type_probs,
            'type_weights': type_weights,
            'query_features': query_features
        }
        
        return fused_scores, info

class DynamicTemperatureController(nn.Module):
    """动态温度控制器"""
    
    def __init__(self, config: EnhancedFusionConfig):
        super().__init__()
        self.config = config
        
        # 温度预测网络
        self.temperature_network = nn.Sequential(
            nn.Linear(config.query_projection_dim + 1, 32),  # query + confidence
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                query_features: torch.Tensor,
                fusion_confidence: torch.Tensor) -> torch.Tensor:
        """预测动态温度"""
        # 拼接特征
        features = torch.cat([query_features, fusion_confidence.unsqueeze(-1)], dim=-1)
        
        # 预测温度
        temp_factor = self.temperature_network(features)
        
        # 缩放到指定范围
        temperature = (
            self.config.temperature_min + 
            temp_factor * (self.config.temperature_max - self.config.temperature_min)
        )
        
        return temperature

class FusionQualityEstimator(nn.Module):
    """融合质量估计器"""
    
    def __init__(self, config: EnhancedFusionConfig):
        super().__init__()
        self.config = config
        
        # 质量估计网络
        self.quality_network = nn.Sequential(
            nn.Linear(8, 32),  # 8个特征：4个源分数统计 + 4个一致性指标
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                source_scores: List[torch.Tensor],
                fused_scores: torch.Tensor) -> torch.Tensor:
        """估计融合质量"""
        # 计算统计特征
        features = []
        
        # 每个源的分数统计
        for scores in source_scores:
            features.extend([
                scores.mean().item(),
                scores.std().item()
            ])
        
        # 一致性指标
        score_matrix = torch.stack(source_scores, dim=1)  # [num_docs, num_sources]
        
        # 计算源间相关性
        correlations = []
        for i in range(len(source_scores)):
            for j in range(i+1, len(source_scores)):
                corr = torch.corrcoef(torch.stack([source_scores[i], source_scores[j]]))[0, 1]
                correlations.append(corr.item() if not torch.isnan(corr) else 0.0)
        
        # 平均相关性作为一致性指标
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # 融合分数的分布特征
        fusion_entropy = -torch.sum(F.softmax(fused_scores, dim=0) * F.log_softmax(fused_scores, dim=0))
        
        features.extend([avg_correlation, fusion_entropy.item()])
        
        # 预测质量
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=fused_scores.device)
        quality_score = self.quality_network(feature_tensor)
        
        return quality_score

class EnhancedFusion(nn.Module):
    """增强融合模块：集成多种先进融合策略"""
    
    def __init__(self, config: EnhancedFusionConfig):
        super().__init__()
        self.config = config
        
        # 基础多头注意力融合
        self.base_fusion = MultiHeadAttentionFusion(config.base_config)
        
        # 各个增强模块
        if config.enable_rank_fusion:
            self.rank_fusion = RankBasedFusion(config)
        
        if config.enable_adaptive_weights:
            self.adaptive_learner = AdaptiveWeightLearner(config)
        
        if config.enable_query_aware_fusion:
            self.query_aware_fusion = QueryAwareFusion(config)
        
        if config.enable_dynamic_temperature:
            self.temperature_controller = DynamicTemperatureController(config)
        
        if config.enable_fusion_quality_estimation:
            self.quality_estimator = FusionQualityEstimator(config)
        
        # 融合策略选择器
        self.strategy_selector = nn.Sequential(
            nn.Linear(config.query_projection_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # attention, rank, hybrid
            nn.Softmax(dim=-1)
        )
    
    def forward(self, 
                query_embedding: torch.Tensor,
                document_embeddings: torch.Tensor,
                source_names: List[str],
                document_ranks: Optional[torch.Tensor] = None,
                document_mask: Optional[torch.Tensor] = None,
                source_scores: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        增强融合前向传播
        
        Args:
            query_embedding: 查询嵌入
            document_embeddings: 文档嵌入
            source_names: 源名称列表
            document_ranks: 文档排名
            document_mask: 文档掩码
            source_scores: 各源原始分数
        """
        results = {}
        
        # 1. 基础注意力融合
        # 确保查询嵌入维度与文档嵌入维度匹配
        if query_embedding.size(-1) != document_embeddings.size(-1):
            if not hasattr(self, 'query_dimension_adapter'):
                self.query_dimension_adapter = nn.Linear(query_embedding.size(-1), document_embeddings.size(-1))
            query_embedding = self.query_dimension_adapter(query_embedding)
        
        attention_results = self.base_fusion(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
            source_names=source_names,
            document_ranks=document_ranks,
            document_mask=document_mask,
            return_attention=True
        )
        
        attention_scores = attention_results.get('fused_representation', attention_results.get('fusion_scores'))
        results.update(attention_results)
        
        # 2. 查询感知融合（如果启用）
        query_aware_scores = None
        if self.config.enable_query_aware_fusion and hasattr(self, 'query_aware_fusion'):
            if source_scores is not None:
                try:
                    query_aware_result = self.query_aware_fusion(
                        query_embedding, source_scores
                    )
                    # 检查返回值格式
                    if isinstance(query_aware_result, tuple) and len(query_aware_result) == 2:
                        query_aware_scores, query_info = query_aware_result
                        if isinstance(query_info, dict):
                            results.update(query_info)
                    elif isinstance(query_aware_result, dict):
                        query_aware_scores = query_aware_result.get('scores', query_aware_result.get('output'))
                        results.update(query_aware_result)
                    else:
                        query_aware_scores = query_aware_result
                except Exception as qa_error:
                    logger.warning(f"查询感知融合失败: {qa_error}")
                    query_aware_scores = None
        
        # 3. 基于排名的融合（如果启用）
        rank_scores = None
        if self.config.enable_rank_fusion and hasattr(self, 'rank_fusion') and source_scores is not None:
            # 学习自适应权重
            if self.config.enable_adaptive_weights and hasattr(self, 'adaptive_learner'):
                # 计算源特征
                source_features = torch.tensor([
                    scores.mean().item() for scores in source_scores
                ], device=query_embedding.device)
                
                adaptive_weights = self.adaptive_learner(
                    query_embedding.mean(dim=0),  # 假设query_embedding是[seq_len, dim]
                    source_features
                )
                
                rank_scores = self.rank_fusion(source_scores, adaptive_weights)
                results['adaptive_weights'] = adaptive_weights
            else:
                rank_scores = self.rank_fusion(source_scores)
        
        # 4. 融合策略选择
        query_features = query_embedding.mean(dim=0) if query_embedding.dim() > 1 else query_embedding
        # 确保特征维度与策略选择器匹配
        if query_features.size(-1) != self.config.query_projection_dim:
            if not hasattr(self, 'query_adapter'):
                self.query_adapter = nn.Linear(query_features.size(-1), self.config.query_projection_dim)
            query_features = self.query_adapter(query_features)
        strategy_weights = self.strategy_selector(query_features)
        
        # 5. 最终融合
        final_scores = torch.zeros_like(attention_scores)
        
        # 注意力融合
        final_scores += strategy_weights[0] * attention_scores
        
        # 排名融合
        if rank_scores is not None:
            final_scores += strategy_weights[1] * rank_scores
        
        # 查询感知融合
        if query_aware_scores is not None:
            final_scores += strategy_weights[2] * query_aware_scores
        
        # 6. 动态温度调整
        if self.config.enable_dynamic_temperature and hasattr(self, 'temperature_controller'):
            fusion_confidence = torch.max(F.softmax(final_scores, dim=0))
            temperature = self.temperature_controller(query_features, fusion_confidence)
            final_scores = final_scores / temperature
            results['dynamic_temperature'] = temperature
        
        # 7. 融合质量估计
        if self.config.enable_fusion_quality_estimation and hasattr(self, 'quality_estimator') and source_scores is not None:
            quality_score = self.quality_estimator(source_scores, final_scores)
            results['fusion_quality'] = quality_score
        
        results.update({
            'final_fusion_scores': final_scores,
            'strategy_weights': strategy_weights,
            'fusion_method': 'enhanced_fusion'
        })
        
        return results

def create_enhanced_fusion(config: Optional[EnhancedFusionConfig] = None) -> EnhancedFusion:
    """创建增强融合模块"""
    if config is None:
        config = EnhancedFusionConfig()
    return EnhancedFusion(config)

def create_enhanced_fusion_config(**kwargs) -> EnhancedFusionConfig:
    """创建增强融合配置"""
    return EnhancedFusionConfig(**kwargs) 