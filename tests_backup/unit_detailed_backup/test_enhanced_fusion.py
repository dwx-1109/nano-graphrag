#!/usr/bin/env python3
"""
增强融合器单元测试
测试 EnhancedFusion 的各种功能
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional

from nano_graphrag.fusion_advanced.enhanced_fusion import (
    EnhancedFusion,
    EnhancedFusionConfig,
    RankBasedFusion,
    AdaptiveWeightLearner,
    QueryAwareFusion,
    DynamicTemperatureController,
    FusionQualityEstimator,
    create_enhanced_fusion,
    create_enhanced_fusion_config
)
from nano_graphrag.fusion_advanced.multihead_attention_fusion import MultiHeadAttentionFusionConfig


class TestEnhancedFusionConfig:
    """测试增强融合配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = EnhancedFusionConfig()
        
        assert config.enable_rank_fusion == True
        assert config.rank_fusion_method == "rrf"
        assert config.rrf_k == 60.0
        assert config.enable_adaptive_weights == True
        assert config.enable_query_aware_fusion == True
        assert config.enable_dynamic_temperature == True
        assert config.enable_fusion_quality_estimation == True
        
    def test_custom_config(self):
        """测试自定义配置"""
        config = EnhancedFusionConfig(
            enable_rank_fusion=False,
            rank_fusion_method="combsum",
            rrf_k=30.0,
            query_projection_dim=256
        )
        
        assert config.enable_rank_fusion == False
        assert config.rank_fusion_method == "combsum"
        assert config.rrf_k == 30.0
        assert config.query_projection_dim == 256


class TestRankBasedFusion:
    """测试基于排名的融合"""
    
    @pytest.fixture
    def config(self):
        return EnhancedFusionConfig(rank_fusion_method="rrf", rrf_k=60.0)
    
    @pytest.fixture
    def rank_fusion(self, config):
        return RankBasedFusion(config)
    
    @pytest.fixture
    def sample_scores(self):
        """创建样本分数"""
        return [
            torch.tensor([0.9, 0.7, 0.5, 0.3]),  # 源1分数
            torch.tensor([0.8, 0.6, 0.4, 0.2]),  # 源2分数
            torch.tensor([0.7, 0.8, 0.6, 0.4])   # 源3分数
        ]
    
    def test_rrf_fusion(self, rank_fusion, sample_scores):
        """测试RRF融合"""
        result = rank_fusion(sample_scores)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([4])  # 4个文档
        assert torch.all(result >= 0)  # 分数应该非负
        
    def test_combsum_fusion(self, config, sample_scores):
        """测试CombSUM融合"""
        config.rank_fusion_method = "combsum"
        rank_fusion = RankBasedFusion(config)
        
        result = rank_fusion(sample_scores)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([4])
        
    def test_combmnz_fusion(self, config, sample_scores):
        """测试CombMNZ融合"""
        config.rank_fusion_method = "combmnz"
        rank_fusion = RankBasedFusion(config)
        
        result = rank_fusion(sample_scores)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([4])
    
    def test_weighted_fusion(self, rank_fusion, sample_scores):
        """测试加权融合"""
        weights = torch.tensor([0.5, 0.3, 0.2])  # 源权重
        result = rank_fusion(sample_scores, weights)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([4])
    
    def test_invalid_method(self, config):
        """测试无效的融合方法"""
        config.rank_fusion_method = "invalid_method"
        rank_fusion = RankBasedFusion(config)
        
        sample_scores = [torch.tensor([0.9, 0.7]), torch.tensor([0.8, 0.6])]
        
        with pytest.raises(ValueError):
            rank_fusion(sample_scores)


class TestAdaptiveWeightLearner:
    """测试自适应权重学习器"""
    
    @pytest.fixture
    def config(self):
        return EnhancedFusionConfig(query_projection_dim=128)
    
    @pytest.fixture
    def weight_learner(self, config):
        return AdaptiveWeightLearner(config)
    
    def test_forward_pass(self, weight_learner):
        """测试前向传播"""
        batch_size, seq_len, hidden_dim = 2, 10, 128
        num_sources = 4
        
        query_embedding = torch.randn(batch_size, hidden_dim)
        source_features = torch.randn(batch_size, num_sources, hidden_dim)
        
        weights = weight_learner(query_embedding, source_features)
        
        assert weights.shape == (batch_size, num_sources)
        # 权重应该经过softmax，和为1
        assert torch.allclose(torch.sum(weights, dim=1), torch.ones(batch_size), atol=1e-6)
    
    def test_with_historical_performance(self, weight_learner):
        """测试包含历史性能的权重学习"""
        batch_size, hidden_dim = 2, 128
        num_sources = 4
        
        query_embedding = torch.randn(batch_size, hidden_dim)
        source_features = torch.randn(batch_size, num_sources, hidden_dim)
        historical_performance = torch.randn(batch_size, num_sources)
        
        weights = weight_learner(query_embedding, source_features, historical_performance)
        
        assert weights.shape == (batch_size, num_sources)
        assert torch.allclose(torch.sum(weights, dim=1), torch.ones(batch_size), atol=1e-6)


class TestQueryAwareFusion:
    """测试查询感知融合"""
    
    @pytest.fixture
    def config(self):
        return EnhancedFusionConfig(query_projection_dim=128)
    
    @pytest.fixture
    def query_fusion(self, config):
        return QueryAwareFusion(config)
    
    def test_forward_pass(self, query_fusion):
        """测试前向传播"""
        batch_size, hidden_dim = 1, 128
        num_docs = 10
        
        query_embedding = torch.randn(batch_size, hidden_dim)
        source_scores = [
            torch.randn(num_docs),
            torch.randn(num_docs),
            torch.randn(num_docs),
            torch.randn(num_docs)
        ]
        
        fused_scores, info = query_fusion(query_embedding, source_scores)
        
        assert fused_scores.shape == (num_docs,)
        assert 'query_type_probs' in info
        assert 'type_weights' in info
        assert 'query_features' in info
        
        # 检查查询类型概率
        query_type_probs = info['query_type_probs']
        assert query_type_probs.shape[1] == 3  # 3种查询类型
        assert torch.allclose(torch.sum(query_type_probs, dim=1), torch.ones(1), atol=1e-6)


class TestDynamicTemperatureController:
    """测试动态温度控制器"""
    
    @pytest.fixture
    def config(self):
        return EnhancedFusionConfig(
            temperature_min=0.1,
            temperature_max=2.0,
            query_projection_dim=128
        )
    
    @pytest.fixture
    def temp_controller(self, config):
        return DynamicTemperatureController(config)
    
    def test_temperature_range(self, temp_controller, config):
        """测试温度范围控制"""
        batch_size, hidden_dim = 2, 128
        
        query_features = torch.randn(batch_size, hidden_dim)
        fusion_confidence = torch.randn(batch_size, 1)
        
        temperature = temp_controller(query_features, fusion_confidence)
        
        assert temperature.shape == (batch_size, 1)
        assert torch.all(temperature >= config.temperature_min)
        assert torch.all(temperature <= config.temperature_max)


class TestFusionQualityEstimator:
    """测试融合质量估计器"""
    
    @pytest.fixture
    def config(self):
        return EnhancedFusionConfig()
    
    @pytest.fixture
    def quality_estimator(self, config):
        return FusionQualityEstimator(config)
    
    def test_quality_estimation(self, quality_estimator):
        """测试质量估计"""
        num_docs = 10
        
        source_scores = [
            torch.randn(num_docs),
            torch.randn(num_docs),
            torch.randn(num_docs)
        ]
        fused_scores = torch.randn(num_docs)
        
        quality_score = quality_estimator(source_scores, fused_scores)
        
        assert quality_score.shape == torch.Size([1])
        assert 0.0 <= quality_score.item() <= 1.0


class TestEnhancedFusion:
    """测试增强融合主类"""
    
    @pytest.fixture
    def config(self):
        base_config = MultiHeadAttentionFusionConfig(
            d_model=128,
            num_heads=4,
            num_sources=4
        )
        return EnhancedFusionConfig(base_config=base_config)
    
    @pytest.fixture
    def enhanced_fusion(self, config):
        return EnhancedFusion(config)
    
    def test_initialization(self, enhanced_fusion, config):
        """测试初始化"""
        assert enhanced_fusion.config == config
        assert hasattr(enhanced_fusion, 'base_fusion')
        
        if config.enable_rank_fusion:
            assert hasattr(enhanced_fusion, 'rank_fusion')
        if config.enable_adaptive_weights:
            assert hasattr(enhanced_fusion, 'adaptive_learner')
        if config.enable_query_aware_fusion:
            assert hasattr(enhanced_fusion, 'query_aware_fusion')
    
    def test_forward_pass(self, enhanced_fusion):
        """测试前向传播"""
        batch_size, seq_len, hidden_dim = 1, 10, 128
        num_sources = 4
        
        query_embedding = torch.randn(batch_size, hidden_dim)
        document_embeddings = torch.randn(batch_size, seq_len, hidden_dim)
        source_names = ["naive", "bm25", "local", "global"]
        
        result = enhanced_fusion(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
            source_names=source_names
        )
        
        assert isinstance(result, dict)
        assert 'fused_embedding' in result
        assert 'attention_weights' in result
        
        fused_embedding = result['fused_embedding']
        assert fused_embedding.shape == (batch_size, hidden_dim)
    
    def test_with_document_ranks(self, enhanced_fusion):
        """测试包含文档排名的融合"""
        batch_size, seq_len, hidden_dim = 1, 10, 128
        
        query_embedding = torch.randn(batch_size, hidden_dim)
        document_embeddings = torch.randn(batch_size, seq_len, hidden_dim)
        source_names = ["naive", "bm25", "local", "global"]
        document_ranks = torch.randint(1, 101, (batch_size, seq_len))
        
        result = enhanced_fusion(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
            source_names=source_names,
            document_ranks=document_ranks
        )
        
        assert isinstance(result, dict)
        assert 'fused_embedding' in result
    
    def test_with_source_scores(self, enhanced_fusion):
        """测试包含源分数的融合"""
        batch_size, seq_len, hidden_dim = 1, 10, 128
        
        query_embedding = torch.randn(batch_size, hidden_dim)
        document_embeddings = torch.randn(batch_size, seq_len, hidden_dim)
        source_names = ["naive", "bm25", "local", "global"]
        source_scores = [torch.randn(seq_len) for _ in range(4)]
        
        result = enhanced_fusion(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
            source_names=source_names,
            source_scores=source_scores
        )
        
        assert isinstance(result, dict)
        assert 'fused_embedding' in result
        
        # 如果启用了融合质量估计，应该有质量分数
        if enhanced_fusion.config.enable_fusion_quality_estimation:
            assert 'fusion_quality' in result


class TestFactoryFunctions:
    """测试工厂函数"""
    
    def test_create_enhanced_fusion_default(self):
        """测试创建默认增强融合器"""
        fusion = create_enhanced_fusion()
        
        assert isinstance(fusion, EnhancedFusion)
        assert isinstance(fusion.config, EnhancedFusionConfig)
    
    def test_create_enhanced_fusion_custom_config(self):
        """测试使用自定义配置创建增强融合器"""
        config = EnhancedFusionConfig(enable_rank_fusion=False)
        fusion = create_enhanced_fusion(config)
        
        assert isinstance(fusion, EnhancedFusion)
        assert fusion.config.enable_rank_fusion == False
    
    def test_create_enhanced_fusion_config(self):
        """测试创建增强融合配置"""
        config = create_enhanced_fusion_config(
            enable_rank_fusion=False,
            rrf_k=30.0
        )
        
        assert isinstance(config, EnhancedFusionConfig)
        assert config.enable_rank_fusion == False
        assert config.rrf_k == 30.0


class TestIntegration:
    """集成测试"""
    
    def test_end_to_end_fusion(self):
        """端到端融合测试"""
        # 创建配置
        base_config = MultiHeadAttentionFusionConfig(
            d_model=64,  # 使用较小的维度以加快测试
            num_heads=2,
            num_sources=3
        )
        config = EnhancedFusionConfig(base_config=base_config)
        
        # 创建融合器
        fusion = EnhancedFusion(config)
        fusion.eval()  # 设置为评估模式
        
        # 准备输入数据
        batch_size, seq_len, hidden_dim = 1, 5, 64
        
        query_embedding = torch.randn(batch_size, hidden_dim)
        document_embeddings = torch.randn(batch_size, seq_len, hidden_dim)
        source_names = ["naive", "bm25", "local"]
        document_ranks = torch.randint(1, 11, (batch_size, seq_len))
        source_scores = [torch.randn(seq_len) for _ in range(3)]
        
        # 执行融合
        with torch.no_grad():
            result = fusion(
                query_embedding=query_embedding,
                document_embeddings=document_embeddings,
                source_names=source_names,
                document_ranks=document_ranks,
                source_scores=source_scores
            )
        
        # 验证结果
        assert isinstance(result, dict)
        assert 'fused_embedding' in result
        assert 'attention_weights' in result
        
        fused_embedding = result['fused_embedding']
        assert fused_embedding.shape == (batch_size, hidden_dim)
        assert not torch.isnan(fused_embedding).any()
        assert not torch.isinf(fused_embedding).any()


# 运行测试的入口
if __name__ == "__main__":
    pytest.main([__file__, "-v"])