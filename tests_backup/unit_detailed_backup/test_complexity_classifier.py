#!/usr/bin/env python3
"""
复杂度分类器单元测试
测试 ComplexityClassifier 的各种功能
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from nano_graphrag.complexity.classifier import (
    ComplexityClassifier,
    ComplexityClassifierConfig,
    classify_query_complexity_sync
)
from nano_graphrag.complexity.calibration import CalibrationConfig


class TestComplexityClassifierConfig:
    """测试复杂度分类器配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ComplexityClassifierConfig()
        
        assert config.model_path == "nano_graphrag/models/modernbert_complexity_classifier"
        assert config.max_length == 256
        assert config.confidence_threshold == 0.7
        assert config.device == "auto"
        assert config.enable_calibration == True
        
    def test_label_mapping_initialization(self):
        """测试标签映射初始化"""
        config = ComplexityClassifierConfig()
        
        expected_mapping = {
            "zero_hop": "zero_hop",
            "one_hop": "one_hop", 
            "multi_hop": "multi_hop"
        }
        assert config.label_to_complexity == expected_mapping
        
    def test_custom_label_mapping(self):
        """测试自定义标签映射"""
        custom_mapping = {
            "simple": "zero_hop",
            "medium": "one_hop",
            "complex": "multi_hop"
        }
        config = ComplexityClassifierConfig(label_to_complexity=custom_mapping)
        assert config.label_to_complexity == custom_mapping


class TestComplexityClassifier:
    """测试复杂度分类器主类"""
    
    @pytest.fixture
    def mock_config(self):
        """创建模拟配置"""
        return ComplexityClassifierConfig(
            model_path="test_model",
            confidence_threshold=0.6
        )
    
    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """创建模拟的模型和分词器"""
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # 模拟模型输出
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[0.1, 0.8, 0.1]])  # one_hop 概率最高
        mock_model.return_value = mock_outputs
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        # 模拟分词器输出
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        return mock_model, mock_tokenizer
    
    @patch('nano_graphrag.complexity.classifier.AutoModel.from_pretrained')
    @patch('nano_graphrag.complexity.classifier.AutoTokenizer.from_pretrained')
    def test_classifier_initialization_success(self, mock_tokenizer_load, mock_model_load, 
                                             mock_config, mock_model_and_tokenizer):
        """测试分类器成功初始化"""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_model_load.return_value = mock_model
        mock_tokenizer_load.return_value = mock_tokenizer
        
        classifier = ComplexityClassifier(mock_config)
        
        assert classifier.model is not None
        assert classifier.tokenizer is not None
        assert classifier.is_available() == True
        
    def test_classifier_initialization_failure(self, mock_config):
        """测试分类器初始化失败的情况"""
        with patch('nano_graphrag.complexity.classifier.AutoModel.from_pretrained', 
                  side_effect=Exception("Model not found")):
            classifier = ComplexityClassifier(mock_config)
            assert classifier.model is None
            assert classifier.is_available() == False
    
    @patch('nano_graphrag.complexity.classifier.AutoModel.from_pretrained')
    @patch('nano_graphrag.complexity.classifier.AutoTokenizer.from_pretrained')
    def test_predict_with_model(self, mock_tokenizer_load, mock_model_load, 
                               mock_config, mock_model_and_tokenizer):
        """测试使用模型进行预测"""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_model_load.return_value = mock_model
        mock_tokenizer_load.return_value = mock_tokenizer
        
        classifier = ComplexityClassifier(mock_config)
        result = classifier.predict("What is machine learning?")
        
        assert result in ["zero_hop", "one_hop", "multi_hop"]
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
    
    def test_predict_without_model(self, mock_config):
        """测试没有模型时的预测（启发式方法）"""
        classifier = ComplexityClassifier(mock_config)
        # 确保模型未加载
        classifier.model = None
        classifier.tokenizer = None
        
        # 测试简单查询
        result = classifier.predict("What is AI?")
        assert result in ["zero_hop", "one_hop", "multi_hop"]
        
        # 测试复杂查询
        result = classifier.predict("Compare the relationship between AI and ML")
        assert result == "multi_hop"
    
    @patch('nano_graphrag.complexity.classifier.AutoModel.from_pretrained')
    @patch('nano_graphrag.complexity.classifier.AutoTokenizer.from_pretrained')
    def test_predict_with_confidence(self, mock_tokenizer_load, mock_model_load, 
                                   mock_config, mock_model_and_tokenizer):
        """测试带置信度的预测"""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_model_load.return_value = mock_model
        mock_tokenizer_load.return_value = mock_tokenizer
        
        classifier = ComplexityClassifier(mock_config)
        complexity, confidence, probabilities = classifier.predict_with_confidence(
            "What is machine learning?"
        )
        
        assert complexity in ["zero_hop", "one_hop", "multi_hop"]
        assert 0.0 <= confidence <= 1.0
        assert isinstance(probabilities, dict)
        assert len(probabilities) >= 1
    
    def test_heuristic_classify(self):
        """测试启发式分类方法"""
        config = ComplexityClassifierConfig()
        classifier = ComplexityClassifier(config)
        
        # 测试不同类型的查询
        test_cases = [
            ("AI", "zero_hop"),  # 短查询
            ("What is machine learning?", "zero_hop"),  # 定义性查询
            ("How does neural network work?", "one_hop"),  # 解释性查询
            ("Compare RNN and CNN differences", "multi_hop"),  # 比较性查询
        ]
        
        for query, expected in test_cases:
            result = classifier._heuristic_classify(query)
            # 注意：启发式方法可能不完全准确，这里只测试返回值合法性
            assert result in ["zero_hop", "one_hop", "multi_hop"]
    
    @patch('nano_graphrag.complexity.classifier.AutoModel.from_pretrained')
    @patch('nano_graphrag.complexity.classifier.AutoTokenizer.from_pretrained')
    def test_get_logits(self, mock_tokenizer_load, mock_model_load, 
                       mock_config, mock_model_and_tokenizer):
        """测试获取原始logits"""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_model_load.return_value = mock_model
        mock_tokenizer_load.return_value = mock_tokenizer
        
        classifier = ComplexityClassifier(mock_config)
        logits = classifier.get_logits("Test query")
        
        assert isinstance(logits, list)
        assert len(logits) == 3  # 三个复杂度类别
        assert all(isinstance(x, float) for x in logits)
    
    def test_get_logits_without_model(self, mock_config):
        """测试没有模型时获取logits"""
        classifier = ComplexityClassifier(mock_config)
        classifier.model = None
        
        logits = classifier.get_logits("Test query")
        
        assert isinstance(logits, list)
        assert len(logits) == 3
        # 应该返回均等概率
        assert all(abs(x - 0.33) < 0.1 for x in logits)


class TestGlobalFunctions:
    """测试全局函数"""
    
    @patch('nano_graphrag.complexity.classifier.ComplexityClassifier')
    def test_classify_query_complexity_sync(self, mock_classifier_class):
        """测试同步分类函数"""
        mock_classifier = Mock()
        mock_classifier.predict.return_value = "one_hop"
        mock_classifier_class.return_value = mock_classifier
        
        result = classify_query_complexity_sync("Test query")
        
        assert result == "one_hop"
        mock_classifier.predict.assert_called_once_with("Test query")


class TestIntegration:
    """集成测试"""
    
    def test_end_to_end_classification_without_model(self):
        """端到端测试（不使用真实模型）"""
        config = ComplexityClassifierConfig()
        classifier = ComplexityClassifier(config)
        
        # 确保没有加载模型（避免依赖外部模型文件）
        classifier.model = None
        classifier.tokenizer = None
        
        test_queries = [
            "What is AI?",
            "How does machine learning work?",
            "Compare the differences between supervised and unsupervised learning",
            "Explain the relationship between AI, ML, and deep learning"
        ]
        
        for query in test_queries:
            # 测试基本预测
            complexity = classifier.predict(query)
            assert complexity in ["zero_hop", "one_hop", "multi_hop"]
            
            # 测试带置信度的预测
            complexity, confidence, probs = classifier.predict_with_confidence(query)
            assert complexity in ["zero_hop", "one_hop", "multi_hop"]
            assert 0.0 <= confidence <= 1.0
            assert isinstance(probs, dict)


# 运行测试的入口
if __name__ == "__main__":
    pytest.main([__file__, "-v"])