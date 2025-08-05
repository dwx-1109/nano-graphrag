#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小化单元测试套件 - 针对 nano_graphrag 项目
仅测试关键组件的核心功能，不依赖外部API
基于Agent测试金字塔理论的底层测试
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入核心组件
from nano_graphrag.complexity.classifier import ComplexityClassifier, ComplexityClassifierConfig
from nano_graphrag.complexity.calibration import ConfidenceCalibrator, CalibrationConfig
from nano_graphrag.cache.lru_cache import LRUCache
from nano_graphrag.answer_generation.generator import AnswerGeneratorConfig
from nano_graphrag.answer_generation.prompts import BasicPromptTemplate
from nano_graphrag.feedback.feedback import FeedbackProcessor
from nano_graphrag.deduplication.deduplication import DocumentDeduplicator

class TestMinimalUnitSuite:
    """最小化单元测试套件"""
    
    def test_complexity_classifier_config(self):
        """测试复杂度分类器配置"""
        config = ComplexityClassifierConfig()
        assert config.confidence_threshold == 0.7
        assert config.device in ["cuda", "cpu", "auto"]
        assert "zero_hop" in config.label_to_complexity
    
    def test_complexity_classifier_fallback(self):
        """测试复杂度分类器回退机制（不依赖模型文件）"""
        config = ComplexityClassifierConfig(model_path="non_existent_path")
        classifier = ComplexityClassifier(config)
        
        # 测试启发式分类
        result = classifier._heuristic_classify("What is AI?")
        assert result in ["zero_hop", "one_hop", "multi_hop"]
    
    def test_confidence_calibrator_basic(self):
        """测试置信度校准器基础功能"""
        config = CalibrationConfig(method="platt_scaling")
        calibrator = ConfidenceCalibrator(config)
        
        # 测试基本配置
        assert calibrator.config.method == "platt_scaling"
        assert calibrator.calibrator is not None
    
    def test_lru_cache_basic_operations(self):
        """测试LRU缓存基本操作"""
        cache = LRUCache(max_size=3)
        
        # 基本存取
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.contains("key1")
        
        # LRU淘汰
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # 应该淘汰key1
        
        assert not cache.contains("key1")
        assert cache.contains("key4")
    
    def test_answer_generator_config_validation(self):
        """测试答案生成器配置验证"""
        config = AnswerGeneratorConfig()
        
        # 默认配置验证
        assert config.model_name == "qwen-turbo"
        assert config.temperature >= 0 and config.temperature <= 2
        assert config.max_tokens > 0
        assert config.max_retries >= 0
    
    def test_prompt_template_basic(self):
        """测试基础提示模板"""
        template = BasicPromptTemplate()
        
        query = "测试查询"
        context = "测试上下文"
        
        # 测试基本生成
        result = template.generate(query, context)
        assert isinstance(result, (str, dict))
        
        if isinstance(result, str):
            assert query in result or context in result
    
    def test_feedback_processor_initialization(self):
        """测试反馈处理器初始化"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = FeedbackProcessor(
                feedback_file=os.path.join(temp_dir, "test_feedback.json")
            )
            
            # 测试基本功能
            assert processor.feedback_data == {}
            assert processor.similarity_threshold == 0.7
    
    def test_document_deduplicator_basic(self):
        """测试文档去重器基础功能"""
        deduplicator = DocumentDeduplicator(similarity_threshold=0.8)
        
        # 测试配置
        assert deduplicator.similarity_threshold == 0.8
        assert deduplicator.method in ["greedy", "clustering"]
        
        # 测试基本相似度计算
        doc1 = {"content": "这是一个测试文档"}
        doc2 = {"content": "这是另一个测试文档"}
        
        similarity = deduplicator.calculate_similarity(doc1, doc2)
        assert 0 <= similarity <= 1
    
    def test_error_handling_robustness(self):
        """测试错误处理鲁棒性"""
        # 测试空输入处理
        template = BasicPromptTemplate()
        
        # 空查询
        result = template.generate("", "context")
        assert isinstance(result, (str, dict))
        
        # 空上下文
        result = template.generate("query", "")
        assert isinstance(result, (str, dict))
    
    def test_numpy_array_handling(self):
        """测试numpy数组处理"""
        # 测试缓存中的numpy数组
        cache = LRUCache(max_size=10)
        
        test_array = np.array([1, 2, 3, 4, 5])
        cache.put("test_array", test_array)
        
        retrieved = cache.get("test_array")
        assert np.array_equal(retrieved, test_array)

class TestEdgeCases:
    """边界情况测试"""
    
    def test_large_input_handling(self):
        """测试大输入处理"""
        template = BasicPromptTemplate()
        
        # 大查询
        large_query = "测试" * 1000
        large_context = "上下文" * 1000
        
        # 应该不抛出异常
        result = template.generate(large_query, large_context)
        assert isinstance(result, (str, dict))
    
    def test_unicode_handling(self):
        """测试Unicode处理"""
        template = BasicPromptTemplate()
        
        unicode_query = "测试查询🚀🔍📊"
        unicode_context = "测试上下文💡🎯✨"
        
        result = template.generate(unicode_query, unicode_context)
        assert isinstance(result, (str, dict))
    
    def test_memory_cleanup(self):
        """测试内存清理"""
        cache = LRUCache(max_size=5)
        
        # 填满缓存
        for i in range(10):
            cache.put(f"key_{i}", f"value_{i}")
        
        # 验证缓存大小限制
        stats = cache.get_stats()
        current_size = stats.get("current_size", stats.get("size", 0))
        assert current_size <= 5
        
        # 清理缓存
        cache.clear()
        stats_after_clear = cache.get_stats()
        current_size_after = stats_after_clear.get("current_size", stats_after_clear.get("size", 0))
        assert current_size_after == 0

# 快速运行函数
def run_minimal_tests():
    """运行最小化测试套件"""
    print("🧪 运行最小化单元测试套件...")
    
    # 使用pytest运行
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x"  # 遇到第一个失败就停止
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("✅ 所有最小化单元测试通过！")
    else:
        print("❌ 部分单元测试失败")
    
    return exit_code

if __name__ == "__main__":
    # 检查环境
    print("🔧 检查测试环境...")
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    
    # 运行测试
    exit_code = run_minimal_tests()
    sys.exit(exit_code)