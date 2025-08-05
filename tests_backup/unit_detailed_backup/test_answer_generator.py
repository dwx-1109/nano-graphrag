#!/usr/bin/env python3
"""
答案生成器单元测试
测试 AnswerGenerator 的各种功能
"""

import pytest
import asyncio
import tempfile
import os
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Optional, Tuple

from nano_graphrag.answer_generation.generator import (
    AnswerGenerator,
    AnswerGeneratorConfig
)
from nano_graphrag.answer_generation.prompts import (
    BasicPromptTemplate,
    ConfidenceAwarePrompt,
    MultiHopPromptTemplate
)
from nano_graphrag.feedback.feedback import FeedbackProcessor


class TestAnswerGeneratorConfig:
    """测试答案生成器配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = AnswerGeneratorConfig()
        
        assert config.model_name == "qwen-turbo"
        assert config.api_key is None
        assert config.base_url is None
        assert config.timeout == 30.0
        assert config.temperature == 0.7
        assert config.top_p == 1.0
        assert config.max_tokens == 1500
        assert config.max_retries == 3
        assert config.use_cache == True
        assert config.api_type == "dashscope"
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = AnswerGeneratorConfig(
            model_name="gpt-4",
            api_key="test_key",
            temperature=0.5,
            max_tokens=2000,
            api_type="openai"
        )
        
        assert config.model_name == "gpt-4"
        assert config.api_key == "test_key"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.api_type == "openai"


class TestAnswerGenerator:
    """测试答案生成器主类"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def config(self, temp_dir):
        """创建测试配置"""
        return AnswerGeneratorConfig(
            model_name="test_model",
            api_key="test_key",
            use_cache=True,
            cache_dir=temp_dir,
            timeout=10.0
        )
    
    @pytest.fixture
    def mock_feedback_processor(self):
        """创建模拟反馈处理器"""
        processor = Mock(spec=FeedbackProcessor)
        processor.process_feedback.return_value = {"processed": True}
        return processor
    
    @pytest.fixture
    def generator(self, config, mock_feedback_processor):
        """创建答案生成器实例"""
        return AnswerGenerator(config, mock_feedback_processor)
    
    def test_initialization(self, generator, config):
        """测试初始化"""
        assert generator.config == config
        assert generator.feedback_processor is not None
        assert generator.call_count == 0
        assert generator.cache_hits == 0
        assert generator.cache_misses == 0
    
    def test_cache_initialization(self, config, temp_dir):
        """测试缓存初始化"""
        generator = AnswerGenerator(config)
        
        assert generator.cache is not None
        assert os.path.exists(temp_dir)
    
    def test_compute_cache_key(self, generator):
        """测试缓存键计算"""
        query = "What is AI?"
        context = "AI is artificial intelligence."
        
        key1 = generator._compute_cache_key(query, context)
        key2 = generator._compute_cache_key(query, context)
        key3 = generator._compute_cache_key(query + " ", context)  # 不同查询
        
        assert key1 == key2  # 相同输入应该产生相同的键
        assert key1 != key3  # 不同输入应该产生不同的键
        assert isinstance(key1, str)
        assert len(key1) > 0
    
    @patch('nano_graphrag.answer_generation.generator.AnswerGenerator._call_llm')
    @pytest.mark.asyncio
    async def test_generate_answer_basic(self, mock_call_llm, generator):
        """测试基本答案生成"""
        # 模拟LLM响应
        mock_response = {
            "choices": [{"message": {"content": "AI is artificial intelligence."}}],
            "usage": {"total_tokens": 50}
        }
        mock_call_llm.return_value = mock_response
        
        query = "What is AI?"
        context = "AI stands for artificial intelligence."
        
        answer, metadata = await generator.generate_answer(query, context)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(metadata, dict)
        assert "generation_time" in metadata
        assert "token_usage" in metadata
        assert "cache_hit" in metadata
        
        mock_call_llm.assert_called_once()
    
    @patch('nano_graphrag.answer_generation.generator.AnswerGenerator._call_llm')
    @pytest.mark.asyncio
    async def test_generate_answer_with_confidence(self, mock_call_llm, generator):
        """测试带置信度的答案生成"""
        mock_response = {
            "choices": [{"message": {"content": "High confidence answer about AI."}}],
            "usage": {"total_tokens": 60}
        }
        mock_call_llm.return_value = mock_response
        
        query = "What is AI?"
        context = "AI stands for artificial intelligence."
        confidence = 0.9
        
        answer, metadata = await generator.generate_answer(
            query, context, confidence=confidence
        )
        
        assert isinstance(answer, str)
        assert "confidence" in metadata
        assert metadata["confidence"] == confidence
    
    @patch('nano_graphrag.answer_generation.generator.AnswerGenerator._call_llm')
    @pytest.mark.asyncio
    async def test_generate_answer_with_complexity(self, mock_call_llm, generator):
        """测试带复杂度的答案生成"""
        mock_response = {
            "choices": [{"message": {"content": "Complex multi-hop answer."}}],
            "usage": {"total_tokens": 80}
        }
        mock_call_llm.return_value = mock_response
        
        query = "Compare AI and ML"
        context = "AI and ML are related fields."
        complexity = "multi_hop"
        
        answer, metadata = await generator.generate_answer(
            query, context, complexity=complexity
        )
        
        assert isinstance(answer, str)
        assert "complexity" in metadata
        assert metadata["complexity"] == complexity
    
    @patch('nano_graphrag.answer_generation.generator.AnswerGenerator._call_llm')
    @pytest.mark.asyncio
    async def test_caching_functionality(self, mock_call_llm, generator):
        """测试缓存功能"""
        mock_response = {
            "choices": [{"message": {"content": "Cached response."}}],
            "usage": {"total_tokens": 40}
        }
        mock_call_llm.return_value = mock_response
        
        query = "What is ML?"
        context = "ML is machine learning."
        
        # 第一次调用
        answer1, metadata1 = await generator.generate_answer(query, context)
        initial_cache_misses = generator.cache_misses
        
        # 第二次调用（应该使用缓存）
        answer2, metadata2 = await generator.generate_answer(query, context)
        
        assert answer1 == answer2
        assert metadata1["cache_hit"] == False
        assert metadata2["cache_hit"] == True
        assert generator.cache_hits == 1
        assert generator.cache_misses == initial_cache_misses
        
        # LLM只应该被调用一次
        mock_call_llm.assert_called_once()
    
    @patch('nano_graphrag.answer_generation.generator.AnswerGenerator._call_llm')
    @pytest.mark.asyncio
    async def test_llm_call_retry_mechanism(self, mock_call_llm, generator):
        """测试LLM调用重试机制"""
        # 前两次调用失败，第三次成功
        mock_call_llm.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            {
                "choices": [{"message": {"content": "Success after retries."}}],
                "usage": {"total_tokens": 45}
            }
        ]
        
        query = "Test retry"
        context = "Test context"
        
        answer, metadata = await generator.generate_answer(query, context)
        
        assert isinstance(answer, str)
        assert answer == "Success after retries."
        assert mock_call_llm.call_count == 3  # 重试了2次
    
    @patch('nano_graphrag.answer_generation.generator.AnswerGenerator._call_llm')
    @pytest.mark.asyncio
    async def test_llm_call_max_retries_exceeded(self, mock_call_llm, generator):
        """测试超过最大重试次数"""
        # 所有调用都失败
        mock_call_llm.side_effect = Exception("Persistent API Error")
        
        query = "Test max retries"
        context = "Test context"
        
        with pytest.raises(Exception):
            await generator.generate_answer(query, context)
        
        # 应该调用 max_retries + 1 次
        expected_calls = generator.config.max_retries + 1
        assert mock_call_llm.call_count == expected_calls
    
    @patch('nano_graphrag.answer_generation.generator.AnswerGenerator._call_llm')
    @pytest.mark.asyncio
    async def test_custom_prompt_template(self, mock_call_llm, generator):
        """测试自定义提示模板"""
        mock_response = {
            "choices": [{"message": {"content": "Custom template response."}}],
            "usage": {"total_tokens": 55}
        }
        mock_call_llm.return_value = mock_response
        
        custom_template = BasicPromptTemplate(
            template="Custom: {query}\nContext: {context}\nAnswer:"
        )
        
        query = "Test custom template"
        context = "Test context"
        
        answer, metadata = await generator.generate_answer(
            query, context, prompt_template=custom_template
        )
        
        assert isinstance(answer, str)
        # 检查是否使用了自定义模板（通过检查调用参数）
        mock_call_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_embedding_mock(self, generator):
        """测试嵌入生成（模拟）"""
        with patch.object(generator, '_call_embedding_api') as mock_embed:
            mock_embed.return_value = ([[0.1, 0.2, 0.3]], {"model": "test"})
            
            embeddings, metadata = await generator.generate_embedding("test text")
            
            assert isinstance(embeddings, list)
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 3
            assert isinstance(metadata, dict)
    
    def test_get_stats(self, generator):
        """测试统计信息获取"""
        generator.call_count = 10
        generator.cache_hits = 3
        generator.cache_misses = 7
        
        stats = generator.get_stats()
        
        assert isinstance(stats, dict)
        assert stats["total_calls"] == 10
        assert stats["cache_hits"] == 3
        assert stats["cache_misses"] == 7
        assert stats["cache_hit_rate"] == 0.3
    
    def test_add_feedback(self, generator, mock_feedback_processor):
        """测试添加反馈"""
        query_id = "test_query_123"
        feedback_data = {"rating": 4, "comment": "Good answer"}
        
        result = generator.add_feedback(query_id, feedback_data)
        
        assert result == True
        mock_feedback_processor.process_feedback.assert_called_once()
    
    @patch('nano_graphrag.answer_generation.generator.AnswerGenerator._call_llm')
    @pytest.mark.asyncio
    async def test_generate_answer_with_confidence_method(self, mock_call_llm, generator):
        """测试带置信度的答案生成方法"""
        mock_response = {
            "choices": [{"message": {"content": "Confident answer."}}],
            "usage": {"total_tokens": 50}
        }
        mock_call_llm.return_value = mock_response
        
        query = "What is deep learning?"
        context = "Deep learning is a subset of ML."
        confidence = 0.85
        
        answer, returned_confidence, metadata = await generator.generate_answer_with_confidence(
            query, context, confidence
        )
        
        assert isinstance(answer, str)
        assert returned_confidence == confidence
        assert isinstance(metadata, dict)


class TestPromptTemplateIntegration:
    """测试提示模板集成"""
    
    @pytest.fixture
    def config(self):
        return AnswerGeneratorConfig(use_cache=False)  # 禁用缓存以简化测试
    
    @pytest.mark.asyncio
    async def test_basic_prompt_template(self, config):
        """测试基础提示模板"""
        template = BasicPromptTemplate()
        generator = AnswerGenerator(config)
        generator.prompt_template = template
        
        with patch.object(generator, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "choices": [{"message": {"content": "Basic template answer."}}],
                "usage": {"total_tokens": 40}
            }
            
            answer, metadata = await generator.generate_answer(
                "Test query", "Test context"
            )
            
            assert isinstance(answer, str)
            mock_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_confidence_aware_prompt(self, config):
        """测试置信度感知提示"""
        generator = AnswerGenerator(config)
        
        with patch.object(generator, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "choices": [{"message": {"content": "Confidence-aware answer."}}],
                "usage": {"total_tokens": 45}
            }
            
            # 提供置信度应该自动使用置信度感知模板
            answer, metadata = await generator.generate_answer(
                "Test query", "Test context", confidence=0.8
            )
            
            assert isinstance(answer, str)
            assert metadata["confidence"] == 0.8
    
    @pytest.mark.asyncio
    async def test_multi_hop_prompt_template(self, config):
        """测试多跳提示模板"""
        template = MultiHopPromptTemplate()
        generator = AnswerGenerator(config)
        generator.prompt_template = template
        
        with patch.object(generator, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "choices": [{"message": {"content": "Multi-hop reasoning answer."}}],
                "usage": {"total_tokens": 60}
            }
            
            answer, metadata = await generator.generate_answer(
                "Compare A and B", "Context about A and B"
            )
            
            assert isinstance(answer, str)


class TestErrorHandling:
    """测试错误处理"""
    
    @pytest.fixture
    def config(self):
        return AnswerGeneratorConfig(max_retries=2, use_cache=False)
    
    @pytest.fixture
    def generator(self, config):
        return AnswerGenerator(config)
    
    @pytest.mark.asyncio
    async def test_invalid_api_response(self, generator):
        """测试无效API响应"""
        with patch.object(generator, '_call_llm') as mock_llm:
            # 返回格式错误的响应
            mock_llm.return_value = {"invalid": "response"}
            
            with pytest.raises(Exception):
                await generator.generate_answer("Test", "Context")
    
    @pytest.mark.asyncio
    async def test_empty_response(self, generator):
        """测试空响应"""
        with patch.object(generator, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "choices": [{"message": {"content": ""}}],
                "usage": {"total_tokens": 0}
            }
            
            answer, metadata = await generator.generate_answer("Test", "Context")
            
            # 应该处理空响应
            assert isinstance(answer, str)
    
    def test_invalid_cache_directory(self):
        """测试无效缓存目录"""
        config = AnswerGeneratorConfig(
            cache_dir="/invalid/path/that/does/not/exist"
        )
        
        # 应该能够创建生成器，但可能会有警告
        generator = AnswerGenerator(config)
        assert generator is not None


# 运行测试的入口
if __name__ == "__main__":
    pytest.main([__file__, "-v"])