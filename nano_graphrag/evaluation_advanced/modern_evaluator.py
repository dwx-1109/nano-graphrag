"""
现代化评估器
基于最新的RAG评估方法：RAGAS, ARES, TruLens等
适配本科生项目的实际需求
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

@dataclass
class ModernEvaluatorConfig:
    """现代评估器配置"""
    
    # 基础配置
    embedding_model: str = "all-MiniLM-L6-v2"  # 轻量级嵌入模型
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # RAGAS相关指标
    enable_ragas_metrics: bool = True
    faithfulness_threshold: float = 0.7
    answer_relevancy_threshold: float = 0.6
    context_precision_threshold: float = 0.8
    
    # 语义相似度阈值
    semantic_similarity_threshold: float = 0.75
    
    # 复杂度感知评估
    enable_complexity_aware_eval: bool = True
    complexity_weights: Dict[str, float] = field(default_factory=lambda: {
        "zero_hop": 1.0,
        "one_hop": 1.2, 
        "multi_hop": 1.5
    })
    
    # 支持的评估指标列表
    metrics: List[str] = field(default_factory=lambda: [
        "faithfulness", 
        "answer_relevancy", 
        "context_precision", 
        "semantic_similarity",
        "complexity_score"
    ])

class RAGASEvaluator:
    """基于RAGAS的评估器（简化版本）"""
    
    def __init__(self, config: ModernEvaluatorConfig):
        self.config = config
        self.embedder = SentenceTransformer(config.embedding_model)
        if torch.cuda.is_available():
            self.embedder = self.embedder.to(config.device)
    
    def faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        忠实度评估：答案是否基于提供的上下文
        简化版本：检查答案中的关键信息是否来源于上下文
        """
        if not answer or not contexts:
            return 0.0
        
        # 提取答案中的关键句子
        answer_sentences = self._split_sentences(answer)
        
        # 计算每个句子与上下文的相似度
        context_text = " ".join(contexts)
        context_embedding = self.embedder.encode([context_text])
        
        supported_count = 0
        total_count = len(answer_sentences)
        
        if total_count == 0:
            return 0.0
        
        for sentence in answer_sentences:
            sentence_embedding = self.embedder.encode([sentence])
            similarity = cosine_similarity(sentence_embedding, context_embedding)[0][0]
            
            if similarity > self.config.faithfulness_threshold:
                supported_count += 1
        
        return supported_count / total_count
    
    def answer_relevancy(self, question: str, answer: str) -> float:
        """
        答案相关性：答案对问题的相关程度
        """
        if not question or not answer:
            return 0.0
        
        # 计算问题和答案的语义相似度
        question_embedding = self.embedder.encode([question])
        answer_embedding = self.embedder.encode([answer])
        
        similarity = cosine_similarity(question_embedding, answer_embedding)[0][0]
        return max(0.0, similarity)  # 确保不为负数
    
    def context_precision(self, question: str, contexts: List[str], ground_truth: str = None) -> float:
        """
        上下文精确度：检索到的上下文的精确程度
        """
        if not contexts or not question:
            return 0.0
        
        question_embedding = self.embedder.encode([question])
        
        relevant_count = 0
        for context in contexts:
            context_embedding = self.embedder.encode([context])
            similarity = cosine_similarity(question_embedding, context_embedding)[0][0]
            
            if similarity > self.config.context_precision_threshold:
                relevant_count += 1
        
        return relevant_count / len(contexts) if contexts else 0.0
    
    def context_recall(self, contexts: List[str], ground_truth: str) -> float:
        """
        上下文召回率：相关上下文的召回程度
        """
        if not contexts or not ground_truth:
            return 0.0
        
        # 计算ground truth与所有上下文的最大相似度
        gt_embedding = self.embedder.encode([ground_truth])
        
        max_similarity = 0.0
        for context in contexts:
            context_embedding = self.embedder.encode([context])
            similarity = cosine_similarity(gt_embedding, context_embedding)[0][0]
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _split_sentences(self, text: str) -> List[str]:
        """简单的句子分割"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

class SemanticEvaluator:
    """语义评估器"""
    
    def __init__(self, config: ModernEvaluatorConfig):
        self.config = config
        self.embedder = SentenceTransformer(config.embedding_model)
        if torch.cuda.is_available():
            self.embedder = self.embedder.to(config.device)
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的语义相似度"""
        if not text1 or not text2:
            return 0.0
        
        embeddings = self.embedder.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return max(0.0, similarity)
    
    def answer_completeness(self, question: str, answer: str, expected_aspects: List[str]) -> float:
        """
        答案完整性：检查答案是否涵盖了预期的各个方面
        """
        if not answer or not expected_aspects:
            return 0.0
        
        answer_embedding = self.embedder.encode([answer])
        
        covered_count = 0
        for aspect in expected_aspects:
            aspect_embedding = self.embedder.encode([aspect])
            similarity = cosine_similarity(answer_embedding, aspect_embedding)[0][0]
            
            if similarity > self.config.semantic_similarity_threshold:
                covered_count += 1
        
        return covered_count / len(expected_aspects)
    
    def information_density(self, text: str) -> float:
        """
        信息密度：文本中信息的密集程度
        简化版本：基于独特词汇比例和句子复杂度
        """
        if not text:
            return 0.0
        
        words = text.lower().split()
        if not words:
            return 0.0
        
        # 独特词汇比例
        unique_words = set(words)
        uniqueness_ratio = len(unique_words) / len(words)
        
        # 平均句子长度（复杂度指标）
        sentences = self._split_sentences(text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        # 结合两个指标
        density_score = (uniqueness_ratio + min(avg_sentence_length / 20, 1.0)) / 2
        return density_score
    
    def _split_sentences(self, text: str) -> List[str]:
        """简单的句子分割"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

class ComplexityAwareEvaluator:
    """复杂度感知评估器"""
    
    def __init__(self, config: ModernEvaluatorConfig):
        self.config = config
        self.ragas_evaluator = RAGASEvaluator(config)
        self.semantic_evaluator = SemanticEvaluator(config)
    
    def evaluate_by_complexity(self, 
                              question: str,
                              answer: str,
                              contexts: List[str],
                              complexity_type: str,
                              ground_truth: str = None) -> Dict[str, float]:
        """
        基于复杂度的评估
        
        Args:
            question: 问题
            answer: 生成的答案
            contexts: 检索到的上下文
            complexity_type: 复杂度类型 (zero_hop, one_hop, multi_hop)
            ground_truth: 标准答案（可选）
        """
        # 基础评估指标
        metrics = {}
        
        # RAGAS指标
        if self.config.enable_ragas_metrics:
            metrics['faithfulness'] = self.ragas_evaluator.faithfulness(answer, contexts)
            metrics['answer_relevancy'] = self.ragas_evaluator.answer_relevancy(question, answer)
            metrics['context_precision'] = self.ragas_evaluator.context_precision(question, contexts)
            
            if ground_truth:
                metrics['context_recall'] = self.ragas_evaluator.context_recall(contexts, ground_truth)
        
        # 语义指标
        if ground_truth:
            metrics['semantic_similarity'] = self.semantic_evaluator.semantic_similarity(answer, ground_truth)
        
        metrics['information_density'] = self.semantic_evaluator.information_density(answer)
        
        # 复杂度加权
        if self.config.enable_complexity_aware_eval:
            complexity_weight = self.config.complexity_weights.get(complexity_type, 1.0)
            
            # 对某些指标应用复杂度权重
            weighted_metrics = {}
            for key, value in metrics.items():
                if key in ['faithfulness', 'answer_relevancy']:
                    # 复杂查询的容错率更高
                    weighted_metrics[f'{key}_weighted'] = min(value * complexity_weight, 1.0)
                else:
                    weighted_metrics[f'{key}_weighted'] = value
            
            metrics.update(weighted_metrics)
        
        # 计算综合分数
        key_metrics = ['faithfulness', 'answer_relevancy', 'context_precision']
        if ground_truth:
            key_metrics.append('semantic_similarity')
        
        available_metrics = [metrics.get(m, 0) for m in key_metrics if m in metrics]
        metrics['overall_score'] = np.mean(available_metrics) if available_metrics else 0.0
        
        return metrics

class ModernEvaluator:
    """现代化RAG评估器"""
    
    def __init__(self, config: Optional[ModernEvaluatorConfig] = None):
        self.config = config or ModernEvaluatorConfig()
        
        # 初始化子评估器
        self.ragas_evaluator = RAGASEvaluator(self.config)
        self.semantic_evaluator = SemanticEvaluator(self.config)
        self.complexity_evaluator = ComplexityAwareEvaluator(self.config)
        
        # 评估历史
        self.evaluation_history = []
    
    def evaluate_single(self,
                       question: str,
                       answer: str,
                       contexts: List[str],
                       complexity_type: str = "one_hop",
                       ground_truth: str = None,
                       source_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        评估单个问答对
        
        Args:
            question: 问题
            answer: 生成的答案
            contexts: 检索到的上下文列表
            complexity_type: 查询复杂度类型
            ground_truth: 标准答案（可选）
            source_info: 源信息（各检索源的贡献等）
        """
        # 基础评估
        evaluation_result = self.complexity_evaluator.evaluate_by_complexity(
            question=question,
            answer=answer,
            contexts=contexts,
            complexity_type=complexity_type,
            ground_truth=ground_truth
        )
        
        # 添加元信息
        evaluation_result.update({
            'question': question,
            'answer': answer,
            'complexity_type': complexity_type,
            'num_contexts': len(contexts),
            'answer_length': len(answer.split()) if answer else 0,
            'contexts_total_length': sum(len(ctx.split()) for ctx in contexts)
        })
        
        # 源信息分析（如果提供）
        if source_info:
            evaluation_result['source_analysis'] = self._analyze_source_contribution(
                contexts, source_info
            )
        
        # 保存到历史
        self.evaluation_history.append(evaluation_result)
        
        return evaluation_result
    
    def evaluate_batch(self,
                      questions: List[str],
                      answers: List[str], 
                      contexts_list: List[List[str]],
                      complexity_types: List[str] = None,
                      ground_truths: List[str] = None) -> Dict[str, Any]:
        """批量评估"""
        
        if complexity_types is None:
            complexity_types = ["one_hop"] * len(questions)
        
        if ground_truths is None:
            ground_truths = [None] * len(questions)
        
        # 逐个评估
        individual_results = []
        for i, (q, a, c, ct, gt) in enumerate(zip(
            questions, answers, contexts_list, complexity_types, ground_truths
        )):
            result = self.evaluate_single(q, a, c, ct, gt)
            individual_results.append(result)
        
        # 计算批次统计
        batch_stats = self._compute_batch_statistics(individual_results)
        
        return {
            'individual_results': individual_results,
            'batch_statistics': batch_stats,
            'total_evaluated': len(questions)
        }
    
    def _analyze_source_contribution(self, 
                                   contexts: List[str],
                                   source_info: Dict[str, Any]) -> Dict[str, float]:
        """分析各源的贡献度"""
        analysis = {}
        
        if 'source_scores' in source_info:
            scores = source_info['source_scores']
            total_score = sum(scores.values())
            
            for source, score in scores.items():
                analysis[f'{source}_contribution'] = score / total_score if total_score > 0 else 0
        
        if 'source_weights' in source_info:
            analysis['source_weights'] = source_info['source_weights']
        
        return analysis
    
    def _compute_batch_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算批次统计信息"""
        metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 
                  'semantic_similarity', 'information_density', 'overall_score']
        
        stats = {}
        
        for metric in metrics:
            values = [r.get(metric, 0) for r in results if metric in r]
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # 按复杂度分组统计
        complexity_stats = {}
        for complexity in ['zero_hop', 'one_hop', 'multi_hop']:
            complexity_results = [r for r in results if r.get('complexity_type') == complexity]
            if complexity_results:
                complexity_scores = [r.get('overall_score', 0) for r in complexity_results]
                complexity_stats[complexity] = {
                    'count': len(complexity_results),
                    'mean_score': np.mean(complexity_scores),
                    'std_score': np.std(complexity_scores)
                }
        
        stats['complexity_breakdown'] = complexity_stats
        
        return stats
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估总结"""
        if not self.evaluation_history:
            return {'message': 'No evaluations performed yet'}
        
        return self._compute_batch_statistics(self.evaluation_history)

def create_modern_evaluator(config: Optional[ModernEvaluatorConfig] = None) -> ModernEvaluator:
    """创建现代评估器"""
    return ModernEvaluator(config)

def create_modern_evaluator_config(**kwargs) -> ModernEvaluatorConfig:
    """创建现代评估器配置"""
    return ModernEvaluatorConfig(**kwargs) 