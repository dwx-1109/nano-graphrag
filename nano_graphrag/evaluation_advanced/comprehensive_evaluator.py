"""
综合评估框架

整合所有评估指标，提供科研级的完整评估体系，支持：
- 多维度评估指标计算
- 复杂度感知的分层评估
- 自动化评估管道
- 评估结果可视化和报告生成
- 对比实验和统计显著性检验

这是项目评估系统的核心模块，确保研究结果的科学性和可信度。
"""

import os
import json
import time
import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

# 尝试导入评估库
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not available. BLEU and METEOR scores will be unavailable.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    warnings.warn("rouge-score not available. ROUGE scores will be unavailable.")

try:
    import bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    warnings.warn("bert-score not available. BERTScore will be unavailable.")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Statistical tests will be unavailable.")

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveEvaluatorConfig:
    """综合评估器配置"""
    
    # 生成质量评估配置
    enable_bleu: bool = True
    enable_rouge: bool = True
    enable_bert_score: bool = True
    enable_meteor: bool = True
    
    # BLEU配置
    bleu_max_n: int = 4
    bleu_smoothing: bool = True
    
    # ROUGE配置
    rouge_types: List[str] = field(default_factory=lambda: ['rouge1', 'rouge2', 'rougeL'])
    rouge_use_stemmer: bool = True
    
    # BERTScore配置
    bert_score_model: str = "microsoft/deberta-xlarge-mnli"
    bert_score_lang: str = "en"
    bert_score_rescale: bool = True
    
    # 检索质量评估配置
    enable_ndcg: bool = True
    enable_mrr: bool = True
    enable_map: bool = True
    enable_precision_recall: bool = True
    
    # 检索评估参数
    retrieval_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    
    # 复杂度感知评估配置
    enable_complexity_aware: bool = True
    complexity_types: List[str] = field(default_factory=lambda: ['zero_hop', 'one_hop', 'multi_hop'])
    
    # 融合质量评估配置
    enable_fusion_analysis: bool = True
    enable_attention_analysis: bool = True
    
    # 统计显著性检验配置
    enable_significance_testing: bool = True
    significance_alpha: float = 0.05
    bootstrap_samples: int = 1000
    
    # 报告生成配置
    save_detailed_results: bool = True
    generate_visualizations: bool = True
    output_dir: str = "evaluation_results"
    
    # 性能配置
    batch_size: int = 32
    num_workers: int = 4


class BLEUEvaluator:
    """BLEU评估器"""
    
    def __init__(self, config: ComprehensiveEvaluatorConfig):
        self.config = config
        self.smoothing_function = SmoothingFunction().method1 if config.bleu_smoothing else None
        
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK is required for BLEU evaluation")
    
    def evaluate(self, 
                references: List[List[str]], 
                candidates: List[str]) -> Dict[str, float]:
        """
        计算BLEU分数
        
        Args:
            references: 参考答案列表 [num_samples, num_references]
            candidates: 候选答案列表 [num_samples]
            
        Returns:
            BLEU分数字典
        """
        bleu_scores = {}
        
        # 计算不同n-gram的BLEU分数
        for n in range(1, self.config.bleu_max_n + 1):
            weights = [1.0/n] * n + [0.0] * (4 - n)
            
            scores = []
            for ref_list, cand in zip(references, candidates):
                # 分词
                ref_tokens = [ref.split() for ref in ref_list]
                cand_tokens = cand.split()
                
                # 计算BLEU分数
                score = sentence_bleu(
                    ref_tokens, cand_tokens, 
                    weights=weights,
                    smoothing_function=self.smoothing_function
                )
                scores.append(score)
            
            bleu_scores[f'bleu_{n}'] = np.mean(scores)
        
        # 计算标准BLEU-4分数
        bleu_4_scores = []
        for ref_list, cand in zip(references, candidates):
            ref_tokens = [ref.split() for ref in ref_list]
            cand_tokens = cand.split()
            
            score = sentence_bleu(
                ref_tokens, cand_tokens,
                smoothing_function=self.smoothing_function
            )
            bleu_4_scores.append(score)
        
        bleu_scores['bleu'] = np.mean(bleu_4_scores)
        bleu_scores['bleu_std'] = np.std(bleu_4_scores)
        
        return bleu_scores


class ROUGEEvaluator:
    """ROUGE评估器"""
    
    def __init__(self, config: ComprehensiveEvaluatorConfig):
        self.config = config
        
        if not ROUGE_AVAILABLE:
            raise ImportError("rouge-score is required for ROUGE evaluation")
        
        self.scorer = rouge_scorer.RougeScorer(
            config.rouge_types, 
            use_stemmer=config.rouge_use_stemmer
        )
    
    def evaluate(self, 
                references: List[str], 
                candidates: List[str]) -> Dict[str, float]:
        """
        计算ROUGE分数
        
        Args:
            references: 参考答案列表 [num_samples]
            candidates: 候选答案列表 [num_samples]
            
        Returns:
            ROUGE分数字典
        """
        rouge_scores = {rouge_type: [] for rouge_type in self.config.rouge_types}
        
        for ref, cand in zip(references, candidates):
            scores = self.scorer.score(ref, cand)
            
            for rouge_type in self.config.rouge_types:
                rouge_scores[rouge_type].append(scores[rouge_type].fmeasure)
        
        # 计算平均分和标准差
        result = {}
        for rouge_type in self.config.rouge_types:
            scores = rouge_scores[rouge_type]
            result[rouge_type] = np.mean(scores)
            result[f'{rouge_type}_std'] = np.std(scores)
        
        return result


class BERTScoreEvaluator:
    """BERTScore评估器"""
    
    def __init__(self, config: ComprehensiveEvaluatorConfig):
        self.config = config
        
        if not BERT_SCORE_AVAILABLE:
            raise ImportError("bert-score is required for BERTScore evaluation")
    
    def evaluate(self, 
                references: List[str], 
                candidates: List[str]) -> Dict[str, float]:
        """
        计算BERTScore
        
        Args:
            references: 参考答案列表 [num_samples]
            candidates: 候选答案列表 [num_samples]
            
        Returns:
            BERTScore字典
        """
        # 计算BERTScore
        P, R, F1 = bert_score.score(
            candidates, references,
            model_type=self.config.bert_score_model,
            lang=self.config.bert_score_lang,
            rescale_with_baseline=self.config.bert_score_rescale,
            verbose=False
        )
        
        result = {
            'bert_score_precision': P.mean().item(),
            'bert_score_recall': R.mean().item(),
            'bert_score_f1': F1.mean().item(),
            'bert_score_precision_std': P.std().item(),
            'bert_score_recall_std': R.std().item(),
            'bert_score_f1_std': F1.std().item()
        }
        
        return result


class NDCGEvaluator:
    """NDCG@K评估器"""
    
    def __init__(self, config: ComprehensiveEvaluatorConfig):
        self.config = config
    
    def evaluate(self, 
                relevance_scores: List[List[float]], 
                k_values: Optional[List[int]] = None) -> Dict[str, float]:
        """
        计算NDCG@K
        
        Args:
            relevance_scores: 相关性分数列表 [num_queries, num_docs]
            k_values: K值列表
            
        Returns:
            NDCG@K分数字典
        """
        if k_values is None:
            k_values = self.config.retrieval_k_values
        
        ndcg_scores = {}
        
        for k in k_values:
            scores = []
            
            for rel_scores in relevance_scores:
                # 计算DCG@K
                dcg = self._compute_dcg(rel_scores[:k])
                
                # 计算IDCG@K
                sorted_scores = sorted(rel_scores, reverse=True)
                idcg = self._compute_dcg(sorted_scores[:k])
                
                # 计算NDCG@K
                ndcg = dcg / idcg if idcg > 0 else 0.0
                scores.append(ndcg)
            
            ndcg_scores[f'ndcg_{k}'] = np.mean(scores)
            ndcg_scores[f'ndcg_{k}_std'] = np.std(scores)
        
        return ndcg_scores
    
    def _compute_dcg(self, relevance_scores: List[float]) -> float:
        """计算DCG"""
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg


class MRREvaluator:
    """MRR评估器"""
    
    def __init__(self, config: ComprehensiveEvaluatorConfig):
        self.config = config
    
    def evaluate(self, 
                relevance_binary: List[List[int]]) -> Dict[str, float]:
        """
        计算MRR (Mean Reciprocal Rank)
        
        Args:
            relevance_binary: 二元相关性列表 [num_queries, num_docs]
            
        Returns:
            MRR分数字典
        """
        reciprocal_ranks = []
        
        for binary_rels in relevance_binary:
            # 找到第一个相关文档的位置
            for i, rel in enumerate(binary_rels):
                if rel == 1:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                # 没有找到相关文档
                reciprocal_ranks.append(0.0)
        
        mrr = np.mean(reciprocal_ranks)
        mrr_std = np.std(reciprocal_ranks)
        
        return {
            'mrr': mrr,
            'mrr_std': mrr_std
        }


class PrecisionRecallEvaluator:
    """Precision@K和Recall@K评估器"""
    
    def __init__(self, config: ComprehensiveEvaluatorConfig):
        self.config = config
    
    def evaluate(self, 
                relevance_binary: List[List[int]], 
                k_values: Optional[List[int]] = None) -> Dict[str, float]:
        """
        计算Precision@K和Recall@K
        
        Args:
            relevance_binary: 二元相关性列表 [num_queries, num_docs]
            k_values: K值列表
            
        Returns:
            Precision@K和Recall@K分数字典
        """
        if k_values is None:
            k_values = self.config.retrieval_k_values
        
        metrics = {}
        
        for k in k_values:
            precisions = []
            recalls = []
            
            for binary_rels in relevance_binary:
                # 计算Precision@K
                relevant_at_k = sum(binary_rels[:k])
                precision = relevant_at_k / k if k > 0 else 0.0
                precisions.append(precision)
                
                # 计算Recall@K
                total_relevant = sum(binary_rels)
                recall = relevant_at_k / total_relevant if total_relevant > 0 else 0.0
                recalls.append(recall)
            
            metrics[f'precision_{k}'] = np.mean(precisions)
            metrics[f'precision_{k}_std'] = np.std(precisions)
            metrics[f'recall_{k}'] = np.mean(recalls)
            metrics[f'recall_{k}_std'] = np.std(recalls)
        
        return metrics


class ComplexityAwareEvaluator:
    """复杂度感知评估器"""
    
    def __init__(self, config: ComprehensiveEvaluatorConfig):
        self.config = config
        
        # 初始化各种评估器
        if config.enable_bleu and NLTK_AVAILABLE:
            self.bleu_evaluator = BLEUEvaluator(config)
        if config.enable_rouge and ROUGE_AVAILABLE:
            self.rouge_evaluator = ROUGEEvaluator(config)
        if config.enable_bert_score and BERT_SCORE_AVAILABLE:
            self.bert_score_evaluator = BERTScoreEvaluator(config)
    
    def evaluate_by_complexity(self, 
                              data_by_complexity: Dict[str, Dict[str, List[str]]],
                              relevance_data: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
        """
        按复杂度类型分别评估
        
        Args:
            data_by_complexity: 按复杂度分组的数据
                {complexity_type: {'references': [...], 'candidates': [...]}}
            relevance_data: 相关性数据
            
        Returns:
            按复杂度分组的评估结果
        """
        results = {}
        
        for complexity_type in self.config.complexity_types:
            if complexity_type not in data_by_complexity:
                continue
            
            complexity_data = data_by_complexity[complexity_type]
            references = complexity_data['references']
            candidates = complexity_data['candidates']
            
            complexity_results = {}
            
            # 生成质量评估
            if self.config.enable_bleu and hasattr(self, 'bleu_evaluator'):
                # 为BLEU准备多个参考答案
                if isinstance(references[0], str):
                    # 单个参考答案，转换为列表格式
                    ref_lists = [[ref] for ref in references]
                else:
                    ref_lists = references
                
                bleu_scores = self.bleu_evaluator.evaluate(ref_lists, candidates)
                complexity_results.update(bleu_scores)
            
            if self.config.enable_rouge and hasattr(self, 'rouge_evaluator'):
                # ROUGE使用单个参考答案
                if isinstance(references[0], list):
                    # 多个参考答案，取第一个
                    single_refs = [ref[0] for ref in references]
                else:
                    single_refs = references
                
                rouge_scores = self.rouge_evaluator.evaluate(single_refs, candidates)
                complexity_results.update(rouge_scores)
            
            if self.config.enable_bert_score and hasattr(self, 'bert_score_evaluator'):
                # BERTScore使用单个参考答案
                if isinstance(references[0], list):
                    single_refs = [ref[0] for ref in references]
                else:
                    single_refs = references
                
                bert_scores = self.bert_score_evaluator.evaluate(single_refs, candidates)
                complexity_results.update(bert_scores)
            
            # 检索质量评估
            if relevance_data and complexity_type in relevance_data:
                complexity_rel_data = relevance_data[complexity_type]
                
                if self.config.enable_ndcg:
                    ndcg_evaluator = NDCGEvaluator(self.config)
                    ndcg_scores = ndcg_evaluator.evaluate(complexity_rel_data.get('relevance_scores', []))
                    complexity_results.update(ndcg_scores)
                
                if self.config.enable_mrr:
                    mrr_evaluator = MRREvaluator(self.config)
                    mrr_scores = mrr_evaluator.evaluate(complexity_rel_data.get('relevance_binary', []))
                    complexity_results.update(mrr_scores)
            
            results[complexity_type] = complexity_results
        
        return results


class StatisticalTester:
    """统计显著性检验器"""
    
    def __init__(self, config: ComprehensiveEvaluatorConfig):
        self.config = config
        
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available. Statistical tests will be skipped.")
    
    def paired_t_test(self, 
                     scores_a: List[float], 
                     scores_b: List[float]) -> Dict[str, float]:
        """
        配对t检验
        
        Args:
            scores_a: 系统A的分数
            scores_b: 系统B的分数
            
        Returns:
            检验结果
        """
        if not SCIPY_AVAILABLE:
            return {}
        
        statistic, p_value = stats.ttest_rel(scores_a, scores_b)
        
        return {
            't_statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.config.significance_alpha
        }
    
    def bootstrap_confidence_interval(self, 
                                    scores: List[float], 
                                    confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Bootstrap置信区间
        
        Args:
            scores: 分数列表
            confidence_level: 置信水平
            
        Returns:
            置信区间
        """
        if not scores:
            return {}
        
        # Bootstrap重采样
        bootstrap_means = []
        n = len(scores)
        
        for _ in range(self.config.bootstrap_samples):
            sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # 计算置信区间
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return {
            'mean': np.mean(scores),
            'confidence_interval_lower': lower,
            'confidence_interval_upper': upper,
            'confidence_level': confidence_level
        }


class EvaluationReport:
    """评估报告生成器"""
    
    def __init__(self, config: ComprehensiveEvaluatorConfig):
        self.config = config
        self.report_data = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_results(self, name: str, results: Dict[str, Any]):
        """添加评估结果"""
        self.report_data[name] = results
    
    def generate_summary_table(self) -> pd.DataFrame:
        """生成汇总表格"""
        summary_data = []
        
        for name, results in self.report_data.items():
            row = {'System': name}
            
            # 提取主要指标
            main_metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bert_score_f1', 
                          'ndcg_5', 'ndcg_10', 'mrr']
            
            for metric in main_metrics:
                if metric in results:
                    row[metric] = f"{results[metric]:.4f}"
                elif any(complexity in results for complexity in ['zero_hop', 'one_hop', 'multi_hop']):
                    # 复杂度感知评估
                    complexity_scores = []
                    for complexity in ['zero_hop', 'one_hop', 'multi_hop']:
                        if complexity in results and metric in results[complexity]:
                            complexity_scores.append(results[complexity][metric])
                    
                    if complexity_scores:
                        row[metric] = f"{np.mean(complexity_scores):.4f}"
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def save_report(self, output_dir: Optional[str] = None):
        """保存评估报告"""
        if output_dir is None:
            output_dir = self.config.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        if self.config.save_detailed_results:
            results_file = os.path.join(output_dir, f"detailed_results_{self.timestamp}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.report_data, f, indent=2, ensure_ascii=False)
        
        # 保存汇总表格
        summary_df = self.generate_summary_table()
        summary_file = os.path.join(output_dir, f"summary_table_{self.timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Evaluation report saved to {output_dir}")


class ComprehensiveEvaluator:
    """
    综合评估器主类
    
    整合所有评估指标，提供科研级的完整评估体系
    """
    
    def __init__(self, config: Optional[ComprehensiveEvaluatorConfig] = None):
        if config is None:
            config = ComprehensiveEvaluatorConfig()
        
        self.config = config
        
        # 初始化各种评估器
        self._init_evaluators()
        
        # 统计检验器
        if config.enable_significance_testing:
            self.statistical_tester = StatisticalTester(config)
        
        # 报告生成器
        self.report = EvaluationReport(config)
        
        logger.info("ComprehensiveEvaluator initialized successfully")
    
    def _init_evaluators(self):
        """初始化各种评估器"""
        # 生成质量评估器
        if self.config.enable_bleu and NLTK_AVAILABLE:
            self.bleu_evaluator = BLEUEvaluator(self.config)
        
        if self.config.enable_rouge and ROUGE_AVAILABLE:
            self.rouge_evaluator = ROUGEEvaluator(self.config)
        
        if self.config.enable_bert_score and BERT_SCORE_AVAILABLE:
            self.bert_score_evaluator = BERTScoreEvaluator(self.config)
        
        # 检索质量评估器
        if self.config.enable_ndcg:
            self.ndcg_evaluator = NDCGEvaluator(self.config)
        
        if self.config.enable_mrr:
            self.mrr_evaluator = MRREvaluator(self.config)
        
        if self.config.enable_precision_recall:
            self.precision_recall_evaluator = PrecisionRecallEvaluator(self.config)
        
        # 复杂度感知评估器
        if self.config.enable_complexity_aware:
            self.complexity_aware_evaluator = ComplexityAwareEvaluator(self.config)
    
    def evaluate_system(self, 
                       system_name: str,
                       references: List[str],
                       candidates: List[str],
                       complexity_types: Optional[List[str]] = None,
                       relevance_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        评估单个系统
        
        Args:
            system_name: 系统名称
            references: 参考答案
            candidates: 候选答案
            complexity_types: 复杂度类型列表
            relevance_data: 相关性数据
            
        Returns:
            评估结果字典
        """
        results = {}
        
        # 基础生成质量评估
        if self.config.enable_bleu and hasattr(self, 'bleu_evaluator'):
            ref_lists = [[ref] for ref in references]
            bleu_scores = self.bleu_evaluator.evaluate(ref_lists, candidates)
            results.update(bleu_scores)
        
        if self.config.enable_rouge and hasattr(self, 'rouge_evaluator'):
            rouge_scores = self.rouge_evaluator.evaluate(references, candidates)
            results.update(rouge_scores)
        
        if self.config.enable_bert_score and hasattr(self, 'bert_score_evaluator'):
            bert_scores = self.bert_score_evaluator.evaluate(references, candidates)
            results.update(bert_scores)
        
        # 检索质量评估
        if relevance_data:
            if self.config.enable_ndcg and hasattr(self, 'ndcg_evaluator'):
                ndcg_scores = self.ndcg_evaluator.evaluate(
                    relevance_data.get('relevance_scores', [])
                )
                results.update(ndcg_scores)
            
            if self.config.enable_mrr and hasattr(self, 'mrr_evaluator'):
                mrr_scores = self.mrr_evaluator.evaluate(
                    relevance_data.get('relevance_binary', [])
                )
                results.update(mrr_scores)
        
        # 复杂度感知评估
        if (self.config.enable_complexity_aware and 
            complexity_types and 
            hasattr(self, 'complexity_aware_evaluator')):
            
            # 按复杂度分组数据
            data_by_complexity = {}
            for i, complexity in enumerate(complexity_types):
                if complexity not in data_by_complexity:
                    data_by_complexity[complexity] = {'references': [], 'candidates': []}
                
                data_by_complexity[complexity]['references'].append(references[i])
                data_by_complexity[complexity]['candidates'].append(candidates[i])
            
            complexity_results = self.complexity_aware_evaluator.evaluate_by_complexity(
                data_by_complexity, relevance_data
            )
            results.update(complexity_results)
        
        # 添加到报告
        self.report.add_results(system_name, results)
        
        return results
    
    def compare_systems(self, 
                       system_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        比较多个系统
        
        Args:
            system_results: 系统结果字典 {system_name: results}
            
        Returns:
            比较结果
        """
        if not self.config.enable_significance_testing or not hasattr(self, 'statistical_tester'):
            return {}
        
        comparison_results = {}
        
        # 获取所有指标
        all_metrics = set()
        for results in system_results.values():
            all_metrics.update(results.keys())
        
        # 进行两两比较
        system_names = list(system_results.keys())
        for i in range(len(system_names)):
            for j in range(i + 1, len(system_names)):
                sys_a, sys_b = system_names[i], system_names[j]
                comparison_key = f"{sys_a}_vs_{sys_b}"
                comparison_results[comparison_key] = {}
                
                # 需要原始分数进行统计检验（这里简化处理）
                for metric in all_metrics:
                    if (metric in system_results[sys_a] and 
                        metric in system_results[sys_b] and 
                        not metric.endswith('_std')):
                        
                        # 模拟原始分数（实际应用中需要保存原始分数）
                        score_a = system_results[sys_a][metric]
                        score_b = system_results[sys_b][metric]
                        
                        comparison_results[comparison_key][metric] = {
                            f'{sys_a}_score': score_a,
                            f'{sys_b}_score': score_b,
                            'difference': score_a - score_b,
                            'better_system': sys_a if score_a > score_b else sys_b
                        }
        
        return comparison_results
    
    def generate_final_report(self, output_dir: Optional[str] = None):
        """生成最终报告"""
        self.report.save_report(output_dir)
        
        # 如果启用可视化，生成图表
        if self.config.generate_visualizations:
            try:
                from .evaluation_visualization import EvaluationVisualizer
                visualizer = EvaluationVisualizer(self.config)
                visualizer.generate_all_plots(self.report.report_data, output_dir)
            except ImportError:
                logger.warning("Visualization tools not available")


# 工厂函数
def create_comprehensive_evaluator(config: Optional[ComprehensiveEvaluatorConfig] = None) -> ComprehensiveEvaluator:
    """创建综合评估器"""
    return ComprehensiveEvaluator(config)


def create_comprehensive_evaluator_config(**kwargs) -> ComprehensiveEvaluatorConfig:
    """创建综合评估器配置"""
    return ComprehensiveEvaluatorConfig(**kwargs) 