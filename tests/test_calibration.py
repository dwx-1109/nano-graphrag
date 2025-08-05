"""
置信度校准测试脚本

用于验证温度缩放在复杂度分类上的效果
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple
from nano_graphrag.complexity.classifier import ComplexityClassifier, ComplexityClassifierConfig
from nano_graphrag.complexity.calibration import compute_ece, compute_ece_binary

logger = logging.getLogger(__name__)

def create_synthetic_validation_data(n_samples: int = 1000) -> Tuple[List[str], List[str]]:
    """
    创建合成验证数据（用于校准演示）
    
    Returns:
        queries: 查询列表
        labels: 标签列表
    """
    queries = []
    labels = []
    
    # Zero-hop 查询（简单定义性问题）
    zero_hop_templates = [
        "What is {}?",
        "Define {}.",
        "Explain the meaning of {}.",
        "{}的定义是什么？",
        "什么是{}？"
    ]
    zero_hop_terms = ["machine learning", "neural network", "AI", "深度学习", "人工智能"]
    
    # One-hop 查询（单步推理）
    one_hop_templates = [
        "How does {} work?",
        "What are the benefits of {}?",
        "List the components of {}.",
        "{}是如何工作的？",
        "{}有什么优点？"
    ]
    one_hop_terms = ["gradient descent", "backpropagation", "transformer", "梯度下降", "反向传播"]
    
    # Multi-hop 查询（复杂推理）
    multi_hop_templates = [
        "Compare {} and {} in terms of performance and complexity.",
        "What is the relationship between {} and {}?",
        "How do {} and {} work together to achieve {}?",
        "比较{}和{}的性能差异。",
        "{}和{}之间有什么关系？"
    ]
    multi_hop_pairs = [
        ("CNN", "RNN"), ("BERT", "GPT"), ("supervised learning", "unsupervised learning"),
        ("卷积神经网络", "循环神经网络"), ("监督学习", "无监督学习")
    ]
    
    # 生成样本
    samples_per_type = n_samples // 3
    
    # Zero-hop
    for _ in range(samples_per_type):
        template = np.random.choice(zero_hop_templates)
        term = np.random.choice(zero_hop_terms)
        query = template.format(term)
        queries.append(query)
        labels.append("zero_hop")
    
    # One-hop
    for _ in range(samples_per_type):
        template = np.random.choice(one_hop_templates)
        term = np.random.choice(one_hop_terms)
        query = template.format(term)
        queries.append(query)
        labels.append("one_hop")
    
    # Multi-hop
    for _ in range(samples_per_type):
        template = np.random.choice(multi_hop_templates)
        if "{}" in template and template.count("{}") >= 2:
            pair = multi_hop_pairs[np.random.randint(0, len(multi_hop_pairs))]
            if template.count("{}") == 2:
                query = template.format(pair[0], pair[1])
            else:  # 3 placeholders
                query = template.format(pair[0], pair[1], "better results")
        else:
            query = template
        queries.append(query)
        labels.append("multi_hop")
    
    return queries, labels

def evaluate_calibration_improvement(classifier: ComplexityClassifier, 
                                   validation_queries: List[str],
                                   validation_labels: List[str]) -> Dict[str, float]:
    """
    评估校准改善效果
    
    Args:
        classifier: 复杂度分类器
        validation_queries: 验证查询
        validation_labels: 验证标签
        
    Returns:
        评估结果
    """
    if not classifier.is_available():
        logger.error("分类器模型未加载")
        return {}
    
    logger.info("开始评估校准改善效果...")
    
    # 收集原始预测
    original_confidences = []
    predictions = []
    correct_predictions = []
    
    for query, true_label in zip(validation_queries, validation_labels):
        complexity, confidence, prob_dict = classifier.predict_with_confidence(query)
        
        original_confidences.append(confidence)
        predictions.append(complexity)
        correct_predictions.append(1 if complexity == true_label else 0)
    
    # 计算原始ECE（二分类版本：预测正确与否）
    original_ece = compute_ece_binary(
        np.array(original_confidences), 
        np.array(correct_predictions)
    )
    
    # 计算准确率
    accuracy = np.mean(correct_predictions)
    
    # 执行校准
    calibration_results = classifier.calibrate_confidence(
        validation_queries, validation_labels, verbose=True
    )
    
    results = {
        'accuracy': accuracy,
        'original_ece': original_ece,
        'validation_samples': len(validation_queries),
        **calibration_results
    }
    
    logger.info("校准评估完成")
    logger.info(f"准确率: {accuracy:.4f}")
    logger.info(f"原始ECE: {original_ece:.4f}")
    if 'post_calibration_ece' in calibration_results:
        logger.info(f"校准后ECE: {calibration_results['post_calibration_ece']:.4f}")
        logger.info(f"ECE改善: {original_ece - calibration_results['post_calibration_ece']:.4f}")
    
    return results

def test_temperature_scaling():
    """测试温度缩放校准"""
    logger.info("=== 开始温度缩放校准测试 ===")
    
    # 创建分类器配置（启用校准）
    config = ComplexityClassifierConfig(
        enable_calibration=True,
        calibration_method="temperature_scaling"
    )
    
    # 初始化分类器
    classifier = ComplexityClassifier(config)
    
    if not classifier.is_available():
        logger.warning("模型未加载，使用模拟数据进行演示")
        return
    
    # 创建验证数据
    validation_queries, validation_labels = create_synthetic_validation_data(n_samples=300)
    
    # 评估校准改善
    results = evaluate_calibration_improvement(
        classifier, validation_queries, validation_labels
    )
    
    logger.info("=== 温度缩放校准测试完成 ===")
    return results

def test_different_calibration_methods():
    """测试不同校准方法的对比"""
    logger.info("=== 开始校准方法对比测试 ===")
    
    methods = ["temperature_scaling", "platt_scaling", "isotonic"]
    results = {}
    
    # 创建验证数据
    validation_queries, validation_labels = create_synthetic_validation_data(n_samples=200)
    
    for method in methods:
        logger.info(f"测试 {method} 校准方法...")
        
        config = ComplexityClassifierConfig(
            enable_calibration=True,
            calibration_method=method
        )
        
        classifier = ComplexityClassifier(config)
        
        if not classifier.is_available():
            logger.warning(f"模型未加载，跳过 {method} 测试")
            continue
        
        method_results = evaluate_calibration_improvement(
            classifier, validation_queries, validation_labels
        )
        
        results[method] = method_results
        logger.info(f"{method} 测试完成")
    
    # 对比结果
    logger.info("=== 校准方法对比结果 ===")
    for method, result in results.items():
        ece_improvement = result.get('ece_improvement', 0)
        logger.info(f"{method}: ECE改善 {ece_improvement:.4f}")
    
    return results

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_temperature_scaling()
    test_different_calibration_methods() 