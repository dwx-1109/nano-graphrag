#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModernBERT复杂度分类器性能测试脚本
用于测试训练好的模型在测试集上的性能，包含完整的评估指标和可视化
"""

import os
import json
import torch
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComplexityClassifierTester:
    def __init__(self, model_path: str, base_model_path: str = "models/modernbert/answerdotai_ModernBERT-large"):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 类别映射
        self.label2id = {"zero_hop": 0, "one_hop": 1, "multi_hop": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        self.tokenizer = None
        self.model = None
        self.log_buffer = []  # 用于存储日志信息
        
    def load_model(self):
        """加载训练好的模型"""
        try:
            logger.info(f"加载分词器: {self.base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            
            logger.info(f"加载基础模型: {self.base_model_path}")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_path,
                num_labels=3,
                id2label=self.id2label,
                label2id=self.label2id
            )
            
            logger.info(f"加载LoRA适配器: {self.model_path}")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ 模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """对单个文本进行预测"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        # 编码文本
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_id = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
        
        return {
            "text": text,
            "predicted_label": self.id2label[predicted_id],
            "predicted_id": predicted_id,
            "confidence": confidence,
            "probabilities": {
                self.id2label[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        }
    
    def predict_batch(self, texts: List[str]) -> Tuple[List[Dict], List[np.ndarray]]:
        """批量预测，返回预测结果和概率分布"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        results = []
        all_probabilities = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"批量预测进度: {i}/{len(texts)}")
            
            result = self.predict_single(text)
            results.append(result)
            
            # 收集概率分布
            probs = np.array([result['probabilities'][label] for label in self.id2label.values()])
            all_probabilities.append(probs)
        
        return results, np.array(all_probabilities)
    
    def evaluate_on_dataset(self, test_data_path: str) -> Dict[str, Any]:
        """在测试集上评估模型性能"""
        logger.info(f"加载测试数据: {test_data_path}")
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        logger.info(f"测试样本数: {len(test_data)}")
        
        # 准备数据
        texts = [item['query'] for item in test_data]
        true_labels = [item['complexity'] for item in test_data]
        true_ids = [self.label2id[label] for label in true_labels]
        
        # 批量预测
        predictions_results, probabilities = self.predict_batch(texts)
        predictions = [result['predicted_label'] for result in predictions_results]
        predicted_ids = [result['predicted_id'] for result in predictions_results]
        confidences = [result['confidence'] for result in predictions_results]
        
        # 计算详细指标
        metrics = self._calculate_comprehensive_metrics(
            true_labels, predictions, predicted_ids, true_ids, 
            confidences, probabilities
        )
        
        return {
            "dataset_name": Path(test_data_path).stem,
            "test_size": len(test_data),
            "metrics": metrics,
            "predictions": list(zip(texts, true_labels, predictions, confidences)),
            "probabilities": probabilities.tolist(),
            "true_ids": [int(x) for x in true_ids],
            "predicted_ids": [int(x) for x in predicted_ids]
        }
    
    def evaluate_mixed_dataset(self, test_data_paths: List[str], show_details: bool = True) -> Dict[str, Any]:
        """在混合打乱的测试集上评估模型性能，避免数据泄露"""
        logger.info("🔄 开始混合数据集测试，避免数据泄露")
        
        # 加载并合并所有测试数据
        all_test_data = []
        for path in test_data_paths:
            logger.info(f"加载测试数据: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_test_data.extend(data)
        
        logger.info(f"总测试样本数: {len(all_test_data)}")
        
        # 随机打乱数据
        import random
        random.seed(42)  # 设置随机种子确保可重现
        random.shuffle(all_test_data)
        
        # 准备数据
        texts = [item['query'] for item in all_test_data]
        true_labels = [item['complexity'] for item in all_test_data]
        true_ids = [self.label2id[label] for label in true_labels]
        
        # 逐个预测并记录结果
        predictions_results = []
        all_probabilities = []
        correct_count = 0
        
        logger.info("🔍 开始逐个预测，确保测试透明度...")
        
        for i, (text, true_label, true_id) in enumerate(zip(texts, true_labels, true_ids)):
            if i % 100 == 0:
                logger.info(f"预测进度: {i}/{len(texts)}")
            
            # 单个预测
            result = self.predict_single(text)
            predictions_results.append(result)
            
            # 收集概率分布
            probs = np.array([result['probabilities'][label] for label in self.id2label.values()])
            all_probabilities.append(probs)
            
            # 检查预测是否正确
            predicted_label = result['predicted_label']
            is_correct = predicted_label == true_label
            if is_correct:
                correct_count += 1
            
            # 显示详细结果（可选）
            if show_details and i < 10:  # 只显示前10个样本的详细信息
                status = "✅" if is_correct else "❌"
                print(f"\n{status} 样本 {i+1}:")
                print(f"  问题: {text[:100]}{'...' if len(text) > 100 else ''}")
                print(f"  真实标签: {true_label}")
                print(f"  预测标签: {predicted_label}")
                print(f"  置信度: {result['confidence']:.3f}")
                print(f"  各类别概率: {dict(result['probabilities'])}")
            
            # 记录到日志文件，按照指定格式
            status_symbol = "✅" if is_correct else "❌"
            log_entry = f"{status_symbol} 样本 {i+1}:\n问题: {text}\n真实标签: {true_label}\n预测标签: {predicted_label}\n置信度: {result['confidence']:.3f}\n各类别概率: {dict(result['probabilities'])}\n"
            self.log_buffer.append(log_entry)
        
        # 统计结果
        predictions = [result['predicted_label'] for result in predictions_results]
        predicted_ids = [result['predicted_id'] for result in predictions_results]
        confidences = [result['confidence'] for result in predictions_results]
        probabilities = np.array(all_probabilities)
        
        # 计算详细指标
        metrics = self._calculate_comprehensive_metrics(
            true_labels, predictions, predicted_ids, true_ids, 
            confidences, probabilities
        )
        
        # 类别分布统计
        label_counts = {}
        for label in true_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info("📊 数据集类别分布:")
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count} 样本")
        
        return {
            "dataset_name": "mixed_test_data",
            "test_size": len(all_test_data),
            "metrics": metrics,
            "predictions": list(zip(texts, true_labels, predictions, confidences)),
            "probabilities": probabilities.tolist(),
            "true_ids": [int(x) for x in true_ids],
            "predicted_ids": [int(x) for x in predicted_ids],
            "label_distribution": label_counts,
            "correct_predictions": correct_count,
            "total_predictions": len(all_test_data)
        }
    
    def _calculate_comprehensive_metrics(self, true_labels, predictions, predicted_ids, 
                                       true_ids, confidences, probabilities):
        """计算全面的评估指标"""
        # 基础指标
        accuracy = accuracy_score(true_labels, predictions)
        
        # 详细分类报告
        # 获取实际存在的类别
        unique_labels = list(set(true_labels))
        if len(unique_labels) == 1:
            # 单类别数据集，创建简化的报告
            label = unique_labels[0]
            accuracy = accuracy_score(true_labels, predictions)
            report = {
                label: {
                    'precision': 1.0 if accuracy > 0 else 0.0,
                    'recall': 1.0 if accuracy > 0 else 0.0,
                    'f1-score': 1.0 if accuracy > 0 else 0.0,
                    'support': len(true_labels)
                }
            }
        else:
            # 多类别数据集，使用完整的分类报告
            report = classification_report(
                true_labels, 
                predictions, 
                target_names=unique_labels,
                output_dict=True,
                zero_division=0
            )
        
        # 混淆矩阵
        cm = confusion_matrix(true_ids, predicted_ids)
        
        # 精确率、召回率、F1分数
        precision, recall, f1, support = precision_recall_fscore_support(
            true_ids, predicted_ids, average=None, labels=[0, 1, 2], zero_division=0
        )
        
        # 宏平均和加权平均
        macro_precision = precision_score(true_ids, predicted_ids, average='macro', zero_division=0)
        macro_recall = recall_score(true_ids, predicted_ids, average='macro', zero_division=0)
        macro_f1 = f1_score(true_ids, predicted_ids, average='macro', zero_division=0)
        
        weighted_precision = precision_score(true_ids, predicted_ids, average='weighted', zero_division=0)
        weighted_recall = recall_score(true_ids, predicted_ids, average='weighted', zero_division=0)
        weighted_f1 = f1_score(true_ids, predicted_ids, average='weighted', zero_division=0)
        
        # 按类别统计
        class_stats = {}
        for i, label in enumerate(self.label2id.keys()):
            label_indices = [j for j, l in enumerate(true_labels) if l == label]
            if label_indices:
                label_confidences = [confidences[j] for j in label_indices]
                label_accuracies = [1 if predictions[j] == true_labels[j] else 0 for j in label_indices]
                
                class_stats[label] = {
                    "count": len(label_indices),
                    "accuracy": np.mean(label_accuracies),
                    "precision": precision[i],
                    "recall": recall[i],
                    "f1": f1[i],
                    "support": support[i],
                    "avg_confidence": np.mean(label_confidences),
                    "min_confidence": np.min(label_confidences),
                    "max_confidence": np.max(label_confidences),
                    "std_confidence": np.std(label_confidences)
                }
        
        # 置信度分析
        confidence_stats = {
            "mean_confidence": np.mean(confidences),
            "std_confidence": np.std(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "confidence_by_accuracy": {
                "correct_predictions": np.mean([confidences[i] for i, (true, pred) in enumerate(zip(true_labels, predictions)) if true == pred]),
                "incorrect_predictions": np.mean([confidences[i] for i, (true, pred) in enumerate(zip(true_labels, predictions)) if true != pred])
            }
        }
        
        return {
            "overall_accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "support_per_class": support.tolist(),
            "macro_metrics": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1
            },
            "weighted_metrics": {
                "precision": weighted_precision,
                "recall": weighted_recall,
                "f1": weighted_f1
            },
            "class_statistics": class_stats,
            "confidence_statistics": confidence_stats
        }
    
    def create_visualizations(self, results: Dict[str, Any], output_dir: str):
        """创建可视化图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 混淆矩阵热力图
        self._plot_confusion_matrix(results, output_dir)
        
        # 2. 类别性能对比
        self._plot_class_performance(results, output_dir)
        
        # 3. 置信度分布
        self._plot_confidence_distribution(results, output_dir)
        
        # 4. 准确率vs置信度关系
        self._plot_accuracy_vs_confidence(results, output_dir)
        
        logger.info(f"可视化图表已保存到: {output_dir}")
    
    def _plot_confusion_matrix(self, results: Dict[str, Any], output_dir: str):
        """绘制混淆矩阵"""
        cm = np.array(results['metrics']['confusion_matrix'])
        labels = list(self.label2id.keys())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'混淆矩阵 - {results["dataset_name"]}')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{results["dataset_name"]}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_performance(self, results: Dict[str, Any], output_dir: str):
        """绘制类别性能对比"""
        class_stats = results['metrics']['class_statistics']
        labels = list(class_stats.keys())
        metrics = ['precision', 'recall', 'f1', 'accuracy']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [class_stats[label][metric] for label in labels]
            axes[i].bar(labels, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[i].set_title(f'{metric.title()} 对比')
            axes[i].set_ylabel(metric.title())
            axes[i].set_ylim(0, 1)
            
            # 添加数值标签
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.suptitle(f'类别性能对比 - {results["dataset_name"]}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'class_performance_{results["dataset_name"]}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, results: Dict[str, Any], output_dir: str):
        """绘制置信度分布"""
        confidences = [item[3] for item in results['predictions']]
        correct = [item[1] == item[2] for item in results['predictions']]
        
        plt.figure(figsize=(12, 8))
        
        # 正确和错误预测的置信度分布
        correct_conf = [confidences[i] for i, c in enumerate(correct) if c]
        incorrect_conf = [confidences[i] for i, c in enumerate(correct) if not c]
        
        plt.hist(correct_conf, bins=30, alpha=0.7, label='正确预测', color='green')
        plt.hist(incorrect_conf, bins=30, alpha=0.7, label='错误预测', color='red')
        
        plt.xlabel('置信度')
        plt.ylabel('频次')
        plt.title(f'置信度分布 - {results["dataset_name"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confidence_distribution_{results["dataset_name"]}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy_vs_confidence(self, results: Dict[str, Any], output_dir: str):
        """绘制准确率vs置信度关系"""
        confidences = [item[3] for item in results['predictions']]
        correct = [item[1] == item[2] for item in results['predictions']]
        
        # 按置信度区间统计准确率
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(confidences, bins) - 1
        
        accuracy_by_confidence = []
        confidence_ranges = []
        
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                acc = np.mean([correct[j] for j in range(len(correct)) if mask[j]])
                conf_range = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                accuracy_by_confidence.append(acc)
                confidence_ranges.append(conf_range)
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(accuracy_by_confidence)), accuracy_by_confidence, 'o-', linewidth=2, markersize=8)
        plt.xlabel('置信度区间')
        plt.ylabel('准确率')
        plt.title(f'准确率 vs 置信度 - {results["dataset_name"]}')
        plt.xticks(range(len(confidence_ranges)), confidence_ranges, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'accuracy_vs_confidence_{results["dataset_name"]}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def interactive_test(self):
        """交互式测试"""
        logger.info("🎯 进入交互式测试模式")
        logger.info("输入文本进行复杂度分类，输入 'quit' 退出")
        
        while True:
            text = input("\n请输入文本: ").strip()
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            try:
                result = self.predict_single(text)
                print(f"\n📊 预测结果:")
                print(f"  文本: {text[:100]}{'...' if len(text) > 100 else ''}")
                print(f"  预测类别: {result['predicted_label']}")
                print(f"  置信度: {result['confidence']:.3f}")
                print(f"  各类别概率:")
                for label, prob in result['probabilities'].items():
                    print(f"    {label}: {prob:.3f}")
                    
            except Exception as e:
                logger.error(f"预测失败: {e}")

def print_comprehensive_results(results: Dict[str, Any]):
    """打印完整的测试结果"""
    print("\n" + "="*80)
    print(f"🎯 ModernBERT复杂度分类器测试结果 - {results['dataset_name']}")
    print("="*80)
    
    metrics = results['metrics']
    
    # 总体指标
    print(f"📊 总体准确率: {metrics['overall_accuracy']:.4f}")
    print(f"📈 测试样本数: {results['test_size']}")
    
    # 宏平均和加权平均指标
    print(f"\n📋 宏平均指标:")
    print(f"  精确率: {metrics['macro_metrics']['precision']:.4f}")
    print(f"  召回率: {metrics['macro_metrics']['recall']:.4f}")
    print(f"  F1分数: {metrics['macro_metrics']['f1']:.4f}")
    
    print(f"\n📋 加权平均指标:")
    print(f"  精确率: {metrics['weighted_metrics']['precision']:.4f}")
    print(f"  召回率: {metrics['weighted_metrics']['recall']:.4f}")
    print(f"  F1分数: {metrics['weighted_metrics']['f1']:.4f}")
    
    # 详细分类报告
    print(f"\n📋 详细分类报告:")
    # 获取实际存在的类别
    actual_labels = list(results['metrics']['class_statistics'].keys())
    if len(actual_labels) == 1:
        # 如果只有一个类别，使用简单的准确率报告
        print(f"单类别数据集 - {actual_labels[0]}:")
        print(f"  准确率: {metrics['overall_accuracy']:.4f}")
        print(f"  样本数: {results['test_size']}")
    else:
        # 多个类别时使用完整的分类报告
        print(classification_report(
            [item[1] for item in results['predictions']],
            [item[2] for item in results['predictions']],
            target_names=actual_labels
        ))
    
    # 类别统计
    print(f"\n📊 类别详细统计:")
    for label, stats in metrics['class_statistics'].items():
        print(f"  {label}:")
        print(f"    样本数: {stats['count']}")
        print(f"    准确率: {stats['accuracy']:.4f}")
        print(f"    精确率: {stats['precision']:.4f}")
        print(f"    召回率: {stats['recall']:.4f}")
        print(f"    F1分数: {stats['f1']:.4f}")
        print(f"    平均置信度: {stats['avg_confidence']:.4f}")
        print(f"    置信度范围: {stats['min_confidence']:.3f} - {stats['max_confidence']:.3f}")
        print(f"    置信度标准差: {stats['std_confidence']:.3f}")
    
    # 置信度统计
    conf_stats = metrics['confidence_statistics']
    print(f"\n📊 置信度统计:")
    print(f"  平均置信度: {conf_stats['mean_confidence']:.4f}")
    print(f"  置信度标准差: {conf_stats['std_confidence']:.4f}")
    print(f"  置信度范围: {conf_stats['min_confidence']:.3f} - {conf_stats['max_confidence']:.3f}")
    print(f"  正确预测平均置信度: {conf_stats['confidence_by_accuracy']['correct_predictions']:.4f}")
    print(f"  错误预测平均置信度: {conf_stats['confidence_by_accuracy']['incorrect_predictions']:.4f}")

def extract_error_samples(log_file_path: str, output_dir: str):
    """从详细测试日志中提取错误样本"""
    print(f"正在从 {log_file_path} 提取错误样本...")
    
    if not os.path.exists(log_file_path):
        print(f"日志文件不存在: {log_file_path}")
        return
    
    error_samples = []
    current_sample = {}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith('❌ 样本'):
                # 保存前一个错误样本
                if current_sample:
                    error_samples.append(current_sample)
                    current_sample = {}
                
                # 解析样本编号
                parts = line.split(':')
                if len(parts) >= 2:
                    current_sample['sample_id'] = parts[1].strip()
            
            elif line.startswith('  问题:'):
                current_sample['query'] = line.replace('  问题:', '').strip()
            
            elif line.startswith('  真实标签:'):
                current_sample['true_label'] = line.replace('  真实标签:', '').strip()
            
            elif line.startswith('  预测标签:'):
                current_sample['predicted_label'] = line.replace('  预测标签:', '').strip()
            
            elif line.startswith('  置信度:'):
                current_sample['confidence'] = line.replace('  置信度:', '').strip()
        
        # 添加最后一个错误样本
        if current_sample:
            error_samples.append(current_sample)
        
        # 保存错误样本
        if error_samples:
            error_txt_path = os.path.join(output_dir, 'error_samples.txt')
            error_json_path = os.path.join(output_dir, 'error_samples.json')
            
            # 保存为可读文本
            with open(error_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"错误样本分析报告\n")
                f.write(f"总错误样本数: {len(error_samples)}\n")
                f.write("="*60 + "\n\n")
                
                for i, sample in enumerate(error_samples, 1):
                    f.write(f"错误样本 {i}:\n")
                    f.write(f"  问题: {sample.get('query', 'N/A')}\n")
                    f.write(f"  真实标签: {sample.get('true_label', 'N/A')}\n")
                    f.write(f"  预测标签: {sample.get('predicted_label', 'N/A')}\n")
                    f.write(f"  置信度: {sample.get('confidence', 'N/A')}\n")
                    f.write("-" * 40 + "\n\n")
            
            # 保存为JSON
            with open(error_json_path, 'w', encoding='utf-8') as f:
                json.dump(error_samples, f, ensure_ascii=False, indent=2)
            
            print(f"错误样本分析已保存:")
            print(f"  文本报告: {error_txt_path}")
            print(f"  JSON数据: {error_json_path}")
            print(f"  总错误样本数: {len(error_samples)}")
        else:
            print("未找到错误样本")
            
    except Exception as e:
        print(f"提取错误样本时出错: {e}")

def main():
    # 内置配置 - 可以直接修改这些路径
    DEFAULT_CONFIG = {
        "model_path": "training_data/models/modernbert_complexity_classifier/run_20250725_211430/final_model",
        "test_data_paths": [
            "training_data/test1/zero_hop_test1.json",
            "training_data/test1/one_hop_test1.json",
            "training_data/test1/mutil_hop_test1.json"
        ],
        "output_file": "training_data/test_results/comprehensive_test_results.json",
        "output_dir": "training_data/test_results",
        "enable_visualization": True,
        "enable_interactive": False,  # 设置为True可以启用交互式测试
        "use_mixed_testing": True,  # 使用混合测试避免数据泄露
        "show_detailed_results": True  # 显示详细的预测结果
    }
    
    # 检查命令行参数，如果提供了参数则使用参数，否则使用默认配置
    import sys
    if len(sys.argv) > 1:
        # 使用命令行参数模式
        parser = argparse.ArgumentParser(description="测试ModernBERT复杂度分类器")
        parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
        parser.add_argument("--test_data", type=str, nargs='+', help="测试数据路径列表")
        parser.add_argument("--interactive", action="store_true", help="交互式测试模式")
        parser.add_argument("--output", type=str, help="结果输出文件")
        parser.add_argument("--visualize", action="store_true", help="生成可视化图表")
        parser.add_argument("--output_dir", type=str, default="test_results", help="输出目录")
        
        args = parser.parse_args()
        
        config = {
            "model_path": args.model_path,
            "test_data_paths": args.test_data if args.test_data else [],
            "output_file": args.output,
            "output_dir": args.output_dir,
            "enable_visualization": args.visualize,
            "enable_interactive": args.interactive,
            "use_mixed_testing": True,
            "show_detailed_results": True
        }
    else:
        # 使用默认配置
        config = DEFAULT_CONFIG
        print("="*80)
        print("🚀 使用内置配置启动ModernBERT复杂度分类器测试")
        print("="*80)
        print(f"📁 模型路径: {config['model_path']}")
        print(f"📊 测试数据: {', '.join(config['test_data_paths'])}")
        print(f"💾 输出文件: {config['output_file']}")
        print(f"📈 可视化: {'启用' if config['enable_visualization'] else '禁用'}")
        print(f"🎯 交互模式: {'启用' if config['enable_interactive'] else '禁用'}")
        print(f"🔄 混合测试: {'启用' if config['use_mixed_testing'] else '禁用'}")
        print(f"🔍 详细结果: {'启用' if config['show_detailed_results'] else '禁用'}")
        print("="*80)
    
    # 创建测试器
    tester = ComplexityClassifierTester(config["model_path"])
    
    # 加载模型
    if not tester.load_model():
        logger.error("模型加载失败，退出")
        return
    
    # 交互式测试
    if config["enable_interactive"]:
        tester.interactive_test()
        return
    
    # 数据集测试
    if config["test_data_paths"]:
        if config["use_mixed_testing"]:
            # 使用混合测试方法，避免数据泄露
            logger.info("🔄 使用混合测试方法，确保测试的真实性")
            results = tester.evaluate_mixed_dataset(
                config["test_data_paths"], 
                show_details=config["show_detailed_results"]
            )
            
            # 打印结果
            print_comprehensive_results(results)
            
            # 生成可视化
            if config["enable_visualization"]:
                viz_dir = os.path.join(config["output_dir"], "mixed_test_data")
                tester.create_visualizations(results, viz_dir)
            
            # 保存结果
            if config["output_file"]:
                # 转换numpy类型为Python原生类型，确保JSON序列化
                def convert_numpy_types(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    return obj
                
                output_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model_path": config["model_path"],
                    "test_data_paths": config["test_data_paths"],
                    "test_method": "mixed_testing",
                    "results": convert_numpy_types(results)
                }
                
                os.makedirs(os.path.dirname(config["output_file"]), exist_ok=True)
                with open(config["output_file"], 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"混合测试结果已保存到: {config['output_file']}")
            
            # 保存详细的测试日志
            log_file_path = os.path.join(config["output_dir"], "detailed_test_log.txt")
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("ModernBERT复杂度分类器详细测试日志\n")
                f.write("="*80 + "\n")
                f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型路径: {config['model_path']}\n")
                f.write(f"测试数据: {', '.join(config['test_data_paths'])}\n")
                f.write(f"总样本数: {results['test_size']}\n")
                f.write(f"正确预测: {results['correct_predictions']}\n")
                f.write(f"总体准确率: {results['correct_predictions']/results['total_predictions']:.4f}\n")
                f.write("="*80 + "\n\n")
                
                # 写入每个样本的详细结果
                for log_entry in tester.log_buffer:
                    f.write(log_entry + "\n")
                
                f.write("="*80 + "\n")
                f.write("测试完成\n")
                f.write("="*80 + "\n")
            
            logger.info(f"详细测试日志已保存到: {log_file_path}")
            
            # 提取错误样本
            extract_error_samples(log_file_path, config["output_dir"])
            
            # 显示类别分布和真实性能
            print("\n" + "="*80)
            print("📊 真实测试性能分析")
            print("="*80)
            print(f"总样本数: {results['test_size']}")
            print(f"正确预测: {results['correct_predictions']}")
            print(f"总体准确率: {results['correct_predictions']/results['total_predictions']:.4f}")
            print(f"\n类别分布:")
            for label, count in results['label_distribution'].items():
                print(f"  {label}: {count} 样本")
            
        else:
            # 原有的分别测试方法
            all_results = {}
            
            for test_data_path in config["test_data_paths"]:
                logger.info(f"开始测试数据集: {test_data_path}")
                results = tester.evaluate_on_dataset(test_data_path)
                all_results[results['dataset_name']] = results
                
                # 打印结果
                print_comprehensive_results(results)
                
                # 生成可视化
                if config["enable_visualization"]:
                    viz_dir = os.path.join(config["output_dir"], results['dataset_name'])
                    tester.create_visualizations(results, viz_dir)
            
            # 保存综合结果
            if config["output_file"]:
                # 转换numpy类型为Python原生类型，确保JSON序列化
                def convert_numpy_types(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    return obj
                
                output_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model_path": config["model_path"],
                    "test_data_paths": config["test_data_paths"],
                    "test_method": "separate_testing",
                    "results": convert_numpy_types(all_results)
                }
                
                os.makedirs(os.path.dirname(config["output_file"]), exist_ok=True)
                with open(config["output_file"], 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"综合结果已保存到: {config['output_file']}")
            
            # 数据集对比
            if len(all_results) > 1:
                print("\n" + "="*80)
                print("📊 数据集性能对比")
                print("="*80)
                
                comparison_data = []
                for dataset_name, results in all_results.items():
                    metrics = results['metrics']
                    comparison_data.append({
                        'Dataset': dataset_name,
                        'Accuracy': metrics['overall_accuracy'],
                        'Macro F1': metrics['macro_metrics']['f1'],
                        'Weighted F1': metrics['weighted_metrics']['f1'],
                        'Samples': results['test_size']
                    })
                
                df = pd.DataFrame(comparison_data)
                print(df.to_string(index=False))
        
        print("\n" + "="*80)
        print("✅ 测试完成！")
        print("="*80)
    
    else:
        logger.info("未指定测试数据，进入交互式模式")
        tester.interactive_test()

if __name__ == "__main__":
    main() 