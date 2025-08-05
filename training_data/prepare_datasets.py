#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据集准备脚本 - 合并并分割数据集

本脚本用于:
1. 加载三个复杂度数据集(zero_hop, one_hop, multi_hop)
2. 合并数据集
3. 平衡各类别数据
4. 分割为训练集、验证集和测试集
5. 保存处理后的数据集
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any, Tuple
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("prepare-datasets")

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    加载数据集
    
    Args:
        file_path: 数据集文件路径
        
    Returns:
        数据列表
    """
    try:
        logger.info(f"加载数据集: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        logger.info(f"成功加载 {len(data)} 条数据")
    return data
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return []

def save_dataset(data: List[Dict[str, Any]], file_path: str) -> bool:
    """
    保存数据集
    
    Args:
        data: 数据列表
        file_path: 保存路径
        
    Returns:
        是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        logger.info(f"保存数据集到: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"成功保存 {len(data)} 条数据")
        return True
    except Exception as e:
        logger.error(f"保存数据集失败: {e}")
        return False

def balance_dataset(data: List[Dict[str, Any]], label_field: str, max_samples_per_class: int = None) -> List[Dict[str, Any]]:
    """
    平衡各类别数据
    
    Args:
        data: 数据列表
        label_field: 标签字段名
        max_samples_per_class: 每个类别的最大样本数，如果为None则使用最小类别的样本数
        
    Returns:
        平衡后的数据列表
    """
    # 按类别分组
    class_data = {}
    for item in data:
        label = str(item.get(label_field, ""))
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(item)
    
    # 计算每个类别的样本数
    class_counts = {label: len(items) for label, items in class_data.items()}
    logger.info(f"类别分布: {class_counts}")
    
    # 确定每个类别的目标样本数
    if max_samples_per_class is None:
        # 使用最小类别的样本数
        min_count = min(class_counts.values())
        max_samples_per_class = min_count
    
    # 平衡数据
    balanced_data = []
    for label, items in class_data.items():
        # 随机打乱
        random.shuffle(items)
        # 选择指定数量的样本
        selected_items = items[:max_samples_per_class]
        balanced_data.extend(selected_items)
        logger.info(f"类别 '{label}': 选择了 {len(selected_items)} 条数据")
        
    # 随机打乱
    random.shuffle(balanced_data)
    
    return balanced_data

def split_dataset(data: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    分割数据集
    
    Args:
        data: 数据列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        (训练集, 验证集, 测试集)
    """
    # 随机打乱
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 计算分割点
    train_end = int(len(shuffled_data) * train_ratio)
    val_end = train_end + int(len(shuffled_data) * val_ratio)
    
    # 分割数据
    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]
    test_data = shuffled_data[val_end:]
    
    logger.info(f"数据分割: 训练集 {len(train_data)}, 验证集 {len(val_data)}, 测试集 {len(test_data)}")
    
    return train_data, val_data, test_data

def compute_stats(train_data: List[Dict[str, Any]], val_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]], label_field: str) -> Dict[str, Any]:
    """
    计算数据集统计信息
    
    Args:
        train_data: 训练集
        val_data: 验证集
        test_data: 测试集
        label_field: 标签字段名
        
    Returns:
        统计信息
    """
    # 合并所有数据
    all_data = train_data + val_data + test_data
    
    # 计算总样本数
    total_samples = len(all_data)
    
    # 计算类别分布
    class_counts = {}
    for item in all_data:
        label = str(item.get(label_field, ""))
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    # 计算训练集类别分布
    train_class_counts = {}
    for item in train_data:
        label = str(item.get(label_field, ""))
        if label not in train_class_counts:
            train_class_counts[label] = 0
        train_class_counts[label] += 1
    
    # 计算验证集类别分布
    val_class_counts = {}
    for item in val_data:
        label = str(item.get(label_field, ""))
        if label not in val_class_counts:
            val_class_counts[label] = 0
        val_class_counts[label] += 1
    
    # 计算测试集类别分布
    test_class_counts = {}
    for item in test_data:
        label = str(item.get(label_field, ""))
        if label not in test_class_counts:
            test_class_counts[label] = 0
        test_class_counts[label] += 1
    
    return {
        "total_samples": total_samples,
        "class_distribution": class_counts,
        "train_samples": len(train_data),
        "train_class_distribution": train_class_counts,
        "val_samples": len(val_data),
        "val_class_distribution": val_class_counts,
        "test_samples": len(test_data),
        "test_class_distribution": test_class_counts
    }

def main():
    parser = argparse.ArgumentParser(description="准备复杂度分类数据集")
    parser.add_argument("--zero_hop", type=str, default="training_data/datasets/data_zero_hop.json", help="zero_hop数据集路径")
    parser.add_argument("--one_hop", type=str, default="training_data/datasets/data_one_hop.json", help="one_hop数据集路径")
    parser.add_argument("--multi_hop", type=str, default="training_data/datasets/data_multi_hop.json", help="multi_hop数据集路径")
    parser.add_argument("--output_dir", type=str, default="training_data/processed", help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--balance", action="store_true", help="是否平衡各类别数据")
    parser.add_argument("--max_samples", type=int, default=None, help="每个类别的最大样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--label_field", type=str, default="complexity", help="标签字段名")
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 加载数据集
    zero_hop_data = load_dataset(args.zero_hop)
    one_hop_data = load_dataset(args.one_hop)
    multi_hop_data = load_dataset(args.multi_hop)
    
    # 合并数据集
    all_data = zero_hop_data + one_hop_data + multi_hop_data
    logger.info(f"合并后共有 {len(all_data)} 条数据")
    
    # 平衡数据集
    if args.balance:
        all_data = balance_dataset(all_data, args.label_field, args.max_samples)
        logger.info(f"平衡后共有 {len(all_data)} 条数据")
    
    # 分割数据集
    train_data, val_data, test_data = split_dataset(all_data, args.train_ratio, args.val_ratio)
    
    # 保存数据集
    os.makedirs(args.output_dir, exist_ok=True)
    save_dataset(train_data, os.path.join(args.output_dir, "train_data.json"))
    save_dataset(val_data, os.path.join(args.output_dir, "val_data.json"))
    save_dataset(test_data, os.path.join(args.output_dir, "test_data.json"))
    
    # 计算统计信息
    stats = compute_stats(train_data, val_data, test_data, args.label_field)
    save_dataset(stats, os.path.join(args.output_dir, "data_statistics.json"))
    
    logger.info("数据集准备完成")
    logger.info(f"总样本数: {stats['total_samples']}")
    logger.info(f"类别分布: {stats['class_distribution']}")
    logger.info(f"训练集: {stats['train_samples']} 条")
    logger.info(f"验证集: {stats['val_samples']} 条")
    logger.info(f"测试集: {stats['test_samples']} 条")

if __name__ == "__main__":
    main() 