#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从详细测试日志中提取错误样本
"""

import os
import json
from datetime import datetime

def extract_error_samples(log_file_path: str, output_dir: str):
    """从详细测试日志中提取错误样本"""
    print("🔍 开始提取错误样本...")
    
    error_samples = []
    current_sample = {}
    in_sample = False
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 检查是否是错误样本的开始
            if line.startswith('❌ 样本'):
                if in_sample and current_sample:
                    error_samples.append(current_sample)
                current_sample = {}
                in_sample = True
                current_sample['sample_id'] = line
                continue
            
            # 检查是否是正确样本的开始，如果是则结束当前样本
            if line.startswith('✅ 样本'):
                if in_sample and current_sample:
                    error_samples.append(current_sample)
                in_sample = False
                current_sample = {}
                continue
            
            # 如果在错误样本中，收集信息
            if in_sample and line:
                if line.startswith('问题:'):
                    current_sample['question'] = line[3:].strip()
                elif line.startswith('真实标签:'):
                    current_sample['true_label'] = line[5:].strip()
                elif line.startswith('预测标签:'):
                    current_sample['predicted_label'] = line[5:].strip()
                elif line.startswith('置信度:'):
                    current_sample['confidence'] = line[4:].strip()
                elif line.startswith('各类别概率:'):
                    current_sample['probabilities'] = line[7:].strip()
    
    # 添加最后一个样本
    if in_sample and current_sample:
        error_samples.append(current_sample)
    
    # 保存错误样本到文件
    error_file_path = os.path.join(output_dir, "error_samples.txt")
    with open(error_file_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ModernBERT复杂度分类器错误样本分析\n")
        f.write("="*80 + "\n")
        f.write(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"错误样本总数: {len(error_samples)}\n")
        f.write("="*80 + "\n\n")
        
        # 按错误类型分类统计
        error_types = {}
        for sample in error_samples:
            true_label = sample.get('true_label', '')
            predicted_label = sample.get('predicted_label', '')
            error_type = f"{true_label} -> {predicted_label}"
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(sample)
        
        f.write("📊 错误类型统计:\n")
        for error_type, samples in error_types.items():
            f.write(f"  {error_type}: {len(samples)} 个样本\n")
        f.write("\n" + "="*80 + "\n\n")
        
        # 按错误类型分组显示样本
        for error_type, samples in error_types.items():
            f.write(f"🔍 错误类型: {error_type} ({len(samples)} 个样本)\n")
            f.write("-" * 60 + "\n")
            
            for i, sample in enumerate(samples, 1):
                f.write(f"\n{i}. {sample.get('sample_id', '')}\n")
                f.write(f"   问题: {sample.get('question', '')}\n")
                f.write(f"   真实标签: {sample.get('true_label', '')}\n")
                f.write(f"   预测标签: {sample.get('predicted_label', '')}\n")
                f.write(f"   置信度: {sample.get('confidence', '')}\n")
                f.write(f"   各类别概率: {sample.get('probabilities', '')}\n")
            
            f.write("\n" + "="*60 + "\n\n")
    
    # 保存错误样本的JSON格式，便于后续分析
    error_json_path = os.path.join(output_dir, "error_samples.json")
    with open(error_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_error_samples": len(error_samples),
            "error_types": error_types,
            "samples": error_samples
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 错误样本提取完成！")
    print(f"   错误样本总数: {len(error_samples)}")
    print(f"   错误样本文件: {error_file_path}")
    print(f"   错误样本JSON: {error_json_path}")
    
    # 打印错误类型统计
    print("\n" + "="*80)
    print("📊 错误样本分析")
    print("="*80)
    print(f"错误样本总数: {len(error_samples)}")
    print("错误类型分布:")
    for error_type, samples in error_types.items():
        print(f"  {error_type}: {len(samples)} 个样本")
    print("="*80)

if __name__ == "__main__":
    # 设置路径
    log_file_path = "training_data/test_results/detailed_test_log.txt"
    output_dir = "training_data/test_results"
    
    # 检查文件是否存在
    if not os.path.exists(log_file_path):
        print(f"❌ 错误：找不到日志文件 {log_file_path}")
        exit(1)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取错误样本
    extract_error_samples(log_file_path, output_dir) 