#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复标注格式不一致的问题
"""

import json
import os

def fix_label_format(input_file, output_file):
    """修复标注格式，统一标签格式"""
    print(f"正在修复 {input_file}...")
    
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计修复前的分布
    before_dist = {}
    for item in data:
        label = item.get('complexity', 'unknown')
        before_dist[label] = before_dist.get(label, 0) + 1
    
    print("修复前分布:", before_dist)
    
    # 修复格式
    fixed_count = 0
    for item in data:
        if item.get('complexity') == 'one_hop':
            item['complexity'] = 'one-hop'
            fixed_count += 1
    
    # 统计修复后的分布
    after_dist = {}
    for item in data:
        label = item.get('complexity', 'unknown')
        after_dist[label] = after_dist.get(label, 0) + 1
    
    print("修复后分布:", after_dist)
    print(f"修复了 {fixed_count} 个标签")
    
    # 保存修复后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"已保存到: {output_file}")

if __name__ == "__main__":
    input_file = "training_data/datasets/data_one_hop_labeled_v2.json"
    output_file = "training_data/datasets/data_one_hop_labeled_v2_fixed.json"
    
    if os.path.exists(input_file):
        fix_label_format(input_file, output_file)
    else:
        print(f"文件不存在: {input_file}") 