#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查测试结果图片文件的完整性
"""

import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

def check_image_file(file_path):
    """检查单个图片文件"""
    try:
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}"
        
        # 尝试打开图片
        with Image.open(file_path) as img:
            width, height = img.size
            format_name = img.format
            mode = img.mode
            
        return True, f"✓ {file_path} - 尺寸: {width}x{height}, 格式: {format_name}, 模式: {mode}"
    
    except Exception as e:
        return False, f"✗ {file_path} - 错误: {str(e)}"

def main():
    """主函数"""
    print("=" * 60)
    print("测试结果图片文件检查")
    print("=" * 60)
    
    # 图片文件列表
    image_files = [
        "test_data/accuracy_vs_confidence_test_data.png",
        "test_data/class_performance_test_data.png", 
        "test_data/confidence_distribution_test_data.png",
        "test_data/confusion_matrix_test_data.png"
    ]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    all_good = True
    for img_file in image_files:
        full_path = os.path.join(current_dir, img_file)
        is_valid, message = check_image_file(full_path)
        print(message)
        if not is_valid:
            all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("✓ 所有图片文件检查通过！")
        print("\n您可以通过以下方式查看图片：")
        print("1. 在浏览器中打开 view_test_results.html 文件")
        print("2. 直接双击PNG文件使用系统默认图片查看器")
        print("3. 使用图片编辑软件（如Photoshop、GIMP等）")
    else:
        print("✗ 发现图片文件问题，请检查文件路径和完整性")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 