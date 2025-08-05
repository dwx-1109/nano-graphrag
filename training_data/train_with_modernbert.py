#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
复杂度分类器训练脚本 - 使用ModernBERT-large模型

本脚本基于现有的trainer.py，使用ModernBERT-large模型和已处理的数据进行训练。
主要特点：
1. 使用本地ModernBERT-large模型
2. 使用已处理的数据集
3. 支持LoRA高效微调
4. 完整的训练和评估流程
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, Any

# 导入训练器
from trainer import ComplexityClassifierTrainer, TrainingConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("train-modernbert")

def create_modernbert_config(args) -> TrainingConfig:
    """
    创建使用ModernBERT-large的训练配置
    
    Args:
        args: 命令行参数
        
    Returns:
        训练配置
    """
    # 基本配置
    config = TrainingConfig(
        # 使用本地ModernBERT-large模型
        model_name="models/modernbert/answerdotai_ModernBERT-large",
        output_dir=args.output_dir,
        
        # 数据集配置 - 使用已处理的数据
        train_file="training_data/processed/train_data.json",
        val_file="training_data/processed/val_data.json", 
        test_file="training_data/processed/test_data.json",
        text_field="query",
        label_field="complexity",
        max_length=args.max_length,
        
        # 训练参数
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # LoRA配置
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        
        # 早停和验证配置
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        eval_strategy="steps",
        save_strategy="steps",
        
        # 其他配置
        seed=args.seed,
        use_fp16=args.use_fp16,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        device=args.device
    )
    
    # 保存配置
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "training_config.json")
    config.save(config_path)
    logger.info(f"配置已保存到: {config_path}")
    
    return config

def print_dataset_info():
    """打印数据集信息"""
    try:
        # 读取数据统计信息
        stats_path = "training_data/processed/data_statistics.json"
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            logger.info("=== 数据集信息 ===")
            logger.info(f"总样本数: {stats['total_samples']}")
            logger.info(f"类别分布: {stats['class_distribution']}")
            logger.info(f"训练集: {stats['train_samples']} 条")
            logger.info(f"验证集: {stats['val_samples']} 条")
            logger.info(f"测试集: {stats['test_samples']} 条")
            logger.info("==================")
        else:
            logger.warning("未找到数据统计文件")
    except Exception as e:
        logger.error(f"读取数据集信息失败: {e}")

def check_model_availability():
    """检查模型可用性"""
    model_path = "models/modernbert/answerdotai_ModernBERT-large"
    if os.path.exists(model_path):
        logger.info(f"✓ ModernBERT-large模型已找到: {model_path}")
        
        # 检查关键文件
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"模型文件不完整，缺少: {missing_files}")
            return False
        else:
            logger.info("✓ 模型文件完整")
            return True
    else:
        logger.error(f"✗ ModernBERT-large模型未找到: {model_path}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ModernBERT复杂度分类器训练脚本")
    
    # 基本参数
    parser.add_argument("--output_dir", type=str, 
                        default="training_data/models/modernbert_complexity_classifier",
                        help="模型输出目录")
    
    # 添加时间戳到输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args, _ = parser.parse_known_args()
    
    # 如果用户没有指定自定义输出目录，则添加时间戳
    if args.output_dir == "training_data/models/modernbert_complexity_classifier":
        args.output_dir = f"training_data/models/modernbert_complexity_classifier_{timestamp}"
    
    # 数据集参数
    parser.add_argument("--max_length", type=int, default=256, 
                        help="最大序列长度")
    
    # 训练参数 - 优化后的默认值
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="批处理大小 (较小的batch_size有助于更好的泛化)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                        help="学习率 (较小的学习率有助于更稳定的训练)")
    parser.add_argument("--num_epochs", type=int, default=15, 
                        help="训练轮次 (增加轮次以提高性能)")
    parser.add_argument("--warmup_ratio", type=float, default=0.15, 
                        help="预热比例 (增加预热比例有助于稳定训练)")
    parser.add_argument("--weight_decay", type=float, default=0.02, 
                        help="权重衰减 (增加权重衰减防止过拟合)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="梯度累积步数 (增加梯度累积以获得更稳定的梯度)")
    
    # 早停和验证参数
    parser.add_argument("--early_stopping_patience", type=int, default=3, 
                        help="早停耐心值 (连续3轮无改善则停止)")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001, 
                        help="早停阈值 (改善幅度小于此值则不计入改善)")
    parser.add_argument("--eval_steps", type=int, default=50, 
                        help="评估步数 (每50步进行一次验证)")
    parser.add_argument("--save_steps", type=int, default=100, 
                        help="保存步数 (每100步保存一次模型)")
    
    # LoRA配置 - 优化后的参数
    parser.add_argument("--use_lora", action="store_true", 
                        help="使用LoRA进行参数高效微调")
    parser.add_argument("--lora_r", type=int, default=32, 
                        help="LoRA秩 (增加秩以提高表达能力)")
    parser.add_argument("--lora_alpha", type=int, default=64, 
                        help="LoRA缩放参数 (通常设置为2*r)")
    parser.add_argument("--lora_dropout", type=float, default=0.15, 
                        help="LoRA dropout (增加dropout防止过拟合)")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, 
                        help="随机种子")
    parser.add_argument("--use_fp16", action="store_true", 
                        help="使用FP16加速训练")
    parser.add_argument("--logging_steps", type=int, default=20, 
                        help="日志步数 (增加日志频率以便监控)")
    parser.add_argument("--device", type=str, default="auto", 
                        help="设备，auto/cuda/cpu")
    
    # 解析参数
    args = parser.parse_args()
    
    logger.info("=== ModernBERT复杂度分类器训练 ===")
    
    # 检查模型可用性
    if not check_model_availability():
        logger.error("模型检查失败，请确保ModernBERT-large模型已正确下载")
        return
    
    # 打印数据集信息
    print_dataset_info()
    
    # 创建训练配置
    logger.info("创建训练配置...")
    config = create_modernbert_config(args)
    
    # 初始化训练器
    logger.info("初始化训练器...")
    trainer = ComplexityClassifierTrainer(config)
    
    # 开始训练
    logger.info("开始训练模型...")
    logger.info(f"使用模型: {config.model_name}")
    logger.info(f"输出目录: {config.output_dir}")
    logger.info(f"批处理大小: {config.batch_size}")
    logger.info(f"学习率: {config.learning_rate}")
    logger.info(f"训练轮次: {config.num_epochs}")
    logger.info(f"使用LoRA: {config.use_lora}")
    logger.info(f"使用FP16: {config.use_fp16}")
    
    try:
        model_path = trainer.train()
        
        # 评估模型
        logger.info("开始评估模型...")
        eval_results = trainer.evaluate(model_path)
        
        # 保存评估结果
        eval_path = os.path.join(config.output_dir, "evaluation_results.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        logger.info(f"评估结果已保存到: {eval_path}")
        
        # 保存训练总结
        summary = {
            "model_path": model_path,
            "eval_results": eval_results,
            "config": {
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "use_lora": config.use_lora,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "early_stopping_patience": config.early_stopping_patience
            },
            "timestamp": timestamp
        }
        
        summary_path = os.path.join(config.output_dir, "training_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"训练总结已保存到: {summary_path}")
        
        logger.info("训练和评估完成!")
        logger.info(f"模型保存在: {model_path}")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 