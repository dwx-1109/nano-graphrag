"""
统一训练流水线运行脚本

提供命令行接口，用于运行完整的训练流水线。
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any

from .pipeline import TrainingPipeline, PipelineConfig, run_training_pipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("training-run")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行复杂度分类器训练流水线")
    
    # 配置文件
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    # 基本配置
    parser.add_argument("--name", type=str, help="流水线名称")
    parser.add_argument("--output", type=str, help="输出目录")
    
    # 数据集配置
    parser.add_argument("--datasets", nargs="+", help="数据集名称列表")
    
    # 模型配置
    parser.add_argument("--model", type=str, help="基础模型名称")
    parser.add_argument("--batch-size", type=int, help="批次大小")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, help="学习率")
    parser.add_argument("--use-lora", action="store_true", help="使用LoRA")
    
    # 银标签配置
    parser.add_argument("--use-silver", action="store_true", help="使用银标签生成")
    parser.add_argument("--api-type", type=str, choices=["openai", "qwen", "custom"], help="API类型")
    parser.add_argument("--api-model", type=str, help="API模型")
    
    # 其他选项
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    return parser.parse_args()


def create_config_from_args(args) -> PipelineConfig:
    """从命令行参数创建配置"""
    # 基本配置
    config_kwargs = {}
    
    if args.name:
        config_kwargs["pipeline_name"] = args.name
    
    if args.output:
        config_kwargs["output_dir"] = args.output
    
    # 数据集配置
    if args.datasets:
        config_kwargs["dataset_names"] = args.datasets
    
    # 训练配置
    training_config = {}
    
    if args.model:
        training_config["model_name"] = args.model
    
    if args.batch_size:
        training_config["batch_size"] = args.batch_size
    
    if args.epochs:
        training_config["num_epochs"] = args.epochs
    
    if args.learning_rate:
        training_config["learning_rate"] = args.learning_rate
    
    if args.use_lora:
        training_config["use_lora"] = True
    
    if training_config:
        config_kwargs["training_config"] = training_config
    
    # 银标签配置
    if args.use_silver:
        config_kwargs["use_silver_labeling"] = True
        
        silver_config = {}
        
        if args.api_type:
            silver_config["api_type"] = args.api_type
        
        if args.api_model:
            silver_config["api_model"] = args.api_model
        
        if silver_config:
            config_kwargs["silver_config"] = silver_config
    
    # 创建配置
    return PipelineConfig(**config_kwargs)


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 如果提供了配置文件，从文件加载配置
        if args.config and os.path.exists(args.config):
            logger.info(f"从文件加载配置: {args.config}")
            config = PipelineConfig.load(args.config)
            
            # 使用命令行参数覆盖配置文件中的值
            override_config = create_config_from_args(args)
            
            # 合并配置
            for key, value in override_config.__dict__.items():
                if value is not None and key in config.__dict__:
                    config.__dict__[key] = value
        else:
            # 从命令行参数创建配置
            config = create_config_from_args(args)
        
        # 创建流水线
        pipeline = TrainingPipeline(config)
        
        # 运行流水线
        results = pipeline.run()
        
        # 打印结果
        logger.info("训练流水线运行完成")
        logger.info(f"模型路径: {results.get('model_path')}")
        logger.info(f"评估指标: {results.get('training_results', {}).get('metrics', {})}")
        
        return 0
        
    except Exception as e:
        logger.error(f"训练流水线运行失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 