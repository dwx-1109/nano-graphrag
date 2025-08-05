"""
统一训练流水线 - 提供端到端的训练流程

本模块提供了一个完整的训练流水线，包括：
1. 数据准备和处理
2. 银标签生成（可选）
3. 模型训练和评估
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

from ..data_processor import DataProcessor, DataProcessorConfig, prepare_dataset
from ..trainer import ComplexityClassifierTrainer, TrainingConfig, train_complexity_classifier
from ..silver_labeling import SilverLabelGenerator, SilverLabelConfig, generate_silver_labels

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("training-pipeline")


@dataclass
class PipelineConfig:
    """训练流水线配置"""
    
    # 基本配置
    pipeline_name: str = "complexity_classifier"
    output_dir: str = "models/complexity_classifier"
    
    # 数据处理配置
    data_config: Dict[str, Any] = field(default_factory=lambda: {
        "datasets_dir": "datasets",
        "processed_data_dir": "processed",
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1
    })
    
    # 银标签生成配置
    use_silver_labeling: bool = False
    silver_config: Dict[str, Any] = field(default_factory=lambda: {
        "api_type": "openai",
        "api_model": "qwen-turbo",
        "batch_size": 10,
        "max_retries": 3
    })
    
    # 训练配置
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        "model_name": "answerdotai/ModernBERT-large",
        "batch_size": 16,
        "num_epochs": 5,
        "learning_rate": 2e-5,
        "use_lora": True
    })
    
    # 数据集配置
    dataset_names: List[str] = field(default_factory=lambda: ["data_zero_hop", "data_one_hop", "data_multi_hop"])
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save(self, path: str):
        """保存配置到JSON文件"""
        # 转换为字典
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        
        # 保存到文件
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'PipelineConfig':
        """从JSON文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)


class TrainingPipeline:
    """训练流水线"""
    
    def __init__(self, config: PipelineConfig = None):
        """
        初始化训练流水线
        
        Args:
            config: 流水线配置，如果为None则使用默认配置
        """
        self.config = config or PipelineConfig()
        
        # 初始化组件
        self.data_processor = None
        self.silver_generator = None
        self.trainer = None
        
        # 初始化结果
        self.dataset_paths = None
        self.silver_stats = None
        self.training_results = None
    
    def prepare_data(self) -> Dict[str, str]:
        """
        准备数据
        
        Returns:
            包含文件路径的字典
        """
        logger.info("开始准备数据")
        
        # 创建数据处理器配置
        data_config = DataProcessorConfig(**self.config.data_config)
        
        # 创建数据处理器
        self.data_processor = DataProcessor(data_config)
        
        # 准备数据集
        self.dataset_paths = self.data_processor.prepare_dataset(
            self.config.dataset_names,
            output_prefix=self.config.pipeline_name
        )
        
        logger.info(f"数据准备完成: {self.dataset_paths}")
        return self.dataset_paths
    
    def generate_silver_labels(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        生成银标签
        
        Returns:
            (标注数据, 统计信息)
        """
        if not self.config.use_silver_labeling:
            logger.info("跳过银标签生成")
            return [], {}
        
        logger.info("开始生成银标签")
        
        # 创建银标签配置
        silver_config = SilverLabelConfig(**self.config.silver_config)
        
        # 如果已经准备了数据，使用准备好的数据
        if self.dataset_paths:
            silver_config.input_file = self.dataset_paths.get("train")
            silver_config.output_file = os.path.join(
                self.config.data_config.get("processed_data_dir", "processed"),
                f"{self.config.pipeline_name}_silver.json"
            )
        
        # 创建银标签生成器
        self.silver_generator = SilverLabelGenerator(silver_config)
        
        # 生成银标签
        labeled_data, stats = self.silver_generator.run()
        
        # 保存统计信息
        self.silver_stats = stats
        
        logger.info(f"银标签生成完成: {stats}")
        return labeled_data, stats
    
    def train_model(self) -> str:
        """
        训练模型
        
        Returns:
            模型路径
        """
        logger.info("开始训练模型")
        
        # 创建训练配置
        training_config = TrainingConfig(**self.config.training_config)
        
        # 如果已经准备了数据，使用准备好的数据
        if self.dataset_paths:
            training_config.train_file = self.dataset_paths.get("train")
            training_config.val_file = self.dataset_paths.get("val")
            training_config.test_file = self.dataset_paths.get("test")
        
        # 设置输出目录
        training_config.output_dir = self.config.output_dir
        
        # 创建训练器
        self.trainer = ComplexityClassifierTrainer(training_config)
        
        # 训练模型
        model_path = self.trainer.train()
        
        # 评估模型
        metrics = self.trainer.evaluate(model_path)
        
        # 保存训练结果
        self.training_results = {
            "model_path": model_path,
            "metrics": metrics
        }
        
        logger.info(f"模型训练完成: {model_path}")
        logger.info(f"评估指标: {metrics}")
        
        return model_path
    
    def run(self) -> Dict[str, Any]:
        """
        运行完整流水线
        
        Returns:
            流水线结果
        """
        logger.info(f"开始运行训练流水线: {self.config.pipeline_name}")
        
        # 准备数据
        dataset_paths = self.prepare_data()
        
        # 生成银标签（可选）
        if self.config.use_silver_labeling:
            labeled_data, silver_stats = self.generate_silver_labels()
        
        # 训练模型
        model_path = self.train_model()
        
        # 汇总结果
        results = {
            "pipeline_name": self.config.pipeline_name,
            "dataset_paths": dataset_paths,
            "model_path": model_path,
            "training_results": self.training_results
        }
        
        # 如果使用了银标签，添加相关结果
        if self.config.use_silver_labeling:
            results["silver_stats"] = self.silver_stats
        
        # 保存结果
        results_path = os.path.join(self.config.output_dir, f"{self.config.pipeline_name}_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"流水线运行完成，结果保存在: {results_path}")
        
        return results


def run_training_pipeline(config_path: str = None) -> Dict[str, Any]:
    """
    运行训练流水线的便捷函数
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
        
    Returns:
        流水线结果
    """
    # 加载配置
    if config_path and os.path.exists(config_path):
        logger.info(f"加载配置: {config_path}")
        config = PipelineConfig.load(config_path)
    else:
        logger.info("使用默认配置")
        config = PipelineConfig()
    
    # 创建流水线
    pipeline = TrainingPipeline(config)
    
    # 运行流水线
    return pipeline.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行训练流水线")
    parser.add_argument("--config", type=str, help="配置文件路径")
    args = parser.parse_args()
    
    results = run_training_pipeline(args.config)
    
    logger.info("训练流水线运行完成")
    logger.info(f"模型路径: {results.get('model_path')}")
    logger.info(f"评估指标: {results.get('training_results', {}).get('metrics', {})}") 