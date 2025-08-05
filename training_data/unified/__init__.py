"""
统一训练模块 - 整合数据处理、训练和银标签生成功能

本模块整合了训练数据文件夹中的核心功能，提供一致的接口。
"""

from ..data_processor import DataProcessor, DataProcessorConfig, prepare_dataset
from ..trainer import ComplexityClassifierTrainer, TrainingConfig, train_complexity_classifier
from ..silver_labeling import SilverLabelGenerator, SilverLabelConfig, generate_silver_labels

__all__ = [
    'DataProcessor',
    'DataProcessorConfig',
    'prepare_dataset',
    'ComplexityClassifierTrainer',
    'TrainingConfig',
    'train_complexity_classifier',
    'SilverLabelGenerator',
    'SilverLabelConfig',
    'generate_silver_labels'
] 