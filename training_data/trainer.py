"""
复杂度分类器训练器 - 提供ModernBERT模型的训练和评估功能

本模块整合了固定标签和扩展标签两种训练方式，支持：
- 基于PEFT/LoRA的高效微调
- 训练过程监控和早停
- 模型评估和结果保存
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

# 先导入基础依赖
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        TrainingArguments, 
        Trainer,
        EvalPrediction,
        EarlyStoppingCallback
    )
    from peft import (
        LoraConfig, 
        TaskType, 
        get_peft_model, 
        PeftModel,
        prepare_model_for_kbit_training
    )
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("complexity-trainer")


@dataclass
class ComplexityDataset(Dataset):
    """复杂度分类数据集"""
    
    texts: List[str]
    labels: List[int]
    tokenizer: Any
    max_length: int = 256
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 移除批次维度
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label)
        
        return item


@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 基本配置
    model_name: str = "answerdotai/ModernBERT-large"
    output_dir: str = "models/fine_tuned_complexity_classifier"
    label_mapping: Dict[str, int] = field(default_factory=lambda: {
        "zero_hop": 0, 
        "one_hop": 1, 
        "multi_hop": 2
    })
    
    # 数据集配置
    train_file: str = "processed/train_data.json"
    val_file: str = "processed/val_data.json"
    test_file: str = "processed/test_data.json"
    text_field: str = "query"
    label_field: str = "complexity"
    max_length: int = 256
    
    # 训练参数
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
    # LoRA配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # 其他配置
    seed: int = 42
    use_fp16: bool = False
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # 早停配置
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    eval_strategy: str = "steps"  # "steps", "epoch"
    save_strategy: str = "steps"  # "steps", "epoch"
    
    def __post_init__(self):
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 确保标签映射键为字符串，值为整数
        self.label_mapping = {str(k): int(v) for k, v in self.label_mapping.items()}
        
        # 创建反向映射
        self.id2label = {v: k for k, v in self.label_mapping.items()}
        
        # 设置设备
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def save(self, path: str):
        """保存配置到JSON文件"""
        # 转换为字典
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        
        # 保存到文件
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """从JSON文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)


class ComplexityClassifierTrainer:
    """复杂度分类器训练器"""
    
    def __init__(self, config: TrainingConfig = None):
        """
        初始化训练器
        
        Args:
            config: 训练配置，如果为None则使用默认配置
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("请安装必要的依赖: pip install torch transformers peft scikit-learn")
        
        self.config = config or TrainingConfig()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # 设置随机种子
        self._set_seed(self.config.seed)
        
        # 初始化分词器
        self._init_tokenizer()
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        import random
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _init_tokenizer(self):
        """初始化分词器"""
        logger.info(f"加载分词器: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
    
    def _init_model(self):
        """初始化模型"""
        logger.info(f"加载基础模型: {self.config.model_name}")
        
        # 获取标签数量
        num_labels = len(self.config.label_mapping)
        
        # 加载预训练模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=num_labels,
            id2label=self.config.id2label,
            label2id=self.config.label_mapping,
            device_map=self.config.device if self.config.device != "auto" else None,
            attn_implementation="eager",  # 禁用Flash Attention
            torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32
        )
        
        # 如果使用LoRA，应用PEFT配置
        if self.config.use_lora:
            logger.info("应用LoRA配置")
            
            # 准备模型进行训练
            self.model = prepare_model_for_kbit_training(self.model)
            
            # 创建LoRA配置 - 修复目标模块
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                target_modules=[
                    "query", "key", "value", "dense",
                    "intermediate.dense", "output.dense"
                ],
                modules_to_save=["classifier"],
                inference_mode=False,
            )
            
            # 应用LoRA
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
    
    def _load_dataset(self, file_path: str) -> Tuple[List[str], List[int]]:
        """
        加载数据集
        
        Args:
            file_path: 数据集文件路径
            
        Returns:
            (文本列表, 标签列表)
        """
        logger.info(f"加载数据集: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = []
            labels = []
            
            for item in data:
                # 获取文本和标签
                text = item.get(self.config.text_field, "")
                label_str = str(item.get(self.config.label_field, ""))
                
                # 如果标签不在映射中，跳过
                if label_str not in self.config.label_mapping:
                    logger.warning(f"标签 '{label_str}' 不在映射中，跳过")
                    continue
                
                # 添加到列表
                texts.append(text)
                labels.append(self.config.label_mapping[label_str])
            
            logger.info(f"加载了 {len(texts)} 条数据")
            return texts, labels
            
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            return [], []
    
    def prepare_datasets(self):
        """准备训练、验证和测试数据集"""
        logger.info("准备数据集")
        
        # 加载训练集
        train_texts, train_labels = self._load_dataset(self.config.train_file)
        self.train_dataset = ComplexityDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
        
        # 加载验证集
        val_texts, val_labels = self._load_dataset(self.config.val_file)
        self.val_dataset = ComplexityDataset(
            texts=val_texts,
            labels=val_labels,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
        
        # 加载测试集
        test_texts, test_labels = self._load_dataset(self.config.test_file)
        self.test_dataset = ComplexityDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
    
    def _compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            eval_pred: 评估预测结果
            
        Returns:
            指标字典
        """
        predictions = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro"),
            "f1_weighted": f1_score(labels, predictions, average="weighted"),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted")
        }
    
    def train(self):
        """训练模型"""
        logger.info("开始训练")
        
        # 初始化模型
        self._init_model()
        
        # 准备数据集
        self.prepare_datasets()
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.config.output_dir, f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建日志目录
        logs_dir = "training_data/logs"
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, f"training_log_{timestamp}.txt")
        
        # 配置文件日志处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"训练日志将保存到: {log_file}")
        
        # 创建训练参数 - 增强版配置
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            # 评估和保存策略
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=5,
            # 模型选择
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            # 性能优化
            fp16=self.config.use_fp16,
            report_to="none",
            remove_unused_columns=False,
            # 学习率调度
            lr_scheduler_type="cosine",
            # 优化器配置
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            # 详细日志配置
            logging_dir=os.path.join(output_dir, "logs"),
            logging_first_step=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
        )
        
        # 创建训练器
        callbacks = []
        if hasattr(self.config, 'early_stopping_patience') and self.config.early_stopping_patience > 0:
            try:
                early_stopping = EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
                callbacks.append(early_stopping)
                logger.info(f"已启用早停机制，耐心值: {self.config.early_stopping_patience}")
            except Exception as e:
                logger.warning(f"早停回调创建失败: {e}")
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self._compute_metrics,
            tokenizer=self.tokenizer,
            callbacks=callbacks
        )
        
        # 记录训练开始时间
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("🚀 开始训练ModernBERT复杂度分类器")
        logger.info("=" * 80)
        logger.info(f"📅 训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"🎯 训练目标: 复杂度分类 (zero_hop, one_hop, multi_hop)")
        logger.info(f"📊 预计训练轮次: {self.config.num_epochs}")
        logger.info(f"⏱️  早停耐心值: {self.config.early_stopping_patience}")
        logger.info(f"📈 评估策略: {self.config.eval_strategy}")
        logger.info(f"🔄 每 {self.config.eval_steps} 步进行一次验证")
        logger.info(f"💾 每 {self.config.save_steps} 步保存一次模型")
        logger.info(f"📝 训练样本数: {len(self.train_dataset)}")
        logger.info(f"🔍 验证样本数: {len(self.val_dataset)}")
        logger.info(f"📊 总训练步数: {len(self.train_dataset) // self.config.batch_size * self.config.num_epochs}")
        logger.info(f"⚙️  批处理大小: {self.config.batch_size}")
        logger.info(f"📚 学习率: {self.config.learning_rate}")
        logger.info(f"🔧 使用LoRA: {self.config.use_lora}")
        logger.info(f"🚀 使用FP16: {self.config.use_fp16}")
        if self.config.use_lora:
            logger.info(f"🎛️  LoRA参数: r={self.config.lora_r}, alpha={self.config.lora_alpha}, dropout={self.config.lora_dropout}")
        logger.info("=" * 80)
        
        # 开始训练
        train_result = self.trainer.train()
        
        # 计算训练时间
        training_time = time.time() - start_time
        training_minutes = training_time / 60
        
        logger.info("=" * 80)
        logger.info("✅ 训练完成!")
        logger.info(f"⏱️  总耗时: {training_minutes:.2f} 分钟")
        logger.info(f"📊 最终训练损失: {train_result.metrics.get('train_loss', 0):.4f}")
        logger.info("=" * 80)
        
        # 保存最终模型
        final_model_path = os.path.join(output_dir, "final_model")
        self.trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        # 保存标签映射
        with open(os.path.join(final_model_path, "label_mapping.json"), 'w', encoding='utf-8') as f:
            json.dump({
                "label2id": self.config.label_mapping,
                "id2label": self.config.id2label
            }, f, indent=2, ensure_ascii=False)
        
        # 保存训练结果
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        # 保存训练参数
        self.trainer.save_state()
        
        # 创建详细的README文件
        with open(os.path.join(final_model_path, "README.md"), 'w', encoding='utf-8') as f:
            f.write(f"# 复杂度分类器模型\n\n")
            f.write(f"训练时间: {timestamp}\n")
            f.write(f"训练耗时: {training_minutes:.2f} 分钟\n\n")
            f.write(f"基础模型: {self.config.model_name}\n")
            f.write(f"训练轮次: {self.config.num_epochs}\n")
            f.write(f"批处理大小: {self.config.batch_size}\n")
            f.write(f"学习率: {self.config.learning_rate}\n")
            f.write(f"使用LoRA: {self.config.use_lora}\n")
            f.write(f"使用FP16: {self.config.use_fp16}\n\n")
            f.write(f"早停配置:\n")
            f.write(f"- 耐心值: {self.config.early_stopping_patience}\n")
            f.write(f"- 阈值: {self.config.early_stopping_threshold}\n\n")
            f.write(f"标签映射:\n")
            for label, idx in self.config.label_mapping.items():
                f.write(f"- {label}: {idx}\n")
            f.write("\n")
            f.write(f"训练指标:\n")
            for key, value in metrics.items():
                f.write(f"- {key}: {value}\n")
        
        logger.info(f"💾 模型保存在: {final_model_path}")
        
        # 移除文件日志处理器
        logger.removeHandler(file_handler)
        
        return final_model_path
    
    def evaluate(self, model_path: str = None):
        """
        评估模型
        
        Args:
            model_path: 模型路径，如果为None则使用当前模型
            
        Returns:
            评估指标
        """
        logger.info("开始评估")
        
        # 如果提供了模型路径，加载模型
        if model_path:
            logger.info(f"加载模型: {model_path}")
            
            # 加载标签映射
            label_mapping_path = os.path.join(model_path, "label_mapping.json")
            if os.path.exists(label_mapping_path):
                with open(label_mapping_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                    self.config.label_mapping = mapping.get("label2id", self.config.label_mapping)
                    self.config.id2label = mapping.get("id2label", self.config.id2label)
            
            # 重新初始化模型
            self._init_model()
            self.prepare_datasets()
            
            # 加载训练好的模型
            try:
                self.model = PeftModel.from_pretrained(self.model, model_path)
                logger.info("成功加载训练好的模型")
            except Exception as e:
                logger.warning(f"加载PeftModel失败，尝试直接加载: {e}")
                # 如果PeftModel加载失败，尝试直接加载
                try:
                    from transformers import AutoModelForSequenceClassification
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    logger.info("成功加载基础模型")
                except Exception as e2:
                    logger.error(f"模型加载失败: {e2}")
                    return None
        
        # 创建评估训练器
        eval_args = TrainingArguments(
            output_dir="./temp_eval",
            per_device_eval_batch_size=self.config.batch_size,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to="none"
        )
        
        eval_trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=self.val_dataset,
            compute_metrics=self._compute_metrics,
            tokenizer=self.tokenizer
        )
        
        # 在验证集上进行评估
        logger.info("在验证集上进行评估...")
        logger.info(f"验证集样本数: {len(self.val_dataset)}")
        
        eval_results = eval_trainer.evaluate()
        
        # 详细显示评估结果
        logger.info("评估指标:")
        for key, value in eval_results.items():
            if key.startswith('eval_'):
                metric_name = key.replace('eval_', '')
                if isinstance(value, float):
                    logger.info(f"- {metric_name}: {value:.4f}")
                else:
                    logger.info(f"- {metric_name}: {value}")
        
        # 计算每个类别的详细指标
        logger.info("\n开始详细类别分析...")
        self._detailed_class_analysis()
        
        # 保存评估结果
        eval_dir = os.path.dirname(model_path) if model_path else self.config.output_dir
        eval_path = os.path.join(eval_dir, "evaluation_results.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        logger.info(f"评估结果保存在: {eval_path}")
        
        return eval_results
    
    def _detailed_class_analysis(self):
        """详细类别分析"""
        logger.info("=== 详细类别分析 ===")
        
        # 准备数据
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(0, len(self.val_dataset), self.config.batch_size):
                batch = self.val_dataset[i:i+self.config.batch_size]
                
                # 处理批次数据
                inputs = {
                    'input_ids': torch.stack([item['input_ids'] for item in batch]),
                    'attention_mask': torch.stack([item['attention_mask'] for item in batch])
                }
                
                if 'token_type_ids' in batch[0]:
                    inputs['token_type_ids'] = torch.stack([item['token_type_ids'] for item in batch])
                
                # 移动到设备
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                
                # 预测
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend([item['labels'] for item in batch])
        
        # 计算每个类别的指标
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
        
        # 分类报告
        class_names = [self.config.id2label[i] for i in range(len(self.config.label_mapping))]
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names, 
                                     output_dict=True)
        
        logger.info("\n分类报告:")
        logger.info(classification_report(all_labels, all_predictions, 
                                        target_names=class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        logger.info(f"\n混淆矩阵:\n{cm}")
        
        # 每个类别的详细指标
        logger.info("\n每个类别的详细指标:")
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                logger.info(f"\n{class_name}:")
                logger.info(f"  精确率: {metrics['precision']:.4f}")
                logger.info(f"  召回率: {metrics['recall']:.4f}")
                logger.info(f"  F1分数: {metrics['f1-score']:.4f}")
                logger.info(f"  支持数: {metrics['support']}")
        
        # 保存详细分析结果
        analysis_results = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
            "predictions": all_predictions,
            "true_labels": all_labels
        }
        
        eval_dir = self.config.output_dir
        analysis_path = os.path.join(eval_dir, "detailed_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n详细分析结果保存在: {analysis_path}")
    
    def predict(self, texts: List[str], model_path: str = None) -> List[Dict[str, Any]]:
        """
        预测复杂度
        
        Args:
            texts: 文本列表
            model_path: 模型路径，如果为None则使用当前模型
            
        Returns:
            预测结果列表，每个元素包含预测的标签和概率
        """
        logger.info(f"预测 {len(texts)} 条文本")
        
        # 如果提供了模型路径，加载模型
        if model_path:
            logger.info(f"加载模型: {model_path}")
            
            # 加载标签映射
            label_mapping_path = os.path.join(model_path, "label_mapping.json")
            if os.path.exists(label_mapping_path):
                with open(label_mapping_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                    self.config.label_mapping = mapping.get("label2id", self.config.label_mapping)
                    self.config.id2label = mapping.get("id2label", self.config.id2label)
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 检查是否为PEFT模型
            if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                # 加载基础模型
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    num_labels=len(self.config.label_mapping),
                    id2label=self.config.id2label,
                    label2id=self.config.label_mapping
                )
                
                # 加载PEFT适配器
                self.model = PeftModel.from_pretrained(base_model, model_path)
            else:
                # 直接加载完整模型
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # 确保模型在评估模式
        self.model.eval()
        
        # 创建预测数据集
        pred_dataset = ComplexityDataset(
            texts=texts,
            labels=[0] * len(texts),  # 虚拟标签
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
        
        # 创建数据加载器
        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # 预测
        results = []
        device = torch.device(self.config.device)
        self.model.to(device)
        
        with torch.no_grad():
            for batch in pred_dataloader:
                # 将批次移动到设备
                batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                
                # 预测
                outputs = self.model(**batch)
                
                # 应用softmax获取概率
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # 获取预测类别
                preds = torch.argmax(probs, dim=-1)
                
                # 添加到结果
                for i, pred in enumerate(preds):
                    pred_idx = pred.item()
                    pred_label = self.config.id2label.get(str(pred_idx), str(pred_idx))
                    confidence = probs[i, pred_idx].item()
                    
                    # 获取所有类别的概率
                    all_probs = {
                        self.config.id2label.get(str(j), str(j)): probs[i, j].item()
                        for j in range(probs.shape[1])
                    }
                    
                    results.append({
                        "label": pred_label,
                        "confidence": confidence,
                        "probabilities": all_probs
                    })
        
        return results


def train_complexity_classifier(config_path: str = None):
    """
    训练复杂度分类器的主函数
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
    """
    # 加载配置
    if config_path and os.path.exists(config_path):
        logger.info(f"加载配置: {config_path}")
        config = TrainingConfig.load(config_path)
    else:
        logger.info("使用默认配置")
        config = TrainingConfig()
    
    # 创建训练器
    trainer = ComplexityClassifierTrainer(config)
    
    # 训练模型
    model_path = trainer.train()
    
    # 评估模型
    trainer.evaluate(model_path)
    
    return model_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练复杂度分类器")
    parser.add_argument("--config", type=str, help="配置文件路径")
    args = parser.parse_args()
    
    train_complexity_classifier(args.config) 