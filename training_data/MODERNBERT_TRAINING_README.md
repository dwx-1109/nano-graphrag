# ModernBERT复杂度分类器训练指南

本指南介绍如何使用ModernBERT-large模型训练复杂度分类器。

## 文件结构

```
training_data/
├── train_with_modernbert.py          # 主要训练脚本
├── test_trained_model.py             # 模型测试脚本
├── run_modernbert_training.bat       # Windows运行脚本
├── trainer.py                        # 训练器核心代码
├── processed/                        # 已处理的数据
│   ├── train_data.json              # 训练集
│   ├── val_data.json                # 验证集
│   ├── test_data.json               # 测试集
│   └── data_statistics.json         # 数据统计
└── MODERNBERT_TRAINING_README.md     # 本文件
```

## 数据准备

数据已经处理完成，包含：
- **总样本数**: 7,403条
- **类别分布**: 
  - zero_hop: 1,903条
  - one_hop: 3,000条  
  - multi_hop: 2,500条
- **数据集分割**:
  - 训练集: 5,922条
  - 验证集: 740条
  - 测试集: 741条

## 快速开始

### 1. 使用批处理脚本（推荐）

在Windows系统上，直接运行：

```bash
training_data/run_modernbert_training.bat
```

### 2. 手动运行训练

```bash
python training_data/train_with_modernbert.py --use_lora --use_fp16 --batch_size 8 --num_epochs 5
```

### 3. 自定义参数训练

```bash
python training_data/train_with_modernbert.py \
    --output_dir models/my_complexity_classifier \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --num_epochs 10 \
    --use_lora \
    --lora_r 32 \
    --use_fp16
```

## 参数说明

### 基本参数
- `--output_dir`: 输出目录（默认: models/modernbert_complexity_classifier）
- `--max_length`: 最大序列长度（默认: 256）

### 训练参数
- `--batch_size`: 批处理大小（默认: 8，ModernBERT-large较大，建议使用较小值）
- `--learning_rate`: 学习率（默认: 2e-5）
- `--num_epochs`: 训练轮次（默认: 5）
- `--warmup_ratio`: 预热比例（默认: 0.1）
- `--weight_decay`: 权重衰减（默认: 0.01）
- `--gradient_accumulation_steps`: 梯度累积步数（默认: 2）

### LoRA参数
- `--use_lora`: 启用LoRA（强烈推荐）
- `--lora_r`: LoRA秩（默认: 16）
- `--lora_alpha`: LoRA alpha（默认: 32）
- `--lora_dropout`: LoRA dropout（默认: 0.1）

### 其他参数
- `--seed`: 随机种子（默认: 42）
- `--use_fp16`: 启用FP16（推荐用于加速）
- `--logging_steps`: 日志步数（默认: 10）
- `--eval_steps`: 评估步数（默认: 100）
- `--save_steps`: 保存步数（默认: 200）
- `--device`: 设备（默认: auto）

## 训练过程

1. **模型检查**: 验证ModernBERT-large模型是否可用
2. **数据加载**: 加载已处理的数据集
3. **配置创建**: 创建训练配置
4. **模型初始化**: 初始化ModernBERT-large模型
5. **LoRA应用**: 如果启用，应用LoRA配置
6. **训练执行**: 执行训练过程
7. **模型评估**: 在测试集上评估模型
8. **结果保存**: 保存模型和评估结果

## 输出文件

训练完成后，输出目录包含：

```
models/modernbert_complexity_classifier/
├── run_YYYYMMDD_HHMMSS/              # 训练运行目录
│   ├── final_model/                  # 最终模型
│   │   ├── adapter_config.json      # LoRA配置
│   │   ├── adapter_model.safetensors # LoRA权重
│   │   ├── label_mapping.json       # 标签映射
│   │   └── README.md                # 模型说明
│   ├── training_config.json         # 训练配置
│   ├── evaluation_results.json      # 评估结果
│   └── training_summary.json        # 训练总结
```

## 测试训练后的模型

### 1. 快速测试

```bash
python training_data/test_trained_model.py --model_path models/modernbert_complexity_classifier/run_YYYYMMDD_HHMMSS/final_model --test_samples 5
```

### 2. 完整评估

```bash
python training_data/test_trained_model.py --model_path models/modernbert_complexity_classifier/run_YYYYMMDD_HHMMSS/final_model --evaluate
```

## 性能优化建议

### 1. 内存优化
- 使用较小的batch_size（4-8）
- 启用FP16训练
- 使用LoRA减少参数量

### 2. 训练加速
- 使用GPU训练
- 启用FP16
- 调整gradient_accumulation_steps

### 3. 模型质量
- 增加训练轮次
- 调整学习率
- 使用更大的LoRA秩

## 常见问题

### Q: 训练时内存不足怎么办？
A: 减小batch_size，启用FP16，使用LoRA

### Q: 训练速度太慢怎么办？
A: 启用FP16，使用GPU，调整gradient_accumulation_steps

### Q: 模型性能不好怎么办？
A: 增加训练轮次，调整学习率，检查数据质量

### Q: 如何知道训练是否成功？
A: 查看训练日志中的loss下降和评估指标提升

## 依赖要求

确保已安装以下依赖：

```bash
pip install torch transformers peft scikit-learn matplotlib numpy
```

## 注意事项

1. ModernBERT-large模型较大，建议使用GPU训练
2. 首次运行会下载模型文件，需要较长时间
3. 训练过程中会保存多个检查点，注意磁盘空间
4. 建议使用LoRA进行高效微调，避免过拟合 