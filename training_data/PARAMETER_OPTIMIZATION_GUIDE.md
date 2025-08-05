# 复杂度分类器参数优化指南

本指南详细说明如何调整训练参数以提高模型的准确率。

## 已优化的默认参数

### 训练参数优化

| 参数 | 原值 | 优化值 | 优化原因 |
|------|------|--------|----------|
| batch_size | 8 | 4 | 较小的batch_size有助于更好的泛化，减少过拟合 |
| learning_rate | 2e-5 | 1e-5 | 较小的学习率有助于更稳定的训练，避免震荡 |
| num_epochs | 8 | 10 | 增加训练轮次以提高模型性能 |
| warmup_ratio | 0.1 | 0.15 | 增加预热比例有助于稳定训练初期 |
| weight_decay | 0.01 | 0.02 | 增加权重衰减防止过拟合 |
| gradient_accumulation_steps | 2 | 4 | 增加梯度累积以获得更稳定的梯度 |

### LoRA参数优化

| 参数 | 原值 | 优化值 | 优化原因 |
|------|------|--------|----------|
| lora_r | 16 | 32 | 增加秩以提高模型表达能力 |
| lora_alpha | 32 | 64 | 增加alpha以增强LoRA效果 |
| lora_dropout | 0.1 | 0.15 | 增加dropout防止过拟合 |

### 训练策略优化

| 优化项 | 说明 |
|--------|------|
| 余弦学习率调度 | 使用cosine调度器，提供更平滑的学习率衰减 |
| 早停机制 | 添加早停，防止过拟合 |
| 梯度裁剪 | 设置max_grad_norm=1.0，防止梯度爆炸 |
| 多进程数据加载 | 使用dataloader_num_workers=2提高数据加载效率 |
| 更多检查点保存 | 保存最近的5个检查点而不是3个 |

## 进一步优化建议

### 1. 数据增强策略

```python
# 在trainer.py中添加数据增强
def augment_text(text):
    """简单的数据增强"""
    # 随机删除一些词
    # 随机替换同义词
    # 随机调整词序
    return augmented_text
```

### 2. 学习率调度优化

```python
# 使用更复杂的学习率调度
lr_scheduler_type="cosine_with_restarts"  # 余弦重启调度
num_train_epochs=15  # 增加训练轮次
```

### 3. 模型集成

```python
# 训练多个模型并集成
models = []
for seed in [42, 123, 456, 789, 999]:
    # 使用不同种子训练模型
    model = train_with_seed(seed)
    models.append(model)

# 集成预测
ensemble_prediction = ensemble_predict(models, text)
```

### 4. 交叉验证

```python
# 使用K折交叉验证
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    # 训练第fold折
    train_fold_model(fold, train_idx, val_idx)
```

## 参数调优策略

### 1. 网格搜索

```bash
# 尝试不同的学习率和batch_size组合
for lr in [5e-6, 1e-5, 2e-5, 5e-5]:
    for bs in [2, 4, 8]:
        python train_with_modernbert.py --learning_rate $lr --batch_size $bs
```

### 2. 贝叶斯优化

```python
# 使用optuna进行贝叶斯优化
import optuna

def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
    bs = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
    # 训练并返回验证准确率
    return validation_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### 3. 渐进式调优

1. **第一阶段**: 调整基础参数（学习率、batch_size）
2. **第二阶段**: 优化正则化参数（weight_decay、dropout）
3. **第三阶段**: 调整LoRA参数（r、alpha）
4. **第四阶段**: 优化训练策略（调度器、早停）

## 性能监控

### 关键指标

- **训练损失**: 应该持续下降
- **验证损失**: 应该下降但不应该过拟合
- **F1分数**: 主要评估指标
- **准确率**: 整体准确率
- **各类别F1**: 每个类别的性能

### 早停策略

```python
# 监控验证F1分数
early_stopping_patience=3
early_stopping_threshold=0.001
metric_for_best_model="f1_weighted"
```

## 常见问题解决

### 1. 过拟合

**症状**: 训练损失持续下降，验证损失开始上升

**解决方案**:
- 增加weight_decay（0.02 → 0.05）
- 增加dropout（0.15 → 0.2）
- 减少模型复杂度（降低lora_r）
- 增加数据增强

### 2. 欠拟合

**症状**: 训练和验证损失都很高

**解决方案**:
- 增加训练轮次
- 提高学习率
- 增加模型复杂度（提高lora_r）
- 减少正则化

### 3. 训练不稳定

**症状**: 损失震荡，收敛慢

**解决方案**:
- 降低学习率
- 增加warmup_ratio
- 使用梯度累积
- 调整batch_size

## 最佳实践

1. **从小开始**: 先用较小的参数开始，逐步增加
2. **监控指标**: 密切关注训练和验证指标
3. **保存检查点**: 定期保存模型检查点
4. **实验记录**: 记录每次实验的参数和结果
5. **验证策略**: 使用交叉验证确保结果稳定

## 预期性能提升

使用优化后的参数，预期可以获得以下提升：

- **准确率**: +3-5%
- **F1分数**: +4-6%
- **训练稳定性**: 显著提升
- **收敛速度**: 更快收敛

## 运行优化训练

```bash
# 使用优化后的参数运行训练
python training_data/train_with_modernbert.py \
    --use_lora \
    --use_fp16 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --num_epochs 10 \
    --lora_r 32 \
    --lora_alpha 64 \
    --weight_decay 0.02 \
    --warmup_ratio 0.15 \
    --gradient_accumulation_steps 4
``` 