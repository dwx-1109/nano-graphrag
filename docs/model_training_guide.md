# ModernBERT模型微调指南

本指南介绍如何针对您的具体数据和使用场景微调ModernBERT模型，以提高查询复杂度分类的准确性。

## 1. 为什么需要微调？

当前系统使用的是通用的ModernBERT-large模型，虽然功能强大，但对于特定的查询复杂度分类任务可能精度不够高。通过微调，您可以：

- 提高分类准确率（从20%提升到80%+）
- 适配您的特定领域和查询模式
- 优化模型参数以减少计算资源消耗

## 2. 准备训练数据

### 2.1 数据格式

创建包含查询-复杂度标签对的数据集：

```json
[
    {
        "query": "什么是人工智能？",
        "complexity": "zero_hop",
        "explanation": "简单定义性问题，无需检索"
    },
    {
        "query": "告诉我关于深度学习的发展历史",
        "complexity": "one_hop", 
        "explanation": "需要单步信息检索"
    },
    {
        "query": "比较深度学习和传统机器学习在图像识别中的优缺点",
        "complexity": "multi_hop",
        "explanation": "需要多步推理和对比分析"
    }
]
```

### 2.2 数据收集策略

1. **真实查询收集**：收集您系统中的真实用户查询
2. **人工标注**：邀请领域专家标注查询复杂度
3. **数据增强**：使用LLM生成更多相似查询
4. **均衡分布**：确保三种复杂度类别的数据相对均衡

### 2.3 推荐数据量

- 最小规模：每类别 500 条（总计 1500 条）
- 推荐规模：每类别 2000 条（总计 6000 条）
- 理想规模：每类别 5000 条（总计 15000 条）

## 3. 微调脚本

### 3.1 安装依赖

```bash
pip install transformers datasets peft accelerate
```

### 3.2 微调代码示例

```python
#!/usr/bin/env python3
"""
ModernBERT复杂度分类器微调脚本
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 配置
MODEL_NAME = "answerdotai/ModernBERT-large"
OUTPUT_DIR = "./fine_tuned_complexity_classifier"
DATA_FILE = "training_data.json"

# 标签映射
LABEL_MAP = {"zero_hop": 0, "one_hop": 1, "multi_hop": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

def load_data(data_file):
    """加载训练数据"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = [item['query'] for item in data]
    labels = [LABEL_MAP[item['complexity']] for item in data]
    
    return Dataset.from_dict({
        'text': queries,
        'labels': labels
    })

def tokenize_function(examples, tokenizer):
    """数据预处理函数"""
    return tokenizer(
        examples['text'], 
        truncation=True, 
        padding=True,
        max_length=256
    )

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # 加载数据
    print("加载训练数据...")
    dataset = load_data(DATA_FILE)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    # 加载tokenizer和模型
    print("加载模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
        attn_implementation="eager"
    )
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value", "dense"]
    )
    model = get_peft_model(model, peft_config)
    
    # 数据预处理
    print("预处理数据...")
    train_dataset = dataset['train'].map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    eval_dataset = dataset['test'].map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{OUTPUT_DIR}/logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to=None  # 禁用wandb等
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    print("保存模型...")
    trainer.save_model()
    
    # 保存标签映射
    with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), 'w') as f:
        json.dump({
            "id2label": ID2LABEL,
            "label2id": LABEL_MAP
        }, f, indent=2)
    
    print(f"训练完成！模型保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
```

## 4. 使用微调后的模型

### 4.1 更新配置

```python
from nano_graphrag import GraphRAG, ComplexityAwareRouter

# 使用微调后的模型
rag = GraphRAG(
    working_dir="./my_rag_cache",
    router_cls=ComplexityAwareRouter,
    router_kwargs={
        "model_path": "./fine_tuned_complexity_classifier",  # 微调后的模型路径
        "confidence_threshold": 0.8,  # 提高置信度阈值
        "enable_fallback": True,
    }
)
```

### 4.2 验证效果

```python
# 测试查询
test_queries = [
    "什么是机器学习？",
    "解释深度学习的工作原理", 
    "比较CNN和RNN在NLP任务中的性能差异"
]

for query in test_queries:
    result = await rag.router.predict_complexity_detailed(query)
    print(f"查询: {query}")
    print(f"复杂度: {result['final_complexity']}")
    print(f"置信度: {result['model_prediction']['confidence']:.3f}")
    print()
```

## 5. 模型优化技巧

### 5.1 超参数调优

- **学习率**：推荐范围 1e-5 到 5e-5
- **批次大小**：根据GPU内存调整，推荐16-32
- **训练轮数**：通常3-5轮即可，避免过拟合
- **LoRA rank**：推荐8-32，更高的rank提供更好的性能但增加参数量

### 5.2 数据增强

```python
# 使用GPT生成更多训练数据
prompt_template = """
生成10个关于{topic}的查询，复杂度为{complexity}:

复杂度定义：
- zero_hop: 简单定义或常识问题
- one_hop: 需要单步信息检索
- multi_hop: 需要多步推理和对比分析

格式：
1. 查询内容
2. 查询内容
...
"""
```

## 6. 部署建议

### 6.1 模型版本管理

- 使用版本控制追踪模型变更
- 保留多个检查点以便回滚
- 定期评估模型性能并重新训练

### 6.2 监控和维护

```python
# 添加性能监控
class MonitoredComplexityClassifier(ComplexityClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_stats = {"total": 0, "high_confidence": 0}
    
    def predict_with_confidence(self, query: str):
        result = super().predict_with_confidence(query)
        
        # 记录统计信息
        self.prediction_stats["total"] += 1
        if result[1] > self.config.confidence_threshold:
            self.prediction_stats["high_confidence"] += 1
        
        return result
    
    def get_performance_stats(self):
        total = self.prediction_stats["total"]
        high_conf = self.prediction_stats["high_confidence"]
        return {
            "total_predictions": total,
            "high_confidence_rate": high_conf / total if total > 0 else 0
        }
```

## 7. 故障排除

### 7.1 常见问题

1. **内存不足**：减少批次大小或使用梯度累积
2. **训练不收敛**：降低学习率或增加训练数据
3. **过拟合**：减少训练轮数或增加数据增强

### 7.2 性能基准

| 数据集大小 | 训练时间 | 预期准确率 |
|-----------|----------|------------|
| 1500 条   | 30分钟   | 70-75%     |
| 6000 条   | 2小时    | 80-85%     |
| 15000 条  | 5小时    | 85-90%     |

## 8. 进阶技巧

### 8.1 多任务学习

同时训练复杂度分类和检索模式选择两个任务：

```python
# 多任务损失函数
def multi_task_loss(complexity_logits, mode_logits, complexity_labels, mode_labels):
    complexity_loss = F.cross_entropy(complexity_logits, complexity_labels)
    mode_loss = F.cross_entropy(mode_logits, mode_labels)
    return 0.7 * complexity_loss + 0.3 * mode_loss
```

### 8.2 主动学习

自动识别模型不确定的查询，优先标注：

```python
def identify_uncertain_queries(queries, model, threshold=0.5):
    """识别模型预测不确定的查询"""
    uncertain = []
    for query in queries:
        _, confidence, _ = model.predict_with_confidence(query)
        if confidence < threshold:
            uncertain.append(query)
    return uncertain
```

通过遵循本指南，您可以显著提高ModernBERT分类器的性能，使其更好地适配您的特定使用场景。