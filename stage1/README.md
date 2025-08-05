# 阶段1：复杂度感知检索系统验证

## 🎯 阶段目标

验证复杂度感知路由系统的有效性，使用真实数据集测试不同复杂度查询的路由准确性和检索效果。

## 📊 实验设计

### 1. 数据集准备

#### 主要数据集：
- **MS MARCO QA Dataset**: 包含不同复杂度的问答对
- **Natural Questions**: Google 自然语言问题数据集  
- **HotpotQA**: 多跳推理问答数据集
- **SQuAD 2.0**: 阅读理解数据集

#### 复杂度标注：
根据查询特征将问题分为三类：
- **Zero-hop**: 直接定义性问题，无需检索
- **One-hop**: 单步检索问题
- **Multi-hop**: 需要多步推理的问题

### 2. 评估指标

#### 路由准确性：
- **路由精度**: 正确路由到目标模式的比例
- **路由召回率**: 各复杂度类别的识别准确率
- **置信度校准**: ECE (Expected Calibration Error)

#### 检索效果：
- **检索准确率**: Hit@K, MRR@K
- **答案质量**: BLEU, ROUGE, BERTScore
- **响应时间**: 平均查询处理时间

### 3. 基线对比

与以下系统进行对比：
- **固定路由**: 所有查询使用相同检索模式
- **随机路由**: 随机选择检索模式  
- **基于规则路由**: 使用启发式规则
- **传统RAG**: 不区分查询复杂度的标准RAG

## 🔬 实验流程

### Phase 1: 数据预处理 (1-2天)
1. 下载和清理数据集
2. 复杂度标注和验证
3. 构建测试集合

### Phase 2: 系统配置 (2-3天)  
1. 配置复杂度分类器
2. 设置各检索模式参数
3. 初始化评估环境

### Phase 3: 实验执行 (3-4天)
1. 运行基线对比实验
2. 测试复杂度感知路由
3. 收集性能数据

### Phase 4: 结果分析 (2-3天)
1. 统计分析和可视化
2. 错误案例分析
3. 撰写实验报告

## 📁 目录结构

```
stage1/
├── README.md                 # 本文档
├── data/                     # 数据目录
│   ├── raw/                 # 原始数据
│   ├── processed/           # 处理后数据
│   └── annotations/         # 复杂度标注
├── scripts/                  # 实验脚本
│   ├── data_preparation.py  # 数据预处理
│   ├── run_experiments.py   # 实验执行
│   └── analysis.py          # 结果分析
├── configs/                  # 配置文件
│   ├── complexity_router.json
│   └── evaluation.json
├── outputs/                  # 实验输出
│   ├── results/             # 实验结果
│   ├── logs/                # 运行日志
│   └── visualizations/      # 可视化图表
└── notebooks/               # 分析笔记本
    ├── data_exploration.ipynb
    └── result_analysis.ipynb
```

## 🚀 快速开始

### 1. 环境准备

⚠️ **重要**: 阶段1实验需要以下关键依赖，请务必提前准备：

#### **必需的深度学习模型**
- **ModernBERT复杂度分类器**: `nano_graphrag/models/modernbert_complexity_classifier/`
- **嵌入模型**: sentence-transformers (用于语义相似度评估)

#### **GPU要求**
- **GPU内存**: 至少8GB (推荐16GB)
- **系统内存**: 至少16GB
- **存储空间**: 至少50GB

#### **完整依赖检查**
```bash
# 确保在 pytorch_new 环境中
conda activate pytorch_new

# 运行依赖检查脚本
python stage1/setup_dependencies.py

# 如果有缺失依赖，自动安装
python stage1/setup_dependencies.py --install

# 检查结果应显示所有项目都通过 ✅
```

#### **如果缺少复杂度分类器模型**
```bash
# 需要先训练模型（这是必需步骤！）
python train_complexity_classifier.py

# 或者使用预训练模型（如果有的话）
# 模型文件应包含：config.json, pytorch_model.bin, tokenizer_config.json 等
```

### 2. 数据准备
```bash
cd stage1
python scripts/data_preparation.py --download --annotate
```

### 3. 运行实验
```bash
python scripts/run_experiments.py --config configs/complexity_router.json
```

### 4. 分析结果
```bash
python scripts/analysis.py --results_dir outputs/results/
```

## 📈 预期结果

### 性能提升预期：
- **路由准确率**: > 85%
- **检索效率**: 提升 15-25%
- **答案质量**: BLEU分数提升 10-15%
- **响应时间**: 优化 20-30%

### 关键发现：
1. 复杂度感知路由的有效性验证
2. 不同复杂度查询的最优检索策略
3. 置信度校准的重要性
4. 系统瓶颈和优化方向

## 📝 实验记录

### 实验日志：
- 每次实验运行自动记录参数和结果
- 错误案例详细分析
- 性能指标趋势跟踪

### 版本控制：
- 所有实验配置和代码版本化
- 结果可复现性保证
- 实验分支管理

## 🔗 相关资源

### 数据集链接：
- [MS MARCO](https://microsoft.github.io/msmarco/)
- [Natural Questions](https://ai.google.com/research/NaturalQuestions)
- [HotpotQA](https://hotpotqa.github.io/)
- [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)

### 参考论文：
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- "FiD: Fusion-in-Decoder for Open-domain Question Answering"
- "Dense Passage Retrieval for Open-Domain Question Answering"

## ⚠️ 注意事项

1. **数据使用合规**: 确保数据集使用符合许可协议
2. **计算资源**: 实验需要GPU资源，预估时间和成本
3. **结果保存**: 及时备份重要实验结果
4. **环境一致性**: 使用指定的conda环境和依赖版本

## 📞 支持

如有问题，请参考：
1. 项目主README文档
2. 相关模块的单元测试
3. 端到端测试示例

---

**实施时间**: 预计 8-12 天
**负责人**: 项目团队
**状态**: 进行中 🚀