# 复杂度分类器模型

训练时间: 20250725_211430
训练耗时: 42.22 分钟

基础模型: models/modernbert/answerdotai_ModernBERT-large
训练轮次: 15
批处理大小: 4
学习率: 1e-05
使用LoRA: True
使用FP16: True

早停配置:
- 耐心值: 3
- 阈值: 0.001

标签映射:
- zero_hop: 0
- one_hop: 1
- multi_hop: 2

训练指标:
- train_runtime: 2531.576
- train_samples_per_second: 35.089
- train_steps_per_second: 2.198
- total_flos: 2.53272432709632e+16
- train_loss: 2.181628589630127
- epoch: 8.086428089128967
