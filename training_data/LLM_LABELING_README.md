# LLM辅助问题复杂度标注工具

## 概述

这个工具使用云服务LLM（如OpenAI GPT-4）对问题进行复杂度分类标注，支持三种类别：
- **ZERO_HOP**：零跳查询（直接事实查询）
- **ONE_HOP**：单跳查询（需要一次信息检索）
- **MULTI_HOP**：多跳查询（需要多个推理步骤）

## 功能特点

- ✅ 支持多种JSON输入格式
- ✅ 批量处理大量问题
- ✅ 自动重试机制
- ✅ 详细的标注分析和推理
- ✅ 完整的日志记录
- ✅ 标注结果统计和摘要

## 安装依赖

```bash
pip install openai requests
```

## 使用方法

### 1. 准备输入文件

支持以下JSON格式：

**格式1：问题列表**
```json
[
  "What is the capital of France?",
  "Who is Beyoncé?",
  "When was Beyoncé born?"
]
```

**格式2：对象列表**
```json
[
  {"question": "What is the capital of France?"},
  {"question": "Who is Beyoncé?"},
  {"question": "When was Beyoncé born?"}
]
```

**格式3：嵌套结构**
```json
{
  "questions": [
    "What is the capital of France?",
    "Who is Beyoncé?",
    "When was Beyoncé born?"
  ]
}
```

### 2. 运行标注工具

```bash
python llm_labeling_tool.py \
  --input example_questions.json \
  --output labeled_results.json \
  --api-key YOUR_OPENAI_API_KEY \
  --model gpt-4 \
  --batch-size 10 \
  --temperature 0.1
```

### 3. 参数说明

- `--input, -i`：输入JSON文件路径（必需）
- `--output, -o`：输出JSON文件路径（必需）
- `--api-key`：OpenAI API密钥（必需）
- `--model`：使用的模型，默认gpt-4
- `--batch-size`：批处理大小，默认10
- `--temperature`：温度参数，默认0.1

## 输出格式

标注结果包含以下信息：

```json
{
  "metadata": {
    "timestamp": "2025-07-26T16:45:00.000000",
    "total_questions": 20,
    "successful_labels": 19,
    "failed_labels": 1,
    "model": "gpt-4",
    "label_distribution": {
      "ZERO_HOP": 8,
      "ONE_HOP": 7,
      "MULTI_HOP": 4,
      "FAILED": 1,
      "UNKNOWN": 0
    }
  },
  "results": [
    {
      "question": "What is the capital of France?",
      "label": "ZERO_HOP",
      "analysis": "这是一个直接的事实查询，答案可以从单一信息源获得",
      "reasoning": "查询询问法国的首都，这是一个基本的地理事实，无需任何推理步骤",
      "llm_response": "{\"analysis\": \"这是一个直接的事实查询...\", \"complexity\": \"ZERO_HOP\", \"reasoning\": \"查询询问法国的首都...\"}"
    }
  ]
}
```

## 标注标准

### ZERO_HOP（零跳查询）
- **定义**：答案可以直接从单一信息源获得，无需任何推理
- **特征**：直接的事实查询，答案明确且单一
- **示例**：
  - "What is the capital of France?" → ZERO_HOP
  - "Who is Beyoncé?" → ZERO_HOP
  - "When was Beyoncé born?" → ZERO_HOP

### ONE_HOP（单跳查询）
- **定义**：需要一次信息检索或简单推理，答案相对直接
- **特征**：单一查询步骤，答案相对明确
- **示例**：
  - "What is Beyoncé's daughter's name?" → ONE_HOP
  - "How many Grammy awards did Beyoncé win?" → ONE_HOP
  - "Where was Beyoncé born?" → ONE_HOP

### MULTI_HOP（多跳查询）
- **定义**：需要多个信息源或多个推理步骤
- **特征**：涉及比较、关联、条件推理或复杂查询
- **示例**：
  - "In addition to co-writing credits, Beyoncé also got what credits for most of her albums?" → MULTI_HOP
  - "Who portrayed Chopin in the 1991 film, Impromptu?" → MULTI_HOP
  - "Which professor from Northeastern University wrote about how race correlates with these criticisms of Beyonce?" → MULTI_HOP

## 使用示例

### 示例1：标注少量问题

```bash
python llm_labeling_tool.py \
  --input example_questions.json \
  --output results.json \
  --api-key sk-your-api-key-here
```

### 示例2：使用不同模型和参数

```bash
python llm_labeling_tool.py \
  --input large_question_set.json \
  --output large_results.json \
  --api-key sk-your-api-key-here \
  --model gpt-3.5-turbo \
  --batch-size 20 \
  --temperature 0.05
```

## 注意事项

1. **API密钥安全**：请妥善保管您的API密钥，不要提交到版本控制系统
2. **API限制**：注意OpenAI的API调用频率限制，适当调整batch-size
3. **成本控制**：大量标注会产生API费用，建议先小批量测试
4. **结果验证**：建议对标注结果进行抽样人工验证
5. **网络连接**：确保网络连接稳定，程序会自动重试失败的请求

## 故障排除

### 常见问题

1. **API调用失败**
   - 检查API密钥是否正确
   - 确认网络连接正常
   - 检查API配额是否充足

2. **JSON格式错误**
   - 确保输入文件是有效的JSON格式
   - 检查问题字段是否正确

3. **标注结果不准确**
   - 调整temperature参数
   - 检查标注标准是否清晰
   - 考虑使用更高级的模型

### 日志文件

程序会生成详细的日志文件 `llm_labeling.log`，包含：
- API调用详情
- 错误信息
- 标注进度
- 性能统计

## 扩展功能

### 支持其他LLM服务

可以通过修改 `call_llm_api` 方法来支持其他LLM服务：

- Anthropic Claude
- Google Gemini
- 阿里云通义千问
- 百度文心一言

### 自定义标注标准

可以修改 `labeling_criteria` 来适应不同的标注需求。

### 批量处理优化

对于大量数据，可以考虑：
- 并行处理
- 断点续传
- 结果缓存

## 许可证

MIT License 