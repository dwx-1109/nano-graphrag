# 配置管理模块

本目录包含 nano-graphrag 的配置管理功能。

## 文件说明

- `config_template.json` - 配置模板文件
- `__init__.py` - 配置管理模块
- `config_tool.py` - 配置管理工具脚本

## 使用方法

### 1. 加载配置模板

```python
from nano_graphrag.config import load_config_template

# 加载配置模板
config_template = load_config_template()
```

### 2. 加载配置文件

```python
from nano_graphrag.config import load_config

# 加载配置文件
config = load_config("path/to/config.json")
```

### 3. 保存配置文件

```python
from nano_graphrag.config import save_config

# 保存配置文件
save_config(config, "path/to/config.json")
```

### 4. 从模板创建配置文件

```python
from nano_graphrag.config import create_config_from_template

# 从模板创建配置文件
config = create_config_from_template(
    "my_config.json",
    working_dir="./my_cache",
    api_type="openai",
    model_name="gpt-4"
)
```

### 5. 验证配置文件

```python
from nano_graphrag.config import validate_config

# 验证配置文件
is_valid = validate_config(config)
```

## 命令行工具

使用 `config_tool.py` 进行命令行配置管理：

### 显示配置模板

```bash
python nano_graphrag/config/config_tool.py template
```

### 创建配置文件

```bash
python nano_graphrag/config/config_tool.py create my_config.json \
    --working-dir ./my_cache \
    --api-type openai \
    --model-name gpt-4
```

### 验证配置文件

```bash
python nano_graphrag/config/config_tool.py validate my_config.json
```

## 配置模板结构

配置模板包含以下主要部分：

- `working_dir` - 工作目录
- `api_type` - API类型 (dashscope, openai, azure等)
- `answer_generator_config` - 答案生成器配置
- `retrieval_config` - 检索配置
- `fusion_config` - 融合配置
- `deduplication_config` - 去重配置

详细配置项请参考 `config_template.json` 文件。 