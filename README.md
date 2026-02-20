# AgentScope Dynamic Agents System

> 基于 [AgentScope](https://github.com/agentscope-ai/agentscope) 框架的动态Agent管理系统

## 项目简介

本项目是一个符合 AgentScope 1.0.16 设计规范的动态Agent管理系统，支持通过对话创建、删除、修改Agent，并支持为每个Agent分配独立的LLM模型。

## 功能特性

- ✅ 动态创建Agent（支持自定义简短名字）
- ✅ 独立LLM模型分配
- ✅ 结构化命令支持
- ✅ 完全符合AgentScope 1.0.16规范

## 安装说明

### 环境要求

- Python 3.10+
- pip 或 uv

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/YKaiXu/agentscope-dynamic-agents.git
cd agentscope-dynamic-agents

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install agentscope flask openai

# 4. 配置环境变量
export MODEL_NAME="your-model-name"
export MODEL_API_KEY="your-api-key"
export MODEL_BASE_URL="your-api-base-url"

# 5. 启动服务
python app_dynamic_agents.py
```

### 使用systemd服务（推荐生产环境）

```bash
# 复制服务文件
sudo cp agentscope.service /etc/systemd/system/

# 启动服务
sudo systemctl daemon-reload
sudo systemctl start agentscope
sudo systemctl enable agentscope  # 开机自启

# 查看状态
sudo systemctl status agentscope

# 查看日志
journalctl -u agentscope -f
```

## 全部命令列表

### Agent管理命令

| 命令 | 功能 | 示例 |
|------|------|------|
| `/create [name] <描述>` | 创建Agent | `/create py 一个Python专家` |
| `/create {json}` | JSON格式创建 | `/create {"name":"py","display_name":"Python专家","sys_prompt":"..."}` |
| `/delete <名称>` | 删除Agent | `/delete py` |
| `/setmodel <Agent> <模型>` | 设置Agent使用的模型 | `/setmodel py gpt4` |
| `/list` 或 `/agents` | 列出所有Agent | `/list` |
| `@<名称> <问题>` | 与Agent对话 | `@py 如何优化代码？` |

### 模型管理命令

| 命令 | 功能 | 示例 |
|------|------|------|
| `/models` | 列出所有模型 | `/models` |
| `/addmodel <名称> <模型名> <API Key> <Base URL>` | 添加模型 | `/addmodel gpt4 gpt-4 sk-xxx https://api.openai.com/v1` |
| `/delmodel <名称>` | 删除模型 | `/delmodel gpt4` |

### 其他命令

| 命令 | 功能 |
|------|------|
| `/help` | 查看帮助信息 |

## 创建Agent的三种方式

### 方式1: 简单创建
```
/create 一个Python专家           # 自动生成名字
/create py 一个Python专家        # 指定名字"py"
/create py 一个Python专家 gpt4   # 指定名字和模型
```

### 方式2: 结构化创建
```
/create name=py display="Python专家" desc="Python编程专家" prompt="你是Python专家" model=default
```

### 方式3: JSON创建
```
/create {"name":"py","display_name":"Python专家","description":"Python编程专家","sys_prompt":"你是Python专家","model":"default"}
```

## AgentScope Agent字段说明

| 字段 | 必需 | 说明 |
|------|------|------|
| `name` | ✅ | Agent名称（简短，用于@调用） |
| `sys_prompt` | ✅ | 系统提示词，定义Agent角色和能力 |
| `model` | ✅ | LLM模型名称 |
| `formatter` | ✅ | 消息格式化器（系统自动提供） |
| `display_name` | ❌ | 显示名称 |
| `description` | ❌ | Agent描述 |

## 配置文件

### dynamic_agents.json
```json
{
  "py": {
    "name": "py",
    "display_name": "Python专家",
    "description": "Python编程专家",
    "sys_prompt": "你是一位Python专家...",
    "model": "default",
    "created_at": "2026-01-01T00:00:00"
  }
}
```

### models_config.json
```json
{
  "default": {
    "model_name": "gpt-4",
    "api_key": "your-api-key",
    "base_url": "https://api.openai.com/v1"
  },
  "gpt4": {
    "model_name": "gpt-4",
    "api_key": "sk-xxx",
    "base_url": "https://api.openai.com/v1"
  }
}
```

## API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态 |
| `/health` | GET | 健康检查 |
| `/agents` | GET | Agent列表 |
| `/models` | GET | 模型列表 |
| `/chat` | POST | 对话接口 |

### 对话API示例
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "@py 如何优化代码？"}'
```

## AgentScope规范

本系统完全遵循 AgentScope 1.0.16 设计规范：

```python
import agentscope
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg

# 初始化
agentscope.init(project='xxx', logging_path='xxx.log')

# 创建模型
model = OpenAIChatModel(
    model_name='xxx',
    api_key='xxx',
    client_kwargs={'base_url': 'xxx'}
)

# 创建Agent
agent = ReActAgent(
    name='xxx',
    sys_prompt='xxx',
    model=model,
    formatter=OpenAIChatFormatter()
)

# 调用 (异步)
response = await agent(Msg(name='user', role='user', content='xxx'))
```

## 相关项目

- [AgentScope 官方仓库](https://github.com/agentscope-ai/agentscope)
- [AgentScope Runtime](https://github.com/agentscope-ai/agentscope-runtime)
- [AgentScope Studio](https://github.com/agentscope-ai/agentscope-studio)
- [AgentScope DingTalk Bot](https://github.com/YKaiXu/agentscope-dingtalk-bot) - 钉钉集成插件

## 许可证

Apache License 2.0

