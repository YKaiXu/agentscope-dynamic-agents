# AgentScope Dynamic Agents System

> 基于 [AgentScope](https://github.com/agentscope-ai/agentscope) 框架的动态Agent管理系统

## 项目简介

本项目是一个符合 AgentScope 1.0.16 设计规范的动态Agent管理系统，支持通过对话创建、删除、修改Agent，并支持为每个Agent分配独立的LLM模型。

## AgentScope 规范

本系统完全遵循 AgentScope 官方设计规范：

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

## 功能特性

| 功能 | 说明 |
|------|------|
| 动态创建Agent | 通过对话描述创建专业Agent |
| Agent管理 | 列出、删除、修改Agent |
| 独立LLM分配 | 每个Agent可使用不同的LLM模型 |
| 模型管理 | 添加、删除、查看LLM模型 |
| AgentScope规范 | 完全符合AgentScope 1.0.16设计规范 |
| Runtime支持 | 支持安全沙箱执行代码 |
| Studio集成 | 支持可视化开发调试 |

## 使用方式

### Agent管理命令

| 命令 | 功能 | 示例 |
|------|------|------|
| `/create <描述> [模型名]` | 创建Agent | `/create 一个Python专家 gpt4` |
| `/delete <名称>` | 删除Agent | `/delete python_expert` |
| `/setmodel <Agent名> <模型名>` | 设置Agent模型 | `/setmodel python_expert gpt4` |
| `/list` | 列出所有Agent | `/list` |

### 模型管理命令

| 命令 | 功能 | 示例 |
|------|------|------|
| `/models` | 查看所有模型 | `/models` |
| `/addmodel <名称> <模型名> <API Key> <Base URL>` | 添加模型 | `/addmodel gpt4 gpt-4 sk-xxx https://api.openai.com/v1` |
| `/delmodel <名称>` | 删除模型 | `/delmodel gpt4` |

### 对话命令

| 命令 | 功能 | 示例 |
|------|------|------|
| `@<名称> <问题>` | 与Agent对话 | `@python_expert 如何优化代码？` |

## 配置文件

### models_config.json

```json
{
  "default": {
    "model_name": "deepseek-v3",
    "api_key": "xxx",
    "base_url": "https://api.example.com/v1"
  },
  "gpt4": {
    "model_name": "gpt-4",
    "api_key": "sk-xxx",
    "base_url": "https://api.openai.com/v1"
  }
}
```

### dynamic_agents.json

```json
{
  "python_expert": {
    "name": "python_expert",
    "display_name": "Python编程专家",
    "description": "精通Python编程的专家",
    "sys_prompt": "你是一位Python专家...",
    "model": "gpt4",
    "created_at": "2026-01-01T00:00:00"
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

## 安装依赖

```bash
pip install agentscope
pip install agentscope-runtime  # 可选，用于沙箱执行
```

## 相关项目

- [AgentScope 官方仓库](https://github.com/agentscope-ai/agentscope)
- [AgentScope Runtime](https://github.com/agentscope-ai/agentscope-runtime)
- [AgentScope Studio](https://github.com/agentscope-ai/agentscope-studio)
- [AgentScope DingTalk Bot](https://github.com/YKaiXu/agentscope-dingtalk-bot)

## 许可证

Apache License 2.0

