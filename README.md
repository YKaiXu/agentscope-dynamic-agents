# AgentScope Dynamic Agents System

> 基于 [AgentScope](https://github.com/agentscope-ai/agentscope) 框架的动态Agent管理系统

## 项目简介

本项目是一个符合 AgentScope 1.0.16 设计规范的动态Agent管理系统，支持通过对话创建、删除、修改Agent。

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
| AgentScope规范 | 完全符合AgentScope 1.0.16设计规范 |
| Runtime支持 | 支持安全沙箱执行代码 |
| Studio集成 | 支持可视化开发调试 |

## 系统架构

```
┌────────────────────────────────────────────────────────────┐
│                    AgentScope Studio (端口: 3001)           │
│   可视化界面 │ 追踪分析 │ 项目管理                            │
└───────────────────────────┬────────────────────────────────┘
                            │ studio_url
                            ▼
┌────────────────────────────────────────────────────────────┐
│                    AgentScope Core (端口: 5000)             │
│   ReActAgent │ OpenAIChatModel │ OpenAIChatFormatter       │
└───────────────────────────┬────────────────────────────────┘
                            │ 沙箱工具调用
                            ▼
┌────────────────────────────────────────────────────────────┐
│                  AgentScope Runtime                         │
│   BaseSandbox │ BrowserSandbox │ GuiSandbox                │
│   Docker容器安全隔离执行                                    │
└────────────────────────────────────────────────────────────┘
```

## 安装依赖

```bash
# 核心框架
pip install agentscope

# Runtime (可选，用于沙箱执行)
pip install agentscope-runtime

# Studio (可选，用于可视化)
npm install -g @agentscope/studio
```

## 配置

### 环境变量

```bash
export MODEL_NAME="your-model-name"
export MODEL_API_KEY="your-api-key"
export MODEL_BASE_URL="your-api-base-url"
```

### 配置文件

编辑 `app_dynamic_agents.py` 中的配置：

```python
MODEL_CONFIG = {
    "model_name": os.environ.get("MODEL_NAME", "your-model-name"),
    "api_key": os.environ.get("MODEL_API_KEY", "your-api-key"),
    "base_url": os.environ.get("MODEL_BASE_URL", "https://api.openai.com/v1")
}
```

## 使用方式

### 命令列表

| 命令 | 功能 | 示例 |
|------|------|------|
| `/create <描述>` | 创建Agent | `/create 一个Python专家` |
| `/delete <名称>` | 删除Agent | `/delete python_expert` |
| `/list` | 列出所有Agent | `/list` |
| `@<名称> <问题>` | 与Agent对话 | `@python_expert 如何优化代码？` |
| `/help` | 查看帮助 | `/help` |

### 启动服务

```bash
# 方式1: 直接运行
python app_dynamic_agents.py

# 方式2: 使用systemd服务
sudo cp agentscope.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start agentscope
```

## API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态 |
| `/health` | GET | 健康检查 |
| `/agents` | GET | Agent列表 |
| `/chat` | POST | 对话接口 |

### 对话示例

```bash
curl -X POST http://localhost:5000/chat   -H "Content-Type: application/json"   -d '{"message": "@python_expert 如何优化代码？"}'
```

## AgentScope 组件说明

| 组件 | 说明 |
|------|------|
| `ReActAgent` | 推理行动Agent，支持工具调用 |
| `OpenAIChatModel` | OpenAI兼容模型封装 |
| `OpenAIChatFormatter` | 消息格式化器 |
| `Msg` | 消息类，Agent间通信载体 |
| `Toolkit` | 工具管理器 |
| `InMemoryMemory` | 内存记忆 |

## 相关项目

- [AgentScope 官方仓库](https://github.com/agentscope-ai/agentscope)
- [AgentScope Runtime](https://github.com/agentscope-ai/agentscope-runtime)
- [AgentScope Studio](https://github.com/agentscope-ai/agentscope-studio)
- [DingTalk Stream Bot (AgentScope集成版)](https://github.com/YKaiXu/dingtalk-stream-bot)

## 许可证

Apache License 2.0

