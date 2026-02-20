# AgentScope Dynamic Agents System

基于AgentScope框架的动态Agent管理系统，支持通过对话创建、删除、修改Agent。

## 功能特性

- 动态创建Agent: 通过对话描述创建专业Agent
- Agent管理: 列出、删除、修改Agent
- 钉钉集成: 支持钉钉Stream消息推送
- AgentScope规范: 完全符合AgentScope 1.0.16设计规范
- Runtime支持: 支持安全沙箱执行代码
- Studio集成: 支持可视化开发调试

## 使用方式

| 命令 | 功能 | 示例 |
|------|------|------|
| `/create <描述>` | 创建Agent | `/create 一个Python专家` |
| `/delete <名称>` | 删除Agent | `/delete python_expert` |
| `/list` | 列出所有Agent | `/list` |
| `@<名称> <问题>` | 与Agent对话 | `@python_expert 如何优化代码？` |

## 配置

设置环境变量：
```bash
export MODEL_NAME="your-model-name"
export MODEL_API_KEY="your-api-key"
export MODEL_BASE_URL="your-api-base-url"
export DINGTALK_CLIENT_ID="your-dingtalk-client-id"
export DINGTALK_CLIENT_SECRET="your-dingtalk-client-secret"
```

## 许可证

Apache License 2.0

