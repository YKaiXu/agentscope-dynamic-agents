# AgentScope Dynamic Agents System

> 基于 [AgentScope](https://github.com/agentscope-ai/agentscope) 框架的动态Agent管理系统

## 功能特性

- ✅ 动态创建Agent（支持自定义简短名字）
- ✅ 独立LLM模型分配
- ✅ 结构化命令支持
- ✅ 完全符合AgentScope 1.0.16规范

## 创建Agent

### 方式1: 简单创建
```
/create 一个Python专家           # 自动生成名字
/create py 一个Python专家        # 指定名字"py"
```

### 方式2: 结构化创建
```
/create name=py display="Python专家" desc="Python编程专家" prompt="你是Python专家" model=default
```

### 方式3: JSON创建
```
/create {"name":"py","display_name":"Python专家","sys_prompt":"...","model":"default"}
```

## AgentScope Agent字段

| 字段 | 必需 | 说明 |
|------|------|------|
| name | ✅ | Agent名称（用于@调用） |
| sys_prompt | ✅ | 系统提示词 |
| model | ✅ | LLM模型 |
| formatter | ✅ | 消息格式化器 |
| display_name | ❌ | 显示名称 |
| description | ❌ | 描述 |

## 命令列表

| 命令 | 功能 |
|------|------|
| `/create [name] <描述>` | 创建Agent |
| `/delete <名称>` | 删除Agent |
| `/setmodel <Agent> <模型>` | 设置模型 |
| `/list` | 列出Agent |
| `/models` | 列出模型 |
| `@<名称> <问题>` | 与Agent对话 |

## 配置文件

### dynamic_agents.json
```json
{
  "py": {
    "name": "py",
    "display_name": "Python专家",
    "sys_prompt": "你是Python专家...",
    "model": "default"
  }
}
```

### models_config.json
```json
{
  "default": {
    "model_name": "gpt-4",
    "api_key": "xxx",
    "base_url": "https://api.openai.com/v1"
  }
}
```

## 相关项目

- [AgentScope](https://github.com/agentscope-ai/agentscope)
- [AgentScope DingTalk Bot](https://github.com/YKaiXu/agentscope-dingtalk-bot)

## 许可证

Apache License 2.0

