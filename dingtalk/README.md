# 钉钉Stream消息推送集成

## 配置步骤

### 1. 创建钉钉企业内部机器人

1. 登录钉钉开发者后台: https://open-dev.dingtalk.com/
2. 创建企业内部应用
3. 获取 Client ID 和 Client Secret

### 2. 配置环境变量

```bash
export DINGTALK_CLIENT_ID="your-client-id"
export DINGTALK_CLIENT_SECRET="your-client-secret"
```

### 3. 权限配置

在钉钉开发者后台，为应用开通以下权限：
- 企业内消息通知: 读取和发送消息

## 消息去重

钉钉Stream可能会重发消息，系统使用 `message_id` 进行去重处理：

```python
# 去重窗口: 60秒
DEDUP_WINDOW = 60

def is_duplicate(message_id: str) -> bool:
    if message_id in processed_messages:
        return True
    processed_messages[message_id] = time.time()
    return False
```

## 测试

发送消息到钉钉机器人，系统会自动响应。

## 故障排查

查看日志：
```bash
tail -f /opt/agentscope/logs/service.log | grep DingTalk
```

