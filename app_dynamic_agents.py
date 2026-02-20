#!/usr/bin/env python3
"""
AgentScope 动态Agent管理系统
符合AgentScope设计规范，支持通过对话创建、删除、修改Agent
使用AgentScope消息框架(Msg)进行交互
"""
import asyncio
import json
import os
import re
import logging
import time
import hashlib
from datetime import datetime
from typing import Dict, Optional
from flask import Flask, request, jsonify
import threading
import openai

import agentscope
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg

try:
    from dingtalk_stream import AckMessage
    import dingtalk_stream
    DINGTALK_AVAILABLE = True
except ImportError:
    DINGTALK_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/agentscope/logs/dynamic_agents.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置
AGENTS_FILE = "/opt/agentscope/dynamic_agents.json"
MODEL_CONFIG = {
    "model_name": "deepseek-ai/deepseek-v3.2",
    "api_key": "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",
    "base_url": "https://integrate.api.nvidia.com/v1"
}

DINGTALK_CONFIG = {
    "client_id": "dingisxrapdsthpucwio",
    "client_secret": "4ByZlcFtACSDzvcaIM1YTQTtsuAgE-GyxRQ-EVzlbPknjX0Z4SVn7s1BexLjL9Jr"
}

# 消息去重
processed_messages = {}
DEDUP_WINDOW = 60  # 秒 - 增加到60秒

def is_duplicate(message_id: str) -> bool:
    now = time.time()
    if message_id in processed_messages:
        # 只要处理过就认为是重复
        return True
    # 清理过期消息
    expired = [k for k, v in processed_messages.items() if now - v > DEDUP_WINDOW * 2]
    for k in expired:
        del processed_messages[k]
    processed_messages[message_id] = now
    return False

def get_message_id(user_id: str, text: str) -> str:
    return hashlib.md5(f"{user_id}:{text}".encode()).hexdigest()

# 初始化AgentScope
agentscope.init(project='dynamic_agents', logging_path='/opt/agentscope/logs/agentscope.log')

# 创建共享模型和formatter
shared_model = OpenAIChatModel(
    model_name=MODEL_CONFIG["model_name"],
    api_key=MODEL_CONFIG["api_key"],
    client_kwargs={"base_url": MODEL_CONFIG["base_url"]}
)
shared_formatter = OpenAIChatFormatter()


class AgentManager:
    """动态Agent管理器"""
    
    def __init__(self):
        self.agents: Dict[str, ReActAgent] = {}
        self.agent_configs: Dict[str, dict] = {}
        self.load_agents()
    
    def load_agents(self):
        if os.path.exists(AGENTS_FILE):
            try:
                with open(AGENTS_FILE, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
                for name, config in configs.items():
                    self._create_agent_from_config(name, config)
                logger.info(f"Loaded {len(self.agents)} agents")
            except Exception as e:
                logger.error(f"Load agents error: {e}")
    
    def save_agents(self):
        try:
            with open(AGENTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.agent_configs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Save agents error: {e}")
    
    def _create_agent_from_config(self, name: str, config: dict) -> Optional[ReActAgent]:
        try:
            agent = ReActAgent(
                name=name,
                sys_prompt=config.get("sys_prompt", "你是一个有用的助手。"),
                model=shared_model,
                formatter=shared_formatter
            )
            self.agents[name] = agent
            self.agent_configs[name] = config
            return agent
        except Exception as e:
            logger.error(f"Create agent error: {e}")
            return None
    
    def create_agent_from_description_sync(self, description: str) -> Optional[dict]:
        """同步方式创建Agent"""
        try:
            client = openai.OpenAI(
                api_key=MODEL_CONFIG["api_key"],
                base_url=MODEL_CONFIG["base_url"]
            )
            
            prompt = f"""根据以下描述创建智能体配置，只返回JSON：

描述: {description}

返回格式:
{{
    "name": "英文标识符(如python_expert)",
    "display_name": "显示名称",
    "description": "智能体描述",
    "sys_prompt": "系统提示词，定义角色和能力"
}}"""

            response = client.chat.completions.create(
                model=MODEL_CONFIG["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                config = json.loads(json_match.group())
                name = config.get("name", f"agent_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                config["created_at"] = datetime.now().isoformat()
                agent = self._create_agent_from_config(name, config)
                if agent:
                    self.save_agents()
                    return {"name": name, **config}
        except Exception as e:
            logger.error(f"Create agent from description error: {e}")
        return None
    
    def get_agent(self, name: str) -> Optional[ReActAgent]:
        return self.agents.get(name)
    
    def list_agents(self) -> Dict[str, dict]:
        return self.agent_configs
    
    def delete_agent(self, name: str) -> bool:
        if name in self.agents:
            del self.agents[name]
            del self.agent_configs[name]
            self.save_agents()
            return True
        return False
    
    def update_agent(self, name: str, config: dict) -> bool:
        if name in self.agents:
            self.agent_configs[name].update(config)
            agent = self._create_agent_from_config(name, self.agent_configs[name])
            if agent:
                self.save_agents()
                return True
        return False


# 主助手系统提示词
MAIN_ASSISTANT_PROMPT = """你是一个智能助手系统的主控助手。你的职责是帮助用户使用系统功能。

## 系统功能

你可以帮助用户管理专业Agent（智能体）：

1. **创建Agent**: 用户可以说"创建一个Python专家"，你会帮他们生成Agent
2. **查看Agent**: 用户可以问"有哪些Agent"或"列表"
3. **删除Agent**: 用户可以说"删除xxx"
4. **与Agent对话**: 用户可以用 @名称 的方式与特定Agent对话

## 当前可用Agent

{agent_list}

## 使用建议

- 如果用户想创建新Agent，告诉他们可以用 /create 命令或直接描述需求
- 如果用户的问题适合某个专业Agent，建议他们使用 @名称 方式
- 如果用户问系统功能，介绍上述功能

请友好、专业地回应用户。"""


# 初始化管理器
manager = AgentManager()


def get_main_assistant_prompt() -> str:
    agents = manager.list_agents()
    if agents:
        agent_list = "\n".join([f"- @{name}: {cfg.get('display_name', name)} - {cfg.get('description', '')}" 
                                for name, cfg in agents.items()])
    else:
        agent_list = "暂无Agent，可以使用 /create 命令创建"
    return MAIN_ASSISTANT_PROMPT.format(agent_list=agent_list)


def call_agent_sync(agent: ReActAgent, message: str) -> str:
    """同步方式调用AgentScope Agent，使用Msg消息框架"""
    try:
        # 创建AgentScope Msg消息
        msg = Msg(name="user", role="user", content=message)
        
        # 使用asyncio.run在同步上下文中运行异步Agent
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(agent(msg))
            return response.content
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Agent call error: {e}")
        return f"❌ Agent调用错误: {e}"


def process_message_sync(text: str, user_id: str = "default") -> str:
    """同步版本的消息处理，符合AgentScope消息框架"""
    text = text.strip()
    
    # 创建Agent
    if text.startswith("/create "):
        description = text[8:].strip()
        config = manager.create_agent_from_description_sync(description)
        if config:
            return f"✅ Agent创建成功！\n\n名称: @{config['name']}\n显示名: {config.get('display_name', config['name'])}\n描述: {config.get('description', '')}\n\n使用: @{config['name']} 你的问题"
        return "❌ 创建失败，请重试"
    
    # 删除Agent
    if text.startswith("/delete "):
        name = text[8:].strip()
        if manager.delete_agent(name):
            return f"✅ 已删除Agent @{name}"
        return f"❌ Agent @{name} 不存在"
    
    # 列出Agent
    if text in ["/list", "/agents"]:
        agents = manager.list_agents()
        if not agents:
            return "暂无Agent。使用 /create <描述> 创建"
        result = "📋 Agent列表:\n\n"
        for name, cfg in agents.items():
            result += f"• @{name} - {cfg.get('display_name', name)}\n  {cfg.get('description', '')}\n\n"
        return result
    
    # 帮助
    if text == "/help":
        return """🤖 动态Agent系统

/create <描述> - 创建Agent
  例: /create 一个Python专家，擅长代码优化

/delete <名称> - 删除Agent

/list - 查看所有Agent

@<名称> <问题> - 与Agent对话
  例: @python_expert 如何优化代码？"""
    
    # 调用指定Agent - 使用AgentScope消息框架
    agent_match = re.match(r'^@([\w-]+)\s+(.+)$', text)
    if agent_match:
        agent_name = agent_match.group(1)
        message = agent_match.group(2)
        agent = manager.get_agent(agent_name)
        if agent:
            # 使用AgentScope Agent和Msg消息框架
            return call_agent_sync(agent, message)
        return f"❌ Agent @{agent_name} 不存在\n可用: {list(manager.agents.keys())}"
    
    # 默认对话
    try:
        client = openai.OpenAI(
            api_key=MODEL_CONFIG["api_key"],
            base_url=MODEL_CONFIG["base_url"]
        )
        response = client.chat.completions.create(
            model=MODEL_CONFIG["model_name"],
            messages=[
                {"role": "system", "content": get_main_assistant_prompt()},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 错误: {e}"


async def process_message(text: str, user_id: str = "default") -> str:
    """异步版本的消息处理"""
    return process_message_sync(text, user_id)


if DINGTALK_AVAILABLE:
    class DingTalkHandler(dingtalk_stream.ChatbotHandler):
        async def process(self, callback: dingtalk_stream.CallbackMessage):
            try:
                msg = dingtalk_stream.ChatbotMessage.from_dict(callback.data)
                text = msg.text.content.strip()
                user_id = msg.sender_id
                message_id = msg.message_id  # 使用钉钉消息ID
                
                # 去重检查 - 使用钉钉message_id
                if message_id and is_duplicate(message_id):
                    logger.info(f"[DingTalk] Duplicate message ignored: {message_id}")
                    return AckMessage.STATUS_OK, 'OK'
                
                logger.info(f"[DingTalk] {user_id}: {text[:50]}... (msg_id: {message_id})")
                
                # 同步处理消息
                response = process_message_sync(text, user_id)
                
                self.reply_text(response, msg)
                logger.info(f"[DingTalk] Response sent for {message_id}")
                
                return AckMessage.STATUS_OK, 'OK'
            except Exception as e:
                logger.error(f"[DingTalk] Error: {e}")
                return AckMessage.STATUS_OK, 'OK'

    async def start_dingtalk_async():
        try:
            cred = dingtalk_stream.Credential(
                DINGTALK_CONFIG["client_id"],
                DINGTALK_CONFIG["client_secret"]
            )
            client = dingtalk_stream.DingTalkStreamClient(cred)
            client.register_callback_handler(
                dingtalk_stream.ChatbotMessage.TOPIC,
                DingTalkHandler()
            )
            logger.info("[DingTalk] Stream starting...")
            await client.start()
        except Exception as e:
            logger.error(f"[DingTalk] Error: {e}")

    def start_dingtalk():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start_dingtalk_async())


@app.route('/')
def index():
    return jsonify({
        "status": "AgentScope Dynamic Agent System",
        "agents": list(manager.agents.keys()),
        "dingtalk_stream": DINGTALK_AVAILABLE
    })


@app.route('/health')
def health():
    return jsonify({"status": "healthy"})


@app.route('/agents', methods=['GET'])
def list_agents():
    return jsonify(manager.agent_configs)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    msg = data.get('message', '')
    user_id = data.get('user_id', 'default')
    if not msg:
        return jsonify({"error": "No message"}), 400
    
    response = process_message_sync(msg, user_id)
    return jsonify({"response": response})


if __name__ == '__main__':
    if DINGTALK_AVAILABLE:
        t = threading.Thread(target=start_dingtalk, daemon=True)
        t.start()
        logger.info("[DingTalk] Thread started")
    
    logger.info("Starting Flask on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

