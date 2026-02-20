#!/usr/bin/env python3
"""
AgentScope Dynamic Agent Management System
支持通过对话创建Agent并分配独立LLM
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

# Logging configuration
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

# Configuration
AGENTS_FILE = "/opt/agentscope/dynamic_agents.json"
MODELS_FILE = "/opt/agentscope/models_config.json"

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "model_name": "deepseek-ai/deepseek-v3.2",
    "api_key": "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",
    "base_url": "https://integrate.api.nvidia.com/v1"
}

DINGTALK_CONFIG = {
    "client_id": "dingisxrapdsthpucwio",
    "client_secret": "4ByZlcFtACSDzvcaIM1YTQTtsuAgE-GyxRQ-EVzlbPknjX0Z4SVn7s1BexLjL9Jr"
}

# Message deduplication
processed_messages = {}
DEDUP_WINDOW = 60

def is_duplicate(message_id: str) -> bool:
    now = time.time()
    if message_id in processed_messages:
        return True
    expired = [k for k, v in processed_messages.items() if now - v > DEDUP_WINDOW * 2]
    for k in expired:
        del processed_messages[k]
    processed_messages[message_id] = now
    return False

# Initialize AgentScope
agentscope.init(project='dynamic_agents', logging_path='/opt/agentscope/logs/agentscope.log')

# Shared formatter
shared_formatter = OpenAIChatFormatter()


class ModelManager:
    """LLM模型管理器"""
    
    def __init__(self):
        self.models: Dict[str, OpenAIChatModel] = {}
        self.model_configs: Dict[str, dict] = {}
        self.load_models()
    
    def load_models(self):
        if os.path.exists(MODELS_FILE):
            try:
                with open(MODELS_FILE, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
                for name, config in configs.items():
                    self._create_model(name, config)
                logger.info(f"Loaded {len(self.models)} models")
            except Exception as e:
                logger.error(f"Load models error: {e}")
        # 确保有默认模型
        if "default" not in self.models:
            self._create_model("default", DEFAULT_MODEL_CONFIG)
            self.save_models()
    
    def save_models(self):
        try:
            with open(MODELS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.model_configs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Save models error: {e}")
    
    def _create_model(self, name: str, config: dict) -> Optional[OpenAIChatModel]:
        try:
            model = OpenAIChatModel(
                model_name=config.get("model_name", "gpt-3.5-turbo"),
                api_key=config.get("api_key", ""),
                client_kwargs={"base_url": config.get("base_url", "https://api.openai.com/v1")}
            )
            self.models[name] = model
            self.model_configs[name] = config
            return model
        except Exception as e:
            logger.error(f"Create model error: {e}")
            return None
    
    def get_model(self, name: str = "default") -> Optional[OpenAIChatModel]:
        return self.models.get(name, self.models.get("default"))
    
    def list_models(self) -> Dict[str, dict]:
        return self.model_configs
    
    def add_model(self, name: str, config: dict) -> bool:
        if self._create_model(name, config):
            self.save_models()
            return True
        return False
    
    def delete_model(self, name: str) -> bool:
        if name == "default":
            return False  # 不能删除默认模型
        if name in self.models:
            del self.models[name]
            del self.model_configs[name]
            self.save_models()
            return True
        return False


class AgentManager:
    """动态Agent管理器"""
    
    def __init__(self, model_manager: ModelManager):
        self.agents: Dict[str, ReActAgent] = {}
        self.agent_configs: Dict[str, dict] = {}
        self.model_manager = model_manager
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
            # 获取Agent指定的模型，如果没有则使用默认模型
            model_name = config.get("model", "default")
            model = self.model_manager.get_model(model_name)
            
            if not model:
                model = self.model_manager.get_model("default")
            
            agent = ReActAgent(
                name=name,
                sys_prompt=config.get("sys_prompt", "你是一个有用的助手。"),
                model=model,
                formatter=shared_formatter
            )
            self.agents[name] = agent
            self.agent_configs[name] = config
            return agent
        except Exception as e:
            logger.error(f"Create agent error: {e}")
            return None
    
    def create_agent_from_description_sync(self, description: str, model: str = "default") -> Optional[dict]:
        """创建Agent并指定模型"""
        try:
            client = openai.OpenAI(
                api_key=DEFAULT_MODEL_CONFIG["api_key"],
                base_url=DEFAULT_MODEL_CONFIG["base_url"]
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
                model=DEFAULT_MODEL_CONFIG["model_name"],
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
                config["model"] = model  # 设置模型
                agent = self._create_agent_from_config(name, config)
                if agent:
                    self.save_agents()
                    return {"name": name, **config}
        except Exception as e:
            logger.error(f"Create agent from description error: {e}")
        return None
    
    def set_agent_model(self, agent_name: str, model_name: str) -> bool:
        """设置Agent使用的模型"""
        if agent_name not in self.agent_configs:
            return False
        if model_name not in self.model_manager.models:
            return False
        
        self.agent_configs[agent_name]["model"] = model_name
        # 重新创建Agent以应用新模型
        self._create_agent_from_config(agent_name, self.agent_configs[agent_name])
        self.save_agents()
        return True
    
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


# Main assistant system prompt
MAIN_ASSISTANT_PROMPT = """你是一个智能助手系统的主控助手。你的职责是帮助用户使用系统功能。

## 系统功能

你可以帮助用户管理专业Agent（智能体）和LLM模型：

### Agent管理
1. **创建Agent**: /create <描述> [模型名]
   - 例: /create 一个Python专家
   - 例: /create 一个Python专家 default
   
2. **删除Agent**: /delete <名称>

3. **查看Agent**: /list 或 /agents

4. **设置Agent模型**: /setmodel <Agent名> <模型名>

### 模型管理
1. **查看模型**: /models

2. **添加模型**: /addmodel <名称> <模型名> <API Key> <Base URL>

3. **删除模型**: /delmodel <名称>

## 当前可用Agent

{agent_list}

## 当前可用模型

{model_list}

请友好、专业地回应用户。"""


# Initialize managers
model_manager = ModelManager()
agent_manager = AgentManager(model_manager)


def get_main_assistant_prompt() -> str:
    agents = agent_manager.list_agents()
    if agents:
        agent_list = "\n".join([f"- @{name}: {cfg.get('display_name', name)} (模型: {cfg.get('model', 'default')})" 
                                for name, cfg in agents.items()])
    else:
        agent_list = "暂无Agent，使用 /create <描述> 创建"
    
    models = model_manager.list_models()
    model_list = "\n".join([f"- {name}: {cfg.get('model_name', 'unknown')}" 
                           for name, cfg in models.items()])
    
    return MAIN_ASSISTANT_PROMPT.format(agent_list=agent_list, model_list=model_list)


def call_agent_sync(agent: ReActAgent, message: str) -> str:
    """同步方式调用AgentScope Agent"""
    try:
        msg = Msg(name="user", role="user", content=message)
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
    """同步版本的消息处理"""
    text = text.strip()
    
    # === 模型管理命令 ===
    
    # 列出模型
    if text in ["/models", "/listmodels"]:
        models = model_manager.list_models()
        if not models:
            return "暂无模型配置"
        result = "📋 模型列表:\n\n"
        for name, cfg in models.items():
            result += f"• {name}: {cfg.get('model_name', 'unknown')}\n"
            result += f"  Base URL: {cfg.get('base_url', 'N/A')}\n\n"
        return result
    
    # 添加模型
    if text.startswith("/addmodel "):
        parts = text[10:].split()
        if len(parts) >= 4:
            name, model_name, api_key, base_url = parts[0], parts[1], parts[2], parts[3]
            config = {
                "model_name": model_name,
                "api_key": api_key,
                "base_url": base_url
            }
            if model_manager.add_model(name, config):
                return f"✅ 模型 '{name}' 添加成功"
            return "❌ 模型添加失败"
        return "用法: /addmodel <名称> <模型名> <API Key> <Base URL>"
    
    # 删除模型
    if text.startswith("/delmodel "):
        name = text[10:].strip()
        if model_manager.delete_model(name):
            return f"✅ 模型 '{name}' 已删除"
        return f"❌ 无法删除模型 '{name}' (默认模型不可删除)"
    
    # 设置Agent模型
    if text.startswith("/setmodel "):
        parts = text[10:].split()
        if len(parts) >= 2:
            agent_name, model_name = parts[0], parts[1]
            if agent_manager.set_agent_model(agent_name, model_name):
                return f"✅ Agent '@{agent_name}' 已设置为使用模型 '{model_name}'"
            return f"❌ 设置失败，请检查Agent和模型名称"
        return "用法: /setmodel <Agent名> <模型名>"
    
    # === Agent管理命令 ===
    
    # 创建Agent (支持指定模型)
    if text.startswith("/create "):
        rest = text[8:].strip()
        parts = rest.rsplit(None, 1)  # 从右边分割一次
        
        # 检查最后一个词是否是模型名
        model_name = "default"
        description = rest
        
        if len(parts) == 2:
            potential_model = parts[1]
            if potential_model in model_manager.models:
                model_name = potential_model
                description = parts[0]
        
        config = agent_manager.create_agent_from_description_sync(description, model_name)
        if config:
            return f"✅ Agent创建成功！\n\n名称: @{config['name']}\n显示名: {config.get('display_name', config['name'])}\n描述: {config.get('description', '')}\n模型: {config.get('model', 'default')}\n\n使用: @{config['name']} 你的问题"
        return "❌ 创建失败，请重试"
    
    # 删除Agent
    if text.startswith("/delete "):
        name = text[8:].strip()
        if agent_manager.delete_agent(name):
            return f"✅ 已删除Agent @{name}"
        return f"❌ Agent @{name} 不存在"
    
    # 列出Agent
    if text in ["/list", "/agents"]:
        agents = agent_manager.list_agents()
        if not agents:
            return "暂无Agent。使用 /create <描述> 创建"
        result = "📋 Agent列表:\n\n"
        for name, cfg in agents.items():
            result += f"• @{name} - {cfg.get('display_name', name)}\n"
            result += f"  描述: {cfg.get('description', '')}\n"
            result += f"  模型: {cfg.get('model', 'default')}\n\n"
        return result
    
    # 帮助
    if text == "/help":
        return """🤖 动态Agent系统

=== Agent管理 ===
/create <描述> [模型名] - 创建Agent
  例: /create 一个Python专家
  例: /create 一个Python专家 gpt4

/delete <名称> - 删除Agent

/setmodel <Agent名> <模型名> - 设置Agent使用的模型

/list - 查看所有Agent

=== 模型管理 ===
/models - 查看所有模型

/addmodel <名称> <模型名> <API Key> <Base URL> - 添加模型

/delmodel <名称> - 删除模型

=== 对话 ===
@<名称> <问题> - 与Agent对话"""
    
    # 调用指定Agent
    agent_match = re.match(r'^@([\w-]+)\s+(.+)$', text)
    if agent_match:
        agent_name = agent_match.group(1)
        message = agent_match.group(2)
        agent = agent_manager.get_agent(agent_name)
        if agent:
            return call_agent_sync(agent, message)
        return f"❌ Agent @{agent_name} 不存在\n可用: {list(agent_manager.agents.keys())}"
    
    # 默认对话
    try:
        client = openai.OpenAI(
            api_key=DEFAULT_MODEL_CONFIG["api_key"],
            base_url=DEFAULT_MODEL_CONFIG["base_url"]
        )
        response = client.chat.completions.create(
            model=DEFAULT_MODEL_CONFIG["model_name"],
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
    return process_message_sync(text, user_id)


if DINGTALK_AVAILABLE:
    class DingTalkHandler(dingtalk_stream.ChatbotHandler):
        async def process(self, callback: dingtalk_stream.CallbackMessage):
            try:
                msg = dingtalk_stream.ChatbotMessage.from_dict(callback.data)
                text = msg.text.content.strip()
                user_id = msg.sender_id
                message_id = msg.message_id
                
                if message_id and is_duplicate(message_id):
                    logger.info(f"[DingTalk] Duplicate message ignored: {message_id}")
                    return AckMessage.STATUS_OK, 'OK'
                
                logger.info(f"[DingTalk] {user_id}: {text[:50]}... (msg_id: {message_id})")
                
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
        "agents": list(agent_manager.agents.keys()),
        "models": list(model_manager.models.keys()),
        "dingtalk_stream": DINGTALK_AVAILABLE
    })


@app.route('/health')
def health():
    return jsonify({"status": "healthy"})


@app.route('/agents', methods=['GET'])
def list_agents():
    return jsonify(agent_manager.agent_configs)


@app.route('/models', methods=['GET'])
def list_models():
    return jsonify(model_manager.model_configs)


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

