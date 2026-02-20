#!/usr/bin/env python3
"""
AgentScope Dynamic Agent Management System
支持结构化命令创建Agent，允许自定义简短名字
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

AGENTS_FILE = "/opt/agentscope/dynamic_agents.json"
MODELS_FILE = "/opt/agentscope/models_config.json"

DEFAULT_MODEL_CONFIG = {
    "model_name": "deepseek-ai/deepseek-v3.2",
    "api_key": "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",
    "base_url": "https://integrate.api.nvidia.com/v1"
}

DINGTALK_CONFIG = {
    "client_id": "dingisxrapdsthpucwio",
    "client_secret": "4ByZlcFtACSDzvcaIM1YTQTtsuAgE-GyxRQ-EVzlbPknjX0Z4SVn7s1BexLjL9Jr"
}

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

agentscope.init(project='dynamic_agents', logging_path='/opt/agentscope/logs/agentscope.log')
shared_formatter = OpenAIChatFormatter()


class ModelManager:
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
            return False
        if name in self.models:
            del self.models[name]
            del self.model_configs[name]
            self.save_models()
            return True
        return False


class AgentManager:
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
    
    def create_agent_with_config(self, config: dict) -> Optional[dict]:
        """使用完整配置创建Agent"""
        try:
            name = config.get("name")
            if not name:
                return None
            
            # 验证name格式（只允许字母、数字、下划线、短横线）
            if not re.match(r'^[\w-]+$', name):
                return None
            
            # 设置默认值
            config.setdefault("display_name", name)
            config.setdefault("description", "")
            config.setdefault("model", "default")
            config.setdefault("sys_prompt", "你是一个有用的助手。")
            config["created_at"] = datetime.now().isoformat()
            
            agent = self._create_agent_from_config(name, config)
            if agent:
                self.save_agents()
                return config
        except Exception as e:
            logger.error(f"Create agent with config error: {e}")
        return None
    
    def create_agent_from_description(self, description: str, name: str = None, model: str = "default") -> Optional[dict]:
        """从描述创建Agent，可选指定名字"""
        try:
            client = openai.OpenAI(
                api_key=DEFAULT_MODEL_CONFIG["api_key"],
                base_url=DEFAULT_MODEL_CONFIG["base_url"]
            )
            
            prompt = f"""根据以下描述创建智能体配置，只返回JSON：

描述: {description}

返回格式:
{{
    "display_name": "显示名称（中文）",
    "description": "智能体描述",
    "sys_prompt": "系统提示词，定义角色和能力"
}}

注意：
- name字段不需要返回，会单独指定
- sys_prompt要详细，定义Agent的专业能力和回答风格"""

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
                # 使用指定的name或生成简短name
                if not name:
                    # 从display_name生成简短name
                    display_name = config.get("display_name", "agent")
                    name = self._generate_short_name(display_name)
                
                config["name"] = name
                config["model"] = model
                return self.create_agent_with_config(config)
        except Exception as e:
            logger.error(f"Create agent from description error: {e}")
        return None
    
    def _generate_short_name(self, display_name: str) -> str:
        """从显示名称生成简短的英文名"""
        # 常见中文到英文的映射
        name_map = {
            "python": "py", "java": "java", "前端": "fe", "后端": "be",
            "数据": "data", "分析": "ana", "专家": "pro", "助手": "bot",
            "工程师": "eng", "开发": "dev", "设计": "design", "产品": "pm",
            "测试": "qa", "运维": "ops", "安全": "sec", "算法": "algo",
            "机器学习": "ml", "深度学习": "dl", "人工智能": "ai",
            "系统": "sys", "网络": "net", "数据库": "db", "架构": "arch"
        }
        
        name = display_name.lower()
        for cn, en in name_map.items():
            if cn in name:
                return en
        
        # 默认使用时间戳
        return f"a{datetime.now().strftime('%m%d%H%M')}"
    
    def set_agent_model(self, agent_name: str, model_name: str) -> bool:
        if agent_name not in self.agent_configs:
            return False
        if model_name not in self.model_manager.models:
            return False
        self.agent_configs[agent_name]["model"] = model_name
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


# 命令帮助信息
COMMAND_HELP = {
    "agenthelp": """📌 /agenthelp - 查看所有命令

可用命令:
  /create   - 创建Agent
  /delete   - 删除Agent
  /setmodel - 设置Agent模型
  /list     - 列出所有Agent
  /models   - 列出所有模型
  /addmodel - 添加模型
  /delmodel - 删除模型

查看详细帮助: /命令 help
例如: /create help""",
    "create": """📌 /create - 创建Agent

用法:
  /create <描述>                    # 自动生成名字
  /create <名字> <描述>             # 指定名字
  /create <名字> <描述> <模型>      # 指定名字和模型

结构化创建:
  /create name=py display="Python专家" desc="描述" prompt="提示词" model=default

JSON创建:
  /create {"name":"py","display_name":"Python专家","sys_prompt":"..."}

示例:
  /create 一个Python专家
  /create py 一个Python专家
  /create py 一个Python专家 gpt4""",
    "delete": """📌 /delete - 删除Agent

用法:
  /delete <Agent名称>

示例:
  /delete py
  /delete python_expert

注意: 删除后无法恢复""",
    "setmodel": """📌 /setmodel - 设置Agent使用的模型

用法:
  /setmodel <Agent名称> <模型名称>

示例:
  /setmodel py gpt4
  /setmodel python_expert default

查看可用模型: /models""",
    "list": """📌 /list - 列出所有Agent

用法:
  /list
  /agents

显示: 名称、显示名、使用的模型""",
    "models": """📌 /models - 列出所有模型

用法:
  /models

显示: 模型名称、模型类型、Base URL""",
    "addmodel": """📌 /addmodel - 添加模型

用法:
  /addmodel <名称> <模型名> <API Key> <Base URL>

示例:
  /addmodel gpt4 gpt-4 sk-xxx https://api.openai.com/v1
  /addmodel deepseek deepseek-chat sk-xxx https://api.deepseek.com/v1

注意: API Key会保存在配置文件中""",
    "delmodel": """📌 /delmodel - 删除模型

用法:
  /delmodel <模型名称>

示例:
  /delmodel gpt4

注意: 
  - 默认模型(default)无法删除
  - 删除后使用该模型的Agent会自动切换到default""",
    "chat": """📌 @<名称> - 与Agent对话

用法:
  @<Agent名称> <问题>

示例:
  @py 如何优化Python代码？
  @fe Vue和React有什么区别？

注意: Agent名称区分大小写""",
}


MAIN_ASSISTANT_PROMPT = """你是一个智能助手系统的主控助手。

## Agent管理命令

### 方式1: 简单创建
/create <描述>                    # 自动生成简短名字
/create py <描述>                 # 指定名字为"py"

### 方式2: 结构化创建
/create name=py display="Python专家" desc="Python编程专家" prompt="你是Python专家..." model=default

### 方式3: JSON创建
/create {"name":"py","display_name":"Python专家","sys_prompt":"..."}

### 其他命令
/delete <名称>        # 删除Agent
/setmodel <Agent> <模型>  # 设置模型
/list                 # 列出Agent

## 模型管理命令
/models               # 列出模型
/addmodel <名称> <模型> <Key> <URL>  # 添加模型

## 当前Agent
{agent_list}

## 当前模型
{model_list}"""

model_manager = ModelManager()
agent_manager = AgentManager(model_manager)


def get_main_assistant_prompt() -> str:
    agents = agent_manager.list_agents()
    if agents:
        agent_list = "\n".join([f"- @{name}: {cfg.get('display_name', name)} (模型: {cfg.get('model', 'default')})" 
                                for name, cfg in agents.items()])
    else:
        agent_list = "暂无Agent"
    
    models = model_manager.list_models()
    model_list = "\n".join([f"- {name}: {cfg.get('model_name', 'unknown')}" 
                           for name, cfg in models.items()])
    
    return MAIN_ASSISTANT_PROMPT.format(agent_list=agent_list, model_list=model_list)


def parse_structured_create(text: str) -> Optional[dict]:
    """解析结构化创建命令"""
    text = text.strip()
    
    # 方式1: JSON格式
    if text.startswith('{'):
        try:
            return json.loads(text)
        except:
            pass
    
    # 方式2: key=value格式
    if '=' in text:
        config = {}
        # 匹配 key="value" 或 key=value
        pattern = r'(\w+)=(?:"([^"]*)"|(\S+))'
        matches = re.findall(pattern, text)
        for key, val1, val2 in matches:
            value = val1 if val1 else val2
            if key == 'name':
                config['name'] = value
            elif key == 'display':
                config['display_name'] = value
            elif key == 'desc':
                config['description'] = value
            elif key == 'prompt':
                config['sys_prompt'] = value
            elif key == 'model':
                config['model'] = value
        return config if config else None
    
    return None


def call_agent_sync(agent: ReActAgent, message: str) -> str:
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
    text = text.strip()
    
    # === 总命令帮助 ===
    if text in ["/agenthelp", "/commands", "/cmds"]:
        return COMMAND_HELP.get("agenthelp", "无帮助信息")
    
    # === 各命令单独帮助 ===
    if text in ["/create help", "/create ?"]:
        return COMMAND_HELP.get("create", "无帮助信息")
    if text in ["/delete help", "/delete ?"]:
        return COMMAND_HELP.get("delete", "无帮助信息")
    if text in ["/setmodel help", "/setmodel ?"]:
        return COMMAND_HELP.get("setmodel", "无帮助信息")
    if text in ["/list help", "/list ?", "/agents help", "/agents ?"]:
        return COMMAND_HELP.get("list", "无帮助信息")
    if text in ["/models help", "/models ?"]:
        return COMMAND_HELP.get("models", "无帮助信息")
    if text in ["/addmodel help", "/addmodel ?"]:
        return COMMAND_HELP.get("addmodel", "无帮助信息")
    if text in ["/delmodel help", "/delmodel ?"]:
        return COMMAND_HELP.get("delmodel", "无帮助信息")
    if text in ["/chat help", "/chat ?"]:
        return COMMAND_HELP.get("chat", "无帮助信息")
    
    # === 模型管理 ===
    if text in ["/models", "/listmodels"]:
        models = model_manager.list_models()
        if not models:
            return "暂无模型配置"
        result = "📋 模型列表:\n\n"
        for name, cfg in models.items():
            result += f"• {name}: {cfg.get('model_name', 'unknown')}\n"
        return result
    
    if text.startswith("/addmodel "):
        parts = text[10:].split()
        if len(parts) >= 4:
            name, model_name, api_key, base_url = parts[0], parts[1], parts[2], parts[3]
            if model_manager.add_model(name, {"model_name": model_name, "api_key": api_key, "base_url": base_url}):
                return f"✅ 模型 '{name}' 添加成功"
            return "❌ 模型添加失败"
        return "用法: /addmodel <名称> <模型名> <API Key> <Base URL>"
    
    if text.startswith("/delmodel "):
        name = text[10:].strip()
        if model_manager.delete_model(name):
            return f"✅ 模型 '{name}' 已删除"
        return f"❌ 无法删除模型 '{name}'"
    
    # === Agent管理 ===
    if text.startswith("/create "):
        rest = text[8:].strip()
        
        # 检查是否是结构化格式
        structured = parse_structured_create(rest)
        if structured:
            config = agent_manager.create_agent_with_config(structured)
            if config:
                return f"✅ Agent创建成功！\n\n名称: @{config['name']}\n显示名: {config.get('display_name', config['name'])}\n模型: {config.get('model', 'default')}\n\n使用: @{config['name']} 你的问题"
            return "❌ 创建失败，请检查配置格式"
        
        # 检查是否指定了名字
        parts = rest.split(None, 1)
        if len(parts) == 2 and re.match(r'^[\w-]+$', parts[0]) and len(parts[0]) <= 10:
            # 第一个词是名字（短于10个字符的英文）
            name = parts[0]
            description = parts[1]
        else:
            name = None
            description = rest
        
        # 检查描述末尾是否指定了模型
        model = "default"
        desc_parts = description.rsplit(None, 1)
        if len(desc_parts) == 2 and desc_parts[1] in model_manager.models:
            model = desc_parts[1]
            description = desc_parts[0]
        
        config = agent_manager.create_agent_from_description(description, name, model)
        if config:
            return f"✅ Agent创建成功！\n\n名称: @{config['name']}\n显示名: {config.get('display_name', config['name'])}\n描述: {config.get('description', '')}\n模型: {config.get('model', 'default')}\n\n使用: @{config['name']} 你的问题"
        return "❌ 创建失败，请重试"
    
    if text.startswith("/delete "):
        name = text[8:].strip()
        if agent_manager.delete_agent(name):
            return f"✅ 已删除Agent @{name}"
        return f"❌ Agent @{name} 不存在"
    
    if text.startswith("/setmodel "):
        parts = text[10:].split()
        if len(parts) >= 2:
            agent_name, model_name = parts[0], parts[1]
            if agent_manager.set_agent_model(agent_name, model_name):
                return f"✅ Agent '@{agent_name}' 已设置为使用模型 '{model_name}'"
            return "❌ 设置失败，请检查Agent和模型名称"
        return "用法: /setmodel <Agent名> <模型名>"
    
    if text in ["/list", "/agents"]:
        agents = agent_manager.list_agents()
        if not agents:
            return "暂无Agent。使用 /create <描述> 创建"
        result = "📋 Agent列表:\n\n"
        for name, cfg in agents.items():
            result += f"• @{name} - {cfg.get('display_name', name)}\n"
            result += f"  模型: {cfg.get('model', 'default')}\n\n"
        return result
    
    if text == "/help":
        return COMMAND_HELP.get("agenthelp", "无帮助信息")
    
    # 调用Agent
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

