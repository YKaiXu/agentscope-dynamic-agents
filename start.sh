#!/bin/bash
# AgentScope Dynamic Agents 启动脚本

# 创建日志目录
mkdir -p /opt/agentscope/logs

# 激活虚拟环境
source /opt/agentscope/venv/bin/activate

# 启动服务
cd /opt/agentscope
python app_dynamic_agents.py

