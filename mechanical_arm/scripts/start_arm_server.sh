#!/usr/bin/env bash
# 机械臂 Arm Skill Server 启动脚本(Linux)
#
# 用法:
#   ./scripts/start_arm_server.sh                 # 默认 0.0.0.0:8000
#   ARM_SERVER_HOST=0.0.0.0 ARM_SERVER_PORT=8001 ./scripts/start_arm_server.sh
#
# 只做三件事: 切项目目录 -> 设环境变量 -> 启动 API。
# 不自动 sudo、不装驱动、不改系统。前置要求(由部署人员自行满足):
#   - Python 3.10+ 已安装
#   - `pip install -r requirements.txt` 已完成(或已有 venv)
#   - 机械臂模块核心(MuJoCo 3D)需 `pip install mujoco`(requirements 已含)
set -euo pipefail

# 1) 切到脚本所在目录(项目根), 保证相对路径(configs/ models/)正确
cd "$(dirname "$(readlink -f "$0")")/.."

# 2) 环境变量(默认 0.0.0.0:8000; 可被外部覆盖)
export ARM_SERVER_HOST="${ARM_SERVER_HOST:-0.0.0.0}"
export ARM_SERVER_PORT="${ARM_SERVER_PORT:-8000}"
# 可选: 强制真 3D 物理后端(默认用 configs/control.yaml 的 backend_mode)
# export ARM_BACKEND="${ARM_BACKEND:-mujoco}"

# 可选: 使用已有虚拟环境(取消注释并按需修改路径)
# if [ -f .venv/bin/activate ]; then
#   source .venv/bin/activate
# fi

# 3) 启动 API
echo "Starting Arm Skill Server on ${ARM_SERVER_HOST}:${ARM_SERVER_PORT} ..."
exec python scripts/run_api.py
