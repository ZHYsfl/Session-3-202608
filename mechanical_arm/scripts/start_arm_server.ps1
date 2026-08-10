# 机械臂 Arm Skill Server 启动脚本 (Windows PowerShell)
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File scripts/start_arm_server.ps1
#   $env:ARM_SERVER_PORT=8001; ./scripts/start_arm_server.ps1
#
# 只做三件事: 切项目目录 -> 设环境变量 -> 启动 API。
# 不自动提权、不装驱动、不改系统。前置: Python 3.10+ 与 requirements.txt 已安装。
param()

# 1) 切到脚本所在目录(项目根)
Set-Location (Split-Path -Parent $PSScriptRoot)

# 2) 环境变量(默认 0.0.0.0:8000; 可被外部覆盖)
if (-not $env:ARM_SERVER_HOST) { $env:ARM_SERVER_HOST = "0.0.0.0" }
if (-not $env:ARM_SERVER_PORT) { $env:ARM_SERVER_PORT = "8000" }
# 可选: 强制真 3D 物理后端
# if (-not $env:ARM_BACKEND) { $env:ARM_BACKEND = "mujoco" }

# 3) 启动 API
Write-Host "Starting Arm Skill Server on ${env:ARM_SERVER_HOST}:${env:ARM_SERVER_PORT} ..."
python scripts/run_api.py
