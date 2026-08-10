# Mechanical Arm Server Deployment

面向负责把机械臂 Arm Skill Server 部署到 Linux 服务器的部署人员。
本文件只讲「怎么装、怎么起、怎么验证」, 不涉及 FK/IK 数学。

---

## 1. Requirements

- **Python**: 3.10+（本项目在本机 Python 3.11 实测）
- **OS**: Linux（本文件以 Linux 为主）；Windows 可用 `scripts/start_arm_server.ps1`（见 §3 备选）
- **系统工具**: `git`、`python3` + `venv`、`pip`
- **MuJoCo**: 通过 `pip install mujoco` 安装（requirements.txt 已含 `mujoco>=3.0`），**纯 CPU 可运行，不需要 GPU**
- **GPU**: 非必须。默认模式（hsv 视觉 + scripted 抓取）完全不加载 PyTorch。`torch` 只在显式切换到 `tiny_detector` / `bc` 模式时才被 import（此时 CPU 版 torch 也足够，无需 CUDA）
- **Headless rendering requirements**（服务器通常无显示器）:
  - MuJoCo 相机用**离屏渲染**（`mujoco.Renderer`，不启动任何 GUI viewer），天然 headless。
  - Linux headless 需要 MuJoCo 的 OpenGL 上下文。默认用 `MUJOCO_GL=egl`（EGL 后端）。详见 §7。
  - 若环境缺少 EGL / OpenGL backend，渲染线程会失败并打印明确错误，**本服务不会伪造「已支持 headless 渲染」**。

## 2. Install

本模块位于仓库根目录下的 `mechanical_arm/`，以下命令均在该目录内执行。

```bash
cd mechanical_arm

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

可选（默认模式不需要；只有用 TinyDetector / BC 抓取才需要）:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 3. Start Server

```bash
# 方式一：直接启动（推荐给部署/联调）
python scripts/run_api.py

# 方式二：用启动脚本（自动 cd 到项目根 + 设环境变量）
./scripts/start_arm_server.sh
```

服务器监听 **`0.0.0.0:8000`**（默认）。可用环境变量覆盖：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `ARM_SERVER_HOST` | `0.0.0.0` | 监听地址 |
| `ARM_SERVER_PORT` | `8000` | 监听端口 |
| `ARM_BACKEND` | （读 configs/control.yaml） | 可选覆盖后端：`mujoco`（真 3D 物理）或 `kinematic`（零依赖） |

启动日志会打印 `backend_mode` / `perception_mode` / `grasp_mode` / `pacing` 供确认。

> 默认后端由 `configs/control.yaml` 的 `backend_mode` 决定（当前为 `kinematic`，零依赖）。
> **机械臂模块核心交付是真 3D 物理（MuJoCo）**：如需启用，启动前设 `ARM_BACKEND=mujoco`，
> 或把 `configs/control.yaml` 的 `backend_mode` 改为 `mujoco`。正式 API path 不变。

### Windows 备选

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_arm_server.ps1
```

## 4. Verify

```bash
# 健康检查（部署运维接口，非 Agent 工具）
curl http://127.0.0.1:8000/health
# => {"status":"ok","simulator":"mujoco","api":"ready"}

# 四个 Arm Tool 冒烟
curl http://127.0.0.1:8000/api/v1/get_current_coordinates
# => {"code":0,"result":"我的坐标是0.230,0.000,0.320","error":null}
```

完整冒烟（会真实移动 + 抓取 + 释放，约几秒到几十秒）：
```bash
python scripts/smoke_test_arm_api.py --base-url http://127.0.0.1:8000
```

## 5. Remote Access

- `127.0.0.1` 只代表**服务器本机**。
- 其他机器访问本服务，必须用服务器在局域网/公网的真实地址：
  ```
  http://<SERVER_IP>:8000/api/v1/get_current_coordinates
  ```
- 确认方法：在服务器上 `hostname -I` 拿到内网 IP；从另一台机器 `curl http://<SERVER_IP>:8000/health`。
- 协议文档 `docs/api_of_embodied_tool.md` 中的 Base URL 写 `http://127.0.0.1:8000` 仅表示本机联调，远程联调用实际 IP/域名，API path 不变。

## 6. Firewall

- 需要让服务器网络允许 **TCP 8000**（或你自定义的 `ARM_SERVER_PORT`）。
- 云厂商（阿里云/腾讯云/AWS 等）还需在安全组放行该端口。
- 本部署**不自动修改 firewall**——由部署人员按所在环境执行：
  ```bash
  # 示例（仅示例，按你的发行版操作，不要盲目执行）
  # sudo ufw allow 8000/tcp
  ```

## 7. Headless

- 服务器默认以 headless 运行：`run_api.py` **不启动任何 GUI viewer**（GUI 只存在于 `run_arm_demo.py` / `run_3d_sim.py`）。
- MuJoCo 相机离屏渲染需要 GL 上下文。Linux 无显示器时设置：
  ```bash
  export MUJOCO_GL=egl
  ```
- 若服务器有显示器 / X 环境，也可用默认或 `MUJOCO_GL=osmesa`（需系统安装 OSMesa）。
- 若 EGL/OSMesa 都不可用，MuJoCo 渲染线程会报错（相机取帧失败）——此时 `get_current_coordinates` / `move_to_coordinates`（不依赖视觉）仍可用；`grab_the_block` 需要视觉，会如实返回失败。**请按真实环境确认，不要假设已支持。**

## 8. Logs

- 启动日志直接打印到**终端（stdout/stderr）**。
- 若需落盘，用 `nohup` 或 systemd 重定向，例如：
  ```bash
  nohup python scripts/run_api.py > arm_server.log 2>&1 &
  tail -f arm_server.log
  ```
- 服务器内部异常会由 FastAPI 记录 traceback 并返回 `{"code":500,"error":"内部异常: <类名>"}`；详细堆栈在服务端日志。

## 9. Stop Server

- 前台启动：按 `Ctrl+C`。
- 后台启动（`nohup ... &`）：`kill <PID>`（`ps aux | grep run_api.py` 查 PID）。
- systemd 管理时用 `systemctl stop <service>`。
