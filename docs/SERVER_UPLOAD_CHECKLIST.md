# Server Upload Checklist

把机械臂 Arm Skill Server 上传到目标 Linux 服务器前的核对清单。
每一项请**核对后打勾**（填写人员自己执行），上传命令在最后一段。

---

## 0. 前置确认

- [ ] 已拿到目标服务器真实地址 `SERVER_IP`（本清单不伪造 IP，用占位符 `<SERVER_IP>`）
- [ ] 已拿到 SSH 用户名（占位符 `<USER>`）与密钥/密码，且本机可连通
- [ ] 已确认服务器系统（本清单以 Linux 为主）

## 1. 代码与配置（必须上传）

- [ ] 项目根目录整体上传（保持相对路径不变：`configs/`、`models/`、`docs/`、`scripts/`、`arm_skill_server/`）
- [ ] `requirements.txt` 已包含运行时依赖：fastapi / uvicorn / pydantic / numpy / PyYAML / opencv-python / mujoco / httpx / torch（torch 仅 `tiny_detector`/`bc` 模式需要）
- [ ] `configs/`：control.yaml / robot.yaml / simulation.yaml / vision.yaml 四个都在，且与代码期望一致
- [ ] `models/`：grasp_policy.pt、tiny_detector.pt 存在（默认 hsv+scripted 模式不强依赖，但文件应随包上传）
- [ ] `docs/api_of_embodied_tool.md` 协议文档随包（交接/联调必读）

## 2. 服务器端安装

- [ ] Python 3.10+ 已安装
- [ ] `pip install -r requirements.txt` 执行成功（无 GPU 也无需 torch CUDA）
- [ ] headless 渲染：Linux 无显示器时确认 `MUJOCO_GL=egl` 可用（或 OSMesa）
- [ ] 端口可用：`ARM_SERVER_PORT`（默认 8000）未被占用

## 3. 启动与验证

- [ ] `python scripts/run_api.py`（或 `./scripts/start_arm_server.sh`）能启动
- [ ] 启动日志打印 backend_mode / perception_mode / grasp_mode / pacing
- [ ] `curl http://127.0.0.1:8000/health` 返回 `{"status":"ok",...}`
- [ ] `curl http://127.0.0.1:8000/api/v1/get_current_coordinates` 返回坐标 result
- [ ] 冒烟：`python scripts/smoke_test_arm_api.py --base-url http://127.0.0.1:8000` → `ALL ARM API SMOKE TESTS PASSED`
- [ ] 远程：从另一台机器 `curl http://<SERVER_IP>:8000/health` 可达
- [ ] 防火墙/安全组已放行 TCP `ARM_SERVER_PORT`（云厂商还需配安全组）
- [ ] 进程守护：前台 `nohup` 或 systemd 已配好（见 `docs/ARM_SERVER_DEPLOYMENT.md` §8/§9）

## 4. 回归（可选但推荐，服务器端跑）

- [ ] `python -m pytest arm_skill_server/tests -m "not gui"` 全绿
- [ ] 手动四工具各调一遍（含负例：不可达 / 无该颜色 / 非法请求 400）

## 5. 上传命令（占位符替换后再执行）

```bash
# scp 整个项目（在项目根目录的上一级执行）
scp -r pr2 <USER>@<SERVER_IP>:<REMOTE_PATH>/pr2

# 或 rsync（推荐，增量）
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
  ./pr2 <USER>@<SERVER_IP>:<REMOTE_PATH>/
```

- `<USER>`：SSH 用户名
- `<SERVER_IP>`：服务器真实 IP/域名
- `<REMOTE_PATH>`：服务器上要放代码的目录

> 提示：`.venv`、`__pycache__`、`*.pyc`、日志文件**不要上传**（上面命令已排除）。
> 上传后登录服务器：`ssh <USER>@<SERVER_IP>`，按 §3 安装并验证。
