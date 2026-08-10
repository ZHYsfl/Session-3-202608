# Mechanical Arm Module Handoff

机械臂模块交接文档。给接手人：如何理解、如何调用、如何排查。
本文件不重复协议细节（见 `docs/api_of_embodied_tool.md`，协议已冻结），
只补交接所需的运行口径。

---

## 1. 模块范围（本次交付）

交付范围 = **1. 机械臂仿真　2. 机械臂控制　3. 视觉/抓取能力　4. 四个机械臂执行工具　5. 通过 REST API 暴露给 coworker 的 Arm Agent**。

- 四个 Arm Tool（REST 路径见协议文档 `docs/api_of_embodied_tool.md`）：
  `get_current_coordinates` / `move_to_coordinates` / `grab_the_block` / `release_the_block`
- 启动入口：`scripts/run_api.py`（headless，不依赖 GUI）
- 本模块之外的 Agent / Voice / STT / TTS / 双 Agent orchestration / Real Robot / Sim2Real
  **不在本次交接范围**，勿在机械臂模块内开发它们。

## 2. Base URL

- 本机联调：`http://127.0.0.1:8000`
- 远程联调：`http://<SERVER_IP>:8000`（服务器实际 IP，端口由 `ARM_SERVER_PORT` 决定）
- 运维健康检查：`GET /health` → `{"status":"ok","simulator":"<backend>","api":"ready"}`

## 3. 启动

本模块位于仓库根目录下的 `mechanical_arm/`，以下命令均在该目录内执行。

```bash
cd mechanical_arm
pip install -r requirements.txt
python scripts/run_api.py            # 默认 0.0.0.0:8000
# 或: ./scripts/start_arm_server.sh
```

环境变量：`ARM_SERVER_HOST`（默认 0.0.0.0）、`ARM_SERVER_PORT`（默认 8000）、
可选 `ARM_BACKEND=mujoco|kinematic`（覆盖 configs/control.yaml 的 backend_mode）。

## 4. 四个 Tool 的协议（冻结）

| Tool | 方法 + 路径 | 请求体 | 成功 result |
| --- | --- | --- | --- |
| get_current_coordinates | GET `/api/v1/get_current_coordinates` | 无 | `我的坐标是{x},{y},{z}` |
| move_to_coordinates | POST `/api/v1/move_to_coordinates` | `{"x":"...","y":"...","z":"..."}`（字符串） | `成功到达{x},{y},{z}` |
| grab_the_block | POST `/api/v1/grab_the_block` | `{"color":"red"}`（red/green/yellow） | `有这种颜色的物块，且夹取物块成功` |
| release_the_block | POST `/api/v1/release_the_block` | 无 | `成功释放物块` |

响应 schema 一律：`{"code": int, "result": str|null, "error": str|null}`。
完整负例 result 字符串（未到达 / 没有这种颜色 / 夹取失败 / 释放失败等）见协议文档。

## 5. 输入字段（冻结）

- `x` / `y` / `z`：**字符串**，单位**米**。表内世界坐标（base 系，z 向上，桌面 z=0）。
- `color`：**字符串**，合法值 `red` / `green` / `yellow`。
- 缺失字段 / 字段类型错 / 非法 color → HTTP 400，`code=400`，`error` 为说明。

## 6. 时间表现（真实测量）

- 单次移动：默认 pacing=realtime，一次 ~1s 内到达（近距离）。
- 抓取/释放：含夹爪闭合/张开的物理结算，约 0.5–1s。
- 默认超时：`move_timeout_s=10.0`、`grasp_timeout_s=10.0`（configs/control.yaml）。
  到点收敛后**立即返回**，不会占用满 10s。

## 7. 物理真实性（诚实原则）

- 坐标来自真实末端位姿（MuJoCo `site_xpos[ee_site]`），**不是命令下发即算到达**。
- 未到达时误差 = ‖当前真实末端 − 目标‖₂（`未到达{x},{y},{z}，误差是{error}`）。
- 视觉默认 hsv 感知 + scripted 抓取；不伪造感知、不传送、不以 ground truth 冒充感知。

## 8. HTTP Error

| 情况 | HTTP | body |
| --- | --- | --- |
| 正常成功/业务失败 | 200 | `{"code":0,"result":...,"error":null}` |
| 输入校验失败 | 400 | `{"code":400,"result":null,"error":"<中文说明>"}` |
| 服务端未捕获异常 | 500 | `{"code":500,"result":null,"error":"内部异常: <类名>"}` |

- **业务失败 ≠ HTTP 错误**：不可达/没找到物块/夹取失败等都走 200 + `code=0` + 相应 result 字符串。
- 客户端判断「是否成功」请以 result 字符串语义为准（协议文档），不要只看 HTTP 200。
- 连接失败（连不上服务器）：非服务器错误，检查服务器是否在跑、端口、防火墙。

## 9. Blocking —— 特别强调

**四个机械臂 Tool 都是同步阻塞（synchronous blocking）API**：

- `move_to_coordinates` / `grab_the_block` / `release_the_block` 会**一直执行到动作收敛或超时**才返回；
- 期间会真实驱动 MuJoCo 物理推进（不是异步投递后立即返回）；
- 因此**调用方超时（HTTP timeout）必须开得足够大**。当前项目真实最大执行时间由
  configs/control.yaml 控制：`move_timeout_s=10.0`、`grasp_timeout_s=10.0`。
  最坏情况一次调用可阻塞约 10s（到超时才返回）。

**推荐调用方 timeout ≥ 15s**（覆盖 10s 超时 + 网络余量），并建议写为**可配置**：
coworker 侧把 `timeout` 做成可调参数，默认 ≥ 15s；如需更高保障，两端同步把
`move_timeout_s` / `grasp_timeout_s` 调大后再把客户端 timeout 相应放大。
**不要用默认几秒的 HTTP 超时去调这些同步 API**——否则移动尚未收敛就被客户端掐断。

## 10. 并发与线程安全

- MuJoCo `mjData` 非线程安全：所有物理推进 + 相机渲染在同一把锁下串行。
- 服务器内部已保证串行；**不要**在客户端并发调用会互相打架的移动/抓取（先到先得，物理状态单一份）。
- 单客户端按顺序调用即可；如需并发，自己加调用方队列。

## 11. Quick Smoke Test

服务器已启动后，在本机执行：

```bash
python scripts/smoke_test_arm_api.py --base-url http://127.0.0.1:8000
```

覆盖：坐标查询 / 移动成功 / 夹红块成功 / 释放成功 / 夹蓝块（无该颜色）失败 / 非法请求 400。
通过输出：`ALL ARM API SMOKE TESTS PASSED`。

## 12. Issue Report

报告问题请带齐以下信息，否则难以定位：

1. 请求：method + 完整 path + 请求体原文。
2. 响应：HTTP 状态 + 完整 body（含 `error`）。
3. 服务器启动日志（终端 stdout，见 `docs/ARM_SERVER_DEPLOYMENT.md` §8）。
4. 环境：backend_mode（mujoco/kinematic）、perception_mode、grasp_mode、pacing、OS、Python 版本。
5. 时间：出问题的具体时刻（前后相邻调用）。

已知边界（非 bug，按设计）：不可达目标如实报未到达；无该颜色物块如实报没有；
release 在未夹持时返回「本身就没加起来物块」；libpng 警告为 MuJoCo viewer 内置资源所致，与本模块无关。
