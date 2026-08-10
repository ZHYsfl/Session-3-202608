# SO-101 具身系统：语音指挥的机械臂双 Agent（纯仿真）

低延迟、全双工、语音指挥的具身双 Agent 系统。**纯仿真，不依赖任何外部 API / 密钥**。

```
人 ──语音──> Voice Agent ──voice→arm 队列──> Arm Agent ──REST 工具──> 机械臂(3D物理仿真)
人 <──语音── Voice Agent <──arm→voice 队列── Arm Agent <──结果────────┘
```

- **Voice Agent（语音侧）**：`SimulatedSTT`（键盘=语音）+ 确定性 NLU（中英混合）+ `SimulatedTTS`（终端播报）。只「听懂 → 下达 → 播报」，不控臂。
- **Arm Agent（执行侧）**：消费命令 → 调 6 个协议工具 → 汇报结果。执行中收到新指令实时中断（人优先）。
- **实时中断**：执行中语音「取消 / 改抓黄色」→ monitor 线程 `state.request_cancel()` 约 20ms 内停臂，如实上报「动作被新指令中断(返回: 原结果)」。
- **3D 物理仿真**：MuJoCo（重力/接触/摩擦由物理引擎驱动），50Hz 关节闭环控制，相机 camera1/camera2 渲染。
- **自训练感知 + 策略**：HSV 色块检测 ↔ 自训练 TinyDetector（微型 CNN）；抓取 `scripted`（IK）↔ `bc`（自训练行为克隆策略）。
- **camera2 第三视角交叉验证**：抓取/释放用第二机位视觉验证（三值：确认/反证/回退物理），不读物块 ground truth。

## 快速开始

本模块位于仓库根目录下的 **`mechanical_arm/`**。以下命令均在该目录内执行。

```bash
cd mechanical_arm
pip install -r requirements.txt

# 机械臂模块独立 3D Demo（arm-only, 不依赖 Voice Agent）：自动顺序演示四个工具
#   1 获取坐标 → 2 移动 → 3 抓红 → 4 获取坐标 → 5 夹着物块移动 → 6 释放
#   每一步的工具返回字符串打印在终端；同时启动 arm REST 网关 :8000
python scripts/run_arm_demo.py

# 无窗口冒烟测试
python scripts/run_arm_demo.py --headless --fast

# 最终 Demo：MuJoCo 3D 窗口 + arm 网关 :8000 + voice 网关 :8001 + 双 Agent
python scripts/run_final_demo.py
#   在语音> 提示符键入命令: 抓红色的物块 / 移到 0.2 -0.05 0.15 / 取消 / 你的坐标在哪 / 放下
#   键盘备选: 1=抓红 2=抓黄 3=抓白  R=释放  H=重置  关闭窗口=退出
```

headless 服务器入口（不依赖 GUI，绑定 0.0.0.0:8000）：

```bash
cd mechanical_arm
python scripts/run_api.py
# 健康检查: curl http://127.0.0.1:8000/health
# 四工具冒烟: python scripts/smoke_test_arm_api.py --base-url http://127.0.0.1:8000
```

## 运行测试

```bash
# 全量回归（含 MuJoCo 真物理测试，较慢；排除需要 GUI 的用例）
cd mechanical_arm
pytest -m "not gui"

# 快速回归（进一步排除 simulator 标记的 MuJoCo 真物理测试）
pytest -m "not simulator"
```

实测：全量回归 **156 passed**。

## 阶段状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| Phase 1 | SO-101 运动学仿真（FK/IK/轨迹/控制）+ REST 网关 :8000 + 队列/取消 | ✅ 交付 |
| Phase 2 | 视觉感知层（HSV 色块检测，相机标定投影） | ✅ 交付 |
| Phase 3 | MuJoCo 真 3D 物理仿真 + 渲染器相机 + GUI（`run_3d_sim.py`） | ✅ 交付 |
| Phase 4 | 自训练 TinyDetector（微型 CNN 物块检测） | ✅ 交付 |
| Phase 5 | 自训练 BC 抓取策略（行为克隆，`grasp_mode: scripted | bc`） | ✅ 交付 |
| Phase 6 | **双 Agent 集成 + 全双工 + 实时中断 + camera2 交叉验证 + 最终 Demo** | ✅ 交付 |
| Phase 7 | Isaac 仿真 / 真机后端（Sim2Real 接口预留 stub） | ⏳ 后续 |

> 边界：双 Agent / Voice Agent / STT / TTS 由**其他成员**负责，不属机械臂模块验收范围。
> 机械臂模块（3D 仿真、控制、视觉/抓取策略、四个 Arm Tool、REST 暴露）独立可演示，
> 入口 `scripts/run_arm_demo.py`，不依赖 Voice Agent。

## 目录

```
mechanical_arm/
  arm_skill_server/
    api/            FastAPI 网关(arm :8000, voice :8001)
    skills/         协议工具(四个机械臂工具 + 队列工具)
    voice/          Voice Agent(双 Agent 演示用)
    perception/     HSV 检测 / TinyDetector / camera2 交叉验证 / 相机模型
    robot/          SO-101 运动学 / 轨迹 / 50Hz 控制 / RobotBackend(Sim2Real)
    runtime/        单例动作互斥 + 取消令牌
    queue/          InMemoryQueue 两条 FIFO 队列
    tests/          unit/ + integration/（含 simulator 标记）
  configs/          *.yaml 阈值配置（业务代码禁止 hardcode）
  models/           最终 checkpoint（grasp_policy.pt / tiny_detector.pt）
  scripts/          3D 仿真 GUI / 服务器入口 / 冒烟 / HTTP 回归
  docs/             协议 / 部署 / 交接文档
```

## 文档

| 文档 | 内容 |
| --- | --- |
| `docs/api_of_embodied_tool.md` | 机械臂工具网关 REST 协议（Source of Truth，**冻结**） |
| `docs/api.md` | 协议实现记录（格式化约定 / 错误分支 / 示例） |
| `docs/ARM_SERVER_DEPLOYMENT.md` | 服务器部署（装 / 起 / 验 / headless / 日志 / 停止） |
| `docs/ARM_MODULE_HANDOFF.md` | 交接文档（Base URL / Blocking 语义 / 冒烟 / 报障） |
| `docs/SERVER_UPLOAD_CHECKLIST.md` | 上传核对清单 |

## 协议冻结

`docs/api_of_embodied_tool.md` 中的 HTTP 路径 / 方法 / 字段 / 返回字符串 / 业务语义为
**冻结协议**，不得擅自修改。如需变更：先写测试证明缺陷 → 修复 → 全量回归。
