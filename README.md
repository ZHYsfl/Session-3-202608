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

```bash
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

## 运行测试

```bash
# 全量回归（含 MuJoCo 真物理测试，较慢）
pytest

# 快速回归（排除 simulator 标记）
pytest -m "not simulator"

# 延迟基准（语音→结果 端到端，HTTP 全链路）
python scripts/benchmark_dual_agent.py
```

Phase 6 实测：全量 **154 passed**；快速回归 130 passed / 24 deselected。
端到端延迟（HTTP 全链路）：fast 节奏约 **0.05s/命令**；realtime 含真实物理 移动 0.97s / 抓取 0.98s / 释放 0.05s（合计约 2.1s）。

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

各阶段详细报告见 `docs/progress/phase1~6_report.md`。

## 目录

```
arm_skill_server/
  api/            FastAPI 网关(arm :8000, voice :8001)
  skills/         协议 6 工具 + 双 Agent 队列工具
  voice/          Voice Agent: SimulatedSTT / parse_intent / SimulatedTTS /
                  VoiceAgent / ArmAgent / 传输层(InProcess / HTTP)
  perception/     HSV 检测 / TinyDetector / camera2 交叉验证 / 相机模型
  robot/          SO-101 运动学 / 轨迹 / 50Hz 控制 / RobotBackend(Sim2Real)
  runtime/        单例动作互斥 + 取消令牌
  queue/          InMemoryQueue 两条 FIFO 队列
  tests/          unit/ + integration/（含 simulator 标记）
configs/          *.yaml 阈值配置（业务代码禁止 hardcode）
scripts/          3D 仿真 GUI / 最终 Demo / 延迟基准 / 训练 / 数据生成
docs/             协议、架构、设计、各阶段报告
```

## 文档

| 文档 | 内容 |
| --- | --- |
| `docs/api_of_embodied_tool.md` | 机械臂工具网关 REST 协议（Source of Truth，**冻结**） |
| `docs/api_of_voice_tools.md` | 语音网关 REST 协议（:8001） |
| `docs/async_dual_agent_system_design.md` | 双 Agent 全双工 + 实时中断设计 |
| `docs/architecture.md` | 分层架构与 Sim2Real 设计 |
| `docs/kinematics.md` | 运动学 / 轨迹数学 |
| `docs/perception.md` | 视觉感知层设计 |
| `docs/architecture_notes.md` | 中断/并发等架构笔记 |
| `docs/progress/phaseX_report.md` | 各阶段交付报告 |

## 协议冻结

`docs/api_of_embodied_tool.md` 与 `docs/api_of_voice_tools.md` 中的 HTTP 路径 / 方法 /
字段 / 返回字符串 / 业务语义为**冻结协议**，不得擅自修改。如需变更：先写测试证明缺陷 → 修复 → 全量回归。
