# 异步双 Agent 系统设计原理

> 本文复述并固化「语音交互 + 具身执行」异步双 agent 系统的设计。
> 系统沿用 EducationAgent（全双工语音 PPT 生成助手，`github.com/ZHYsfl/EducationAgent`）验证过的异步架构：前台 Voice Agent 与人实时语音交互，后台 PPT Agent 替换为操作机械臂真机的 Arm Agent。
> 与 PPT 系统的主要差异：Voice Agent **没有** remember / require_confirm 那套需求收集与前端确认管线——人直接语音下达任务，Voice Agent 在对话中澄清意图，意图明确即转发给 Arm Agent（见 §5）。
> 原 PPT（`/home/zane/ppt`，pics/1–7.png）中除 **pics/4.png 左下角一句话** 外全部照搬，该处 bug 的修正见 §4.3。

## 1. 系统总览

人对机械臂说话即可完成任务下发与变更：Voice Agent 与人实时语音交互、理解意图、下发任务并转述结果；Arm Agent 在后台调用具身工具链执行物理任务；两个 agent 通过两条消息队列异步解耦，互不阻塞。

```
                ┌───── 浏览器前端：VAD（含回声消除、防吞字缓存）+ 播放器 ─────┐
  人 ⇄ 麦克风 → 前端 VAD → STT → ┌────────────────┐ → 前端播放器 ⇄ 人
                            │  Voice Agent   │
                            │ (Qwen3-4B 微调) │
                            └───┬────────▲───┘
        send_to_arm_agent ──────┘        └────── get_message_from_arm_agent
                │                            ▲
                ▼                            │
   message_from_voice_agent_queue   message_from_arm_agent_queue
        （voice → arm 方向）          （arm → voice 方向）
                │                            ▲
 get_message_from_voice_agent ─────┐         └────── send_to_voice_agent
                            └───┬──┴────────┐
                            │   Arm Agent    │
                            │ (Qwen3-4B 微调) │
                            └───┬───────────┘
                                │ RESTful 调用具身工具网关（127.0.0.1:8000）
                                ▼
   get_current_coordinates / move_to_coordinates / grab_the_block / release_the_block
                                │
                                ▼
                          机械臂真机 + 视觉摄像头
```

## 2. 角色划分

| 角色 | 基座（微调前） | 部署 | 职责 | 工具 |
| --- | --- | --- | --- | --- |
| Voice Agent | `Qwen/Qwen3-4B-Instruct-2507` | RTX 3090 #1（微调+推理） | 与人语音交互；理解任务意图（不清晰时对话澄清）；向 Arm Agent 下发/变更任务；把 Arm Agent 的进度/结果语音转述给人 | `send_to_arm_agent`、`get_message_from_arm_agent` |
| Arm Agent | `Qwen/Qwen3-4B-Instruct-2507` | RTX 3090 #2（微调+推理） | 后台执行具身任务链（移动、抓取、释放）；上报进度与异常；中途消费新指令并调整执行 | `get_current_coordinates`、`move_to_coordinates`、`grab_the_block`、`release_the_block`、`send_to_voice_agent`、`get_message_from_voice_agent` |

- 人永远只跟 Voice Agent 说话。Arm Agent 不直接对人输出，它对外界的一切反馈都通过 `send_to_voice_agent` 生产消息，由 Voice Agent 消费后用语音转述给人。
- 两个 agent 基座模型相同，但微调数据各自独立（见 `finetuning_of_voice_agent.md` / `finetuning_of_arm_agent.md`）。

## 3. 语音链路（前端 VAD/播放器 + 后端 STT + 打断 + 边听边想）

串行基线 `VAD → STT → LLM → TTS` 的问题：延迟叠加（一轮交互延迟 ≈ 四级之和）、无法插话、资源闲置。本系统的划分：**VAD 和 TTS 模型都不在后端**——VAD 用浏览器前端算法（自带回声消除、开口前缓存防吞字），播报用前端播放器。这样还顺带消除了打断时 TTS 与 LLM 之间的延迟时差问题。保留两个模型侧机制：

1. **打断机制**：assistant 回复播报期间人插话（前端 VAD 检出、前端停播并通知后端），已生成部分以 `</interrupted>` 标记截断点、原样保留在上下文中，用户新输入自然接续其后，上下文保持连贯。
2. **边听边想（核心观察）**：`tool_call` 生成阶段对用户天然静默，因此用户打断可与当前工具调用**并行处理**（goroutine）。待 tool response 与用户语音转写**双双就绪**后，再让 LLM 做下一步推理。

## 4. 异步双 Agent 通信（核心）

### 4.1 两条 FIFO 队列

| 队列 | 方向 | 生产者工具 | 消费者工具 |
| --- | --- | --- | --- |
| `message_from_voice_agent_queue` | voice → arm | `send_to_arm_agent`（Voice 侧） | `get_message_from_voice_agent`（Arm 侧） |
| `message_from_arm_agent_queue` | arm → voice | `send_to_voice_agent`（Arm 侧） | `get_message_from_arm_agent`（Voice 侧） |

- `send_*` 把一条消息追加到对方方向的队尾，返回确认字符串。
- `get_*` **排空**自己方向的队列，把全部消息拼接为一条返回：`all_messages_from_xxx_agent:消息1;消息2;...`；队列为空时返回 `当前没有新消息`。
- 队列是系统级共享状态（编排运行时持有），两个 agent 的工具网关都对接同一队列后端。

### 4.2 人优先原则

人的语音交互永远不被后台任务阻塞；后台 agent 的消息只能通过状态栏感知 + 主动消费的方式进入上下文，**绝不直接插队打断**当前推理。

### 4.3 `queue_status` 状态栏机制（统一为独立 user 消息 + 固定排序）

状态栏在两个 agent 侧**统一**以一条独立的 role=user 消息进入上下文，内容仅为 `<queue_status>empty/not empty</queue_status>`。**排序约定**：当 tool response、user input、状态栏三者同时存在时，固定顺序为 **tool response → user input → 状态栏**。

- **Voice Agent 侧**：每一条人类 user 消息之后紧跟一条状态栏消息，模型随时感知 arm 方向是否有未读消息；为 `not empty` 时可调用 `get_message_from_arm_agent` 主动消费。上一轮的工具结果（`role="tool"` 消息，渲染进 user 的 `<tool_response>` 块）本就写在该条人类消息之前，三者顺序天然满足排序约定。
- **Arm Agent 侧**（原 pics/4.png 左下角写"作为状态栏给每条 user 信息加上 `<queue_status>`"，**有误**——Arm Agent 不与人直接对话，忙碌时根本没有新的 user 消息可供注入）。正确机制：
  - **空闲（空下来）时**：自动消费 `message_from_voice_agent_queue`（由运行时触发一次 `get_message_from_voice_agent`），新任务以一条 user 消息（`all_messages_from_voice_agent:...`）进入上下文。**此时不追加状态栏**——队列刚被排空，状态栏必然为 empty，没有信息量；模型看到任务消息直接开始执行。
  - **忙碌（工具调用循环中）时**：在**每条 tool response 消息之后**，追加一条 **role=user、内容为 `<queue_status>empty/not empty</queue_status>`** 的状态栏消息，然后让 LLM **继续推理**；若队列不空，Arm Agent 可以调用 `get_message_from_voice_agent` **主动消费**，从而在任务执行中途感知新指令（改颜色 / 改位置 / 取消）。

### 4.4 消费结果的消息形式

`get_*` 排空队列后，全部消息以一条 `role="tool"` 消息进入上下文（content 以 `all_messages_from_*:` 前缀开头），Qwen3 chat template 渲染时自动包进 user 块的 `<tool_response>` 段：

```
<|im_start|>user
<tool_response>
all_messages_from_voice_agent:请改抓黄色物块;目标位置不变
</tool_response><|im_end|>
```

保证多轮之后队列状态与上下文始终自洽。

## 5. 上下文工程

- **与 PPT 系统的差异**：本系统没有 `remember` / `require_confirm` 需求收集与前端确认管线。人直接语音下达任务；Voice Agent 只在对话层面澄清意图（缺颜色、缺位置就问清楚），意图明确即调用 `send_to_arm_agent` 下发。微调数据要保证：**意图不明确时不得调用 `send_to_arm_agent`**。
- **Qwen3 原生工具协议**：工具声明走 `tools` 字段（渲染进 system 块 `<tools>` 段）；assistant 的工具调用走结构化 `tool_calls` 字段（渲染为 `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`）；工具结果以单条 `role="tool"` 消息写回，chat template 自动包成 user 块的 `<tool_response>` 段——原生格式自带"tool 结果进 user 块"效果，**不再做 tool/user 双角色手工回写**。
- **`</interrupted>` 打断重组**：被截断的 assistant 消息原样保留，用户新输入（可带 `</interrupted>` 标记）接续其后，随后紧跟一条独立的 `<queue_status>` 状态栏 user 消息。
- **滚动压缩（长对话防溢出，两侧同思路）**：历史条数超过阈值时，把最旧的一段交给同一个 LLM 压缩成滚动摘要（可与前次摘要合并），近期消息原样保留——Voice 侧阈值 24 条 / 保留最近 12 条，摘要存于 `AppState` 并在每次推理时注入 system 之后；Arm 侧阈值 48 条 / 保留最近 24 条，摘要以 `【此前执行摘要】` 开头的 user 消息存于历史内 system 之后。压缩调用本身不带工具 schema。相比滑动窗口直接丢弃旧消息，压缩机制不丢信息，且让训练（短对话）与推理（长对话）的上下文分布保持一致。

## 6. 一次完整链调时序（示例）

1. 人：「帮我把红色的物块抓到 1.0, 2.0, 3.0 那里。」
2. Voice Agent：意图明确（若不明确，先对话问清颜色/位置）→ `send_to_arm_agent` 下发任务。
3. Arm Agent 空闲自动消费：拿到任务 → `get_current_coordinates` → `move_to_coordinates`（物块处）→ `grab_the_block("red")` → `move_to_coordinates(1.0,2.0,3.0)` → `release_the_block`；**每条 tool response 后注入 `<queue_status>` 状态栏**。
4. 中途人改主意：「换成黄色的。」→ Voice Agent `send_to_arm_agent` 变更消息 → Arm Agent 在某条 tool response 后的状态栏看到 `not empty` → `get_message_from_voice_agent` 主动消费 → 调整执行（先把红块 `release_the_block`，再抓黄块）→ `send_to_voice_agent` 上报完成。
5. Voice Agent 在下一条人类消息的状态栏看到 `not empty` → `get_message_from_arm_agent` 消费 → 语音向人汇报结果。

## 7. 基座模型与算力规划

- 基座（微调前）：两个 agent 均为 [`Qwen/Qwen3-4B-Instruct-2507`](https://www.modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507)，各自独立微调。
- 算力：租用 2 × RTX 3090（24GB），一卡负责一个 agent 的微调与推理常驻；队列、工具网关、STT 等编排组件跑在 CPU 侧；VAD（含回声消除）与播放器在浏览器前端。

## 8. 文档索引

| 文档 | 内容 |
| --- | --- |
| `api_of_embodied_tools.md` | Arm Agent 六个工具的 RESTful 网关文档（4 具身 + 2 通信），端口 8000 |
| `api_of_voice_tools.md` | Voice Agent 两个工具（生产/消费）的 RESTful 网关文档，端口 8001 |
| `finetuning_of_arm_agent.md` | Arm Agent 微调数据方案（造什么数据、验收标准、分工） |
| `finetuning_of_voice_agent.md` | Voice Agent 微调数据方案（同上） |
