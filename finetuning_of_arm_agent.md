# Arm Agent 微调数据方案

> 目标：把 `Qwen/Qwen3-4B-Instruct-2507` 微调为异步双 agent 系统中的 **Arm Agent**（后台具身执行）。
> 本文只解决"造什么数据、怎么算合格"，供分工造数据使用。系统设计见 `async_dual_agent_system_design.md`，工具契约见 `api_of_embodied_tools.md`（工具返回字符串必须与该文档**逐字一致**）。训练配置建议见附录 A。

## 1. 前提

- 基座（微调前）：[`Qwen/Qwen3-4B-Instruct-2507`](https://www.modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507)。
- 部署：RTX 3090 #2，负责 Arm Agent 的微调与推理常驻（Voice Agent 在 3090 #1）。
- Arm Agent **不与人直接对话**。它的输入是：任务消息（`all_messages_from_voice_agent:...`）、工具结果（`role="tool"` 消息）、`<queue_status>` 状态栏；它的输出是：规划/说明文本 + 结构化 tool_calls。

## 2. 数据格式规范

- 模板：Qwen3 ChatML（`<|im_start|>` / `<|im_end|>`），交付为 JSONL（每行 `{"messages": [...], "tools": [...]}`，role ∈ `system` / `user` / `assistant` / `tool`）。
- 每个样本以 `role="system"` 的系统提示词开头；顶层 `tools` 字段携带本侧 6 个工具的 JSON schema，训练时由 chat template 渲染进 system 块的 `<tools>` 段。
- tool_call 用 **Qwen3 原生结构化协议**——assistant 消息的 `tool_calls` 字段（`arguments` 为 JSON 对象），`content` 只放伴随文本；chat template 渲染为：

```
<|im_start|>assistant
我先移动到物块位置。
<tool_call>
{"name": "move_to_coordinates", "arguments": {"x": "0.5", "y": "0.2", "z": "0.1"}}
</tool_call><|im_end|>
```

禁止自定义紧凑/括号协议（如 `[tool_call: ...]`、`move_to_coordinates:0.5,0.2,0.1`）。

- 工具结果以**单条 `role="tool"` 消息**写回；chat template 渲染时自动包进 user 块的 `<tool_response>` 段：

```
<|im_start|>user
<tool_response>
成功到达0.5,0.2,0.1
</tool_response><|im_end|>
```

**不再做 tool/user 双角色手工回写**（原生格式自带"tool 结果进 user 块"效果）。

- **`<queue_status>` 状态栏（最关键，pics/4.png 左下角修正后的机制）**：Arm Agent 忙碌时没有人类 user 消息可注入，因此在**每条 `role="tool"` 工具结果消息之后**，必须追加一条 role=user、内容为 `<queue_status>empty</queue_status>` 或 `<queue_status>not empty</queue_status>` 的消息，随后 assistant 继续推理。所有含工具调用的样本都必须带状态栏，位置错一律打回。状态栏在两个 agent 侧统一为独立 user 消息；**排序约定**：当 tool response、user input、状态栏三者同时存在时，顺序固定为 **tool response → user input → 状态栏**。
- **例外：空闲自动消费不带状态栏**。Arm Agent 空闲时由运行时自动消费队列，任务消息（`all_messages_from_voice_agent:...`）以一条 user 消息注入后**不追加状态栏**——队列刚被排空，状态栏必然为 empty，没有信息量。样本中任务消息后直接是 assistant 开始规划/调用工具，不要画蛇添足补状态栏。
- 消费到的消息以 `role="tool"` 消息进入上下文，content 为 `all_messages_from_voice_agent:消息1;消息2`（渲染后同样落在 user 的 `<tool_response>` 块内）。
- Arm 侧数据**不使用** `</interrupted>` 标记（打断只发生在面向人的 Voice 侧）。
- label mask：只对 assistant 段计算 loss。

### 完整示例（忙碌中消费新指令）

存储形态（JSONL，一行一样本；`tools` 字段略）：

```json
{"messages": [
  {"role": "system", "content": "你是异步双 agent 系统中的 Arm Agent，……"},
  {"role": "user", "content": "all_messages_from_voice_agent:用户给了以下信息：抓取 red 物块并放到 (1.0,2.0,3.0)，请你开始执行。"},
  {"role": "assistant", "content": "收到，我先确认当前坐标，再移动到物块位置抓取红色物块。", "tool_calls": [{"type": "function", "function": {"name": "get_current_coordinates", "arguments": {}}}]},
  {"role": "tool", "content": "当前坐标为(0.0,0.0,0.0)"},
  {"role": "user", "content": "<queue_status>empty</queue_status>"},
  {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "move_to_coordinates", "arguments": {"x": "0.5", "y": "0.2", "z": "0.1"}}}]},
  {"role": "tool", "content": "成功到达0.5,0.2,0.1"},
  {"role": "user", "content": "<queue_status>not empty</queue_status>"},
  {"role": "assistant", "content": "队列里有新消息，我先消费看看再决定下一步。", "tool_calls": [{"type": "function", "function": {"name": "get_message_from_voice_agent", "arguments": {}}}]},
  {"role": "tool", "content": "all_messages_from_voice_agent:用户改主意了，请改抓 yellow 物块;目标位置不变"},
  {"role": "user", "content": "<queue_status>empty</queue_status>"},
  {"role": "assistant", "content": "收到变更：放弃抓取红色，改抓黄色物块，目标位置不变。", "tool_calls": [{"type": "function", "function": {"name": "grab_the_block", "arguments": {"color": "yellow"}}}]},
  {"role": "tool", "content": "有这种颜色的物块，且夹取物块成功"},
  {"role": "user", "content": "<queue_status>empty</queue_status>"},
  {"role": "assistant", "content": "……（移动到目标位置 → release → send_to_voice_agent 上报完成）"}
], "tools": [...]}
```

经 Qwen3 chat template 渲染后（训练时实际喂给模型的形态）：

```
<|im_start|>system
……<tools>……（6 个工具的 JSON schema）……</tools>……<|im_end|>
<|im_start|>user
all_messages_from_voice_agent:用户给了以下信息：抓取 red 物块并放到 (1.0,2.0,3.0)，请你开始执行。<|im_end|>
<|im_start|>assistant
收到，我先确认当前坐标，再移动到物块位置抓取红色物块。
<tool_call>
{"name": "get_current_coordinates", "arguments": {}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
当前坐标为(0.0,0.0,0.0)
</tool_response><|im_end|>
<|im_start|>user
<queue_status>empty</queue_status><|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "move_to_coordinates", "arguments": {"x": "0.5", "y": "0.2", "z": "0.1"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
成功到达0.5,0.2,0.1
</tool_response><|im_end|>
<|im_start|>user
<queue_status>not empty</queue_status><|im_end|>
<|im_start|>assistant
队列里有新消息，我先消费看看再决定下一步。
<tool_call>
{"name": "get_message_from_voice_agent", "arguments": {}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
all_messages_from_voice_agent:用户改主意了，请改抓 yellow 物块;目标位置不变
</tool_response><|im_end|>
<|im_start|>user
<queue_status>empty</queue_status><|im_end|>
<|im_start|>assistant
收到变更：放弃抓取红色，改抓黄色物块，目标位置不变。
<tool_call>
{"name": "grab_the_block", "arguments": {"color": "yellow"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
有这种颜色的物块，且夹取物块成功
</tool_response><|im_end|>
<|im_start|>user
<queue_status>empty</queue_status><|im_end|>
<|im_start|>assistant
……（移动到目标位置 → release → send_to_voice_agent 上报完成）
```

## 3. 数据集分类与场景清单

### 3.1 纯对话 / 规划文本集（约 15%，900 条）

Arm Agent 的"纯对话"不是闲聊，而是不触发工具调用时的文本能力：

- 任务理解与步骤规划：收到 `all_messages_from_voice_agent` 后的自然语言计划（先确认坐标 → 移动 → 抓取 → …）。
- `send_to_voice_agent` 上报文本的质量：进度汇报、完成汇报、异常求助，要求人话、信息完整（做了什么、结果是什么、需要人做什么）。
- 含糊/矛盾指令处理：任务消息缺颜色或坐标时，不臆造参数，通过 `send_to_voice_agent` 反问，由 voice agent 回去问人（负样本重点）。
- 收到闲聊/无关消息时的处理：说明自己只执行具身任务，不展开闲聊。

### 3.2 单工具调用集（约 35%，2100 条）——每个工具的每个返回分支都要覆盖

| 工具 | 返回分支 | 条数 |
| --- | --- | --- |
| `get_current_coordinates` | 低速返回当前坐标 / 高速返回"我的坐标是{x},{y},{z}" | 100 / 100 |
| `move_to_coordinates` | `成功到达{x},{y},{z}` / `未到达{x},{y},{z}，误差是{error}` | 150 / 150 |
| `grab_the_block` | 颜色非法 / 颜色合法但物块不在场（均返回"没有这种颜色的物块，无法夹取"）/ 夹取成功 / 夹取失败 | 100 / 100 / 150 / 150 |
| `release_the_block` | `本身就没加起来物块` / `成功释放物块` / `释放物块失败，请用手直接拿出来物块` | 100 / 150 / 150 |
| `send_to_voice_agent` | 进度汇报 / 完成汇报 / 异常求助 | 各 100 |
| `get_message_from_voice_agent` | 非空消费（1 条/多条消息）/ **空队列负样本**（queue_status 为 empty 时不调用本工具） | 100 / 150 |

注意：失败分支后 assistant 要有合理后续动作（重试一次 / 换策略 / `send_to_voice_agent` 求助），不许"失败后无事发生"。工具返回字符串逐字照抄 API 文档（含"本身就没加起来物块"的用字）。

### 3.3 多工具链调集（约 30%，1800 条）

标准任务链：`get_current_coordinates → move_to_coordinates(物块位置) → grab_the_block(color) → move_to_coordinates(目标) → release_the_block → send_to_voice_agent(完成)`。

- **平静链路**：全程 `<queue_status>empty</queue_status>`，每个工具成功分支。600 条（3 种颜色 × 不同坐标组合）。
- **失败恢复链路**：链中某一步失败（move 未到达 / grab 失败 / release 失败）→ 重试或上报。每个失败点各 150 条。
- **中断变更链路**：链中第 k 步后 `<queue_status>not empty</queue_status>` → `get_message_from_voice_agent` 消费 → 决策（继续 / 变更 / 中止）。变更场景：改颜色（已抓取的要先 `release_the_block` 再抓新颜色）、改目标坐标、取消任务、追加新任务。各 150 条。
- 链中工具参数必须前后一致（坐标、颜色不自相矛盾）。

### 3.4 人–Voice–Arm 三者交互集（约 20%，1200 条）

以 Arm 视角构造（人的意图经 voice agent 转发，以 `all_messages_from_voice_agent:...` 进入上下文）：

- 人中途改颜色 / 改坐标 / 取消 / 追加任务 → Arm 调整执行链。
- 人问进度 → voice 转发 → Arm `send_to_voice_agent` 汇报当前进度。
- Arm 异常升级：`grab` 反复失败 / `release_the_block` 返回"释放物块失败，请用手直接拿出来物块" → `send_to_voice_agent` 请人介入 → 人回复"已取出"/"再试一次" → 继续。
- 冲突指令：执行中收到与当前动作矛盾的消息（如移动中又让抓块）→ 安全优先（完成/停止当前动作、必要时保持夹持），再上报说明。

## 4. 总量与配比

建议总量 **6000 条**（可按 ±20% 浮动）：纯对话 900 / 单工具 2100 / 链调 1800 / 三者交互 1200。另留 **5% 作 held-out 验证集**（不参与训练），按相同分布切分。

## 5. 验收标准（造完每批数据自检 + 交付前抽检 5%）

1. 格式合法率 100%：样本带 `system` 消息与 `tools` 字段、assistant 工具调用为结构化 `tool_calls`（JSON arguments）、工具结果为单条 `role="tool"` 消息、`<queue_status>` 出现在每条工具结果之后（且仅出现在工具结果之后——空闲注入的任务消息不带状态栏）。
2. 工具返回字符串与 `api_of_embodied_tools.md` **逐字一致**（含标点与用字）。
3. 分支覆盖率：§3.2 表格中每个分支条数达标。
4. 行为正确性：`not empty` 后 assistant 下一步应调用 `get_message_from_voice_agent`；`empty` 时不误调用；未收到任务消息时不凭空执行具身工具。
5. 链内一致性：颜色、坐标等参数在同一样本内前后一致。
6. 去重：与其他批次无近重复样本。

## 6. 分工建议（共 19 人，工号 001–019，具体分工由组长填）

| 模块 | 建议人数 | 目标条数 | 负责人（工号） |
| --- | --- | --- | --- |
| 3.1 纯对话/规划 | 3 | 900 | 待填 |
| 3.2 单工具（4 具身工具） | 4（每人 1 工具） | 1500 | 待填 |
| 3.2 单工具（2 通信工具） | 2 | 600 | 待填 |
| 3.3 链调 | 5 | 1800 | 待填 |
| 3.4 三者交互 | 4 | 1200 | 待填 |
| 校验与 held-out 切分 | 1 | — | 待填 |

## 附录 A. 训练配置建议（3090 24GB 可跑）

- 方式：QLoRA（4bit NF4 + LoRA r=16、alpha=32、dropout 0.05，作用于全部线性层）；框架 Unsloth 或 LLaMA-Factory。
- 超参：lr 1e-4（cosine）、warmup 3%、3 epochs、seq_len 4096（packing）、bf16、gradient checkpointing、per_device_bs 4 × grad_accum 4。
- 评估：held-out 上统计 tool_call 格式合法率、工具选择/分支行为正确率、queue_status 响应正确率；训练前后对比。
