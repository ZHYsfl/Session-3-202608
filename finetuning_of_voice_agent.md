# Voice Agent 微调数据方案

> 目标：把 `Qwen/Qwen3-4B-Instruct-2507` 微调为异步双 agent 系统中的 **Voice Agent**（前台全双工语音交互）。
> 本文只解决"造什么数据、怎么算合格"，供分工造数据使用。系统设计见 `async_dual_agent_system_design.md`，工具契约见 `api_of_voice_tools.md`（工具返回字符串必须与该文档**逐字一致**）。训练配置建议见附录 A。

## 1. 前提

- 基座（微调前）：[`Qwen/Qwen3-4B-Instruct-2507`](https://www.modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507)。
- 部署：RTX 3090 #1，负责 Voice Agent 的微调与推理常驻（Arm Agent 在 3090 #2）。
- Voice Agent 是人唯一的交互入口，工具只有两个：`send_to_arm_agent`（下发/变更任务）和 `get_message_from_arm_agent`（消费 arm 上报并语音转述）。**与 PPT 系统不同，没有 remember / require_confirm 需求收集与前端确认管线**：意图不明确时通过对话问清，意图明确即下发。
- 语音链路（浏览器前端 VAD 含回声消除、后端 STT、前端播放器、打断）由前端与编排层实现，但**打断后的上下文重组是模型行为**，必须靠数据训练（见 §3.5）。

## 2. 数据格式规范

- 模板：Qwen3 ChatML（`<|im_start|>` / `<|im_end|>`），交付为 JSONL（每行 `{"messages": [...], "tools": [...]}`，role ∈ `system` / `user` / `assistant` / `tool`）。
- 每个样本以 `role="system"` 的系统提示词开头；顶层 `tools` 字段携带本侧 2 个工具的 JSON schema，训练时由 chat template 渲染进 system 块的 `<tools>` 段。
- tool_call 用 **Qwen3 原生结构化协议**：assistant 消息的 `tool_calls` 字段为 `[{"type": "function", "function": {"name": "send_to_arm_agent", "arguments": {"content": "..."}}}]`，`content` 只放伴随文本；chat template 渲染为 `<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>`。禁止自定义紧凑/括号协议。
- 工具结果以**单条 `role="tool"` 消息**写回；chat template 渲染时自动包进 user 块的 `<tool_response>` 段（即原生格式自带"tool 结果进 user 块"效果），**不再做 tool/user 双角色手工回写**。
- **`<queue_status>` 状态栏**：**每一条人类 user 消息之后**紧跟一条独立的 role=user 状态栏消息，内容仅为 `<queue_status>empty</queue_status>` 或 `<queue_status>not empty</queue_status>`（反映 `message_from_arm_agent_queue` 是否非空）。所有含人类发言的样本都必须带。**排序约定**：当 tool response、user input、状态栏三者同时存在时，顺序固定为 **tool response → user input → 状态栏**，排错一律打回。
- 消费到的消息以 `role="tool"` 消息进入上下文，content 为 `all_messages_from_arm_agent:消息1;消息2`（渲染后同样落在 user 的 `<tool_response>` 块内）。
- **打断标记**：被打断的 assistant 消息以 `</interrupted>` 截断并原样保留；下一条用户消息以 `</interrupted>` 开头（后接用户新文本），其后再跟独立的 `<queue_status>` 状态栏消息。
- label mask：只对 assistant 段计算 loss。

### 完整示例（下发 → 问进度 → 消费转述）

存储形态（JSONL，一行一样本；`tools` 字段略）：

```json
{"messages": [
  {"role": "system", "content": "你是异步双 agent 系统中的 Voice Agent，……"},
  {"role": "user", "content": "帮我把红色的物块放到 1.0, 2.0, 3.0 那里。"},
  {"role": "user", "content": "<queue_status>empty</queue_status>"},
  {"role": "assistant", "content": "好的，这就安排机械臂抓取红色物块，放到坐标 (1.0, 2.0, 3.0)。", "tool_calls": [{"type": "function", "function": {"name": "send_to_arm_agent", "arguments": {"content": "抓取 red 物块并放到 (1.0,2.0,3.0)，完成后通过 send_to_voice_agent() 将结果返回给 voice agent。"}}}]},
  {"role": "tool", "content": "发送成功"},
  {"role": "user", "content": "进行得怎么样了？"},
  {"role": "user", "content": "<queue_status>not empty</queue_status>"},
  {"role": "assistant", "content": "我看一下后台的消息。", "tool_calls": [{"type": "function", "function": {"name": "get_message_from_arm_agent", "arguments": {}}}]},
  {"role": "tool", "content": "all_messages_from_arm_agent:已到达目标位置 (1.0,2.0,3.0) 并成功释放物块，任务完成"},
  {"role": "assistant", "content": "任务已经完成了，红色物块已经放到指定位置。"}
], "tools": [...]}
```

经 Qwen3 chat template 渲染后（训练时实际喂给模型的形态）：

```
<|im_start|>system
……<tools>……（2 个工具的 JSON schema）……</tools>……<|im_end|>
<|im_start|>user
帮我把红色的物块放到 1.0, 2.0, 3.0 那里。<|im_end|>
<|im_start|>user
<queue_status>empty</queue_status><|im_end|>
<|im_start|>assistant
好的，这就安排机械臂抓取红色物块，放到坐标 (1.0, 2.0, 3.0)。
<tool_call>
{"name": "send_to_arm_agent", "arguments": {"content": "抓取 red 物块并放到 (1.0,2.0,3.0)，完成后通过 send_to_voice_agent() 将结果返回给 voice agent。"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
发送成功
</tool_response><|im_end|>
<|im_start|>user
进行得怎么样了？<|im_end|>
<|im_start|>user
<queue_status>not empty</queue_status><|im_end|>
<|im_start|>assistant
我看一下后台的消息。
<tool_call>
{"name": "get_message_from_arm_agent", "arguments": {}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
all_messages_from_arm_agent:已到达目标位置 (1.0,2.0,3.0) 并成功释放物块，任务完成
</tool_response><|im_end|>
<|im_start|>assistant
任务已经完成了，红色物块已经放到指定位置。<|im_end|>
```

## 3. 数据集分类与场景清单

### 3.1 纯对话集（约 25%，1500 条）

- 意图澄清对话：人只给部分信息（"帮我抓个物块"），agent 通过对话问清颜色、目标位置等缺项——**只靠对话，不依赖任何工具**。
- 闲聊与兜底：打招呼、问能力范围、感谢；与具身任务无关的请求礼貌拒绝并引导回任务。
- 进度安抚：人催进度但队列里没有新消息（状态栏 empty）时，如实说"还没有新进展，有消息我第一时间告诉你"。
- 打断后续接：assistant 发言被打断（`</interrupted>`）后，能顺着用户新输入继续，不重复已说内容、不丢上下文。

### 3.2 单工具调用集（约 20%，1200 条）

| 工具 | 场景 | 条数 |
| --- | --- | --- |
| `send_to_arm_agent` | **正样本**：意图明确即下发，`content` 完整转述任务（动作 + 颜色 + 坐标） | 300 |
| `send_to_arm_agent` | **负样本**：意图不明 / 缺关键信息 / 人在闲聊或问进度时不调用，继续对话澄清或如实应答 | 300 |
| `get_message_from_arm_agent` | `<queue_status>not empty</queue_status>` 时调用并语音转述（进度 / 完成 / 求助） | 300 |
| `get_message_from_arm_agent` | 多条消息拼接时的完整转述 | 100 |
| `get_message_from_arm_agent` | **empty 负样本**：状态栏为 empty 时不调用 | 200 |

### 3.3 多工具链调集（约 25%，1500 条）

基本链：`send_to_arm_agent（下发）→ 后续某条用户消息状态栏 not empty → get_message_from_arm_agent → 语音转述`。

- 标准链：下发 → 完成汇报 → 转述。500 条。
- 变更链：人中途改颜色 / 改坐标 / 取消 → `send_to_arm_agent` 变更消息 → 消费到 arm 的调整结果 → 转述。400 条。
- 进度跟踪链：执行中人问进度 → 消费 → 转述；或人没问但状态栏 not empty → 消费后主动播报重要事件（完成 / 失败）。300 条。
- 异常链：消费到"夹取物块失败"/"释放物块失败，请用手直接拿出来物块" → 如实转述并询问人下一步（重试 / 人工处理 / 换颜色）。300 条。

### 3.4 人–Voice–Arm 三者交互集（约 20%，1200 条）

- 人中途改主意（任务已下发）：换颜色 / 换位置 / 取消 / 追加任务。400 条。
- arm 求助升级：释放失败请人用手取出 → voice 转述 → 人答复"已取出" → `send_to_arm_agent` 通知继续。300 条。
- 多任务排队：第一个任务执行中，人下发第二个任务 → voice send（arm 空闲后自动消费）→ 两个结果先后转述，不混淆。300 条。
- 矛盾请求：人前后说法矛盾（先说红色又说"我刚才是不是说的黄色"）→ 先跟人确认清楚再 send，不把模糊指令丢给 arm。200 条。

### 3.5 打断专项集（约 10%，600 条，voice 侧独有能力，重点）

- **说话阶段被打断**：assistant 语音回复说到一半被截断（`</interrupted>`），用户新输入接续；模型继续对话且不重复已说内容。300 条。
- **tool_call 阶段被打断（边听边想）**：assistant 正在生成 tool_call（对用户静默）时用户插话；tool response 与用户转写双双就绪后，assistant 同时看到工具结果和新输入，一次性做下一步推理（如：正在 `send_to_arm_agent` 下发任务时用户插话"等等，位置改成 4,5,6" → 下一步把更正后的任务重新 `send_to_arm_agent`；或正在 `get_message_from_arm_agent` 时用户插话 → 同一条回复里既转述结果又应答新输入）。300 条。

## 4. 总量与配比

建议总量 **6000 条**（可按 ±20% 浮动）：纯对话 1500 / 单工具 1200 / 链调 1500 / 三者交互 1200 / 打断专项 600。另留 **5% 作 held-out 验证集**（不参与训练），按相同分布切分。

## 5. 验收标准（造完每批数据自检 + 交付前抽检 5%）

1. 格式合法率 100%：样本带 `system` 消息与 `tools` 字段、assistant 工具调用为结构化 `tool_calls`（JSON arguments）、工具结果为单条 `role="tool"` 消息、每条人类 user 消息之后紧跟独立的 `<queue_status>` 状态栏消息（三者同时存在时顺序为 tool response → user input → 状态栏）、打断样本带 `</interrupted>`。
2. 工具返回字符串与 `api_of_voice_tools.md` **逐字一致**。
3. 分支覆盖率：§3.2 表格中每个场景条数达标。
4. 行为正确性：状态栏 `not empty` 后应调用 `get_message_from_arm_agent`；`empty` 不误调用；**意图不明确时不得调用 `send_to_arm_agent`**（红线）。
5. 语音风格：assistant 文本口语化、简洁，适合语音播报；转述 arm 消息时信息完整（结果 + 需要人做什么）。
6. 去重：与其他批次无近重复样本。

## 6. 分工建议（共 19 人，工号 001–019，具体分工由组长填）

| 模块 | 建议人数 | 目标条数 | 负责人（工号） |
| --- | --- | --- | --- |
| 3.1 纯对话 | 3 | 1500 | 待填 |
| 3.2 单工具（send / get + 负样本） | 4 | 1200 | 待填 |
| 3.3 链调 | 5 | 1500 | 待填 |
| 3.4 三者交互 | 4 | 1200 | 待填 |
| 3.5 打断专项 | 2 | 600 | 待填 |
| 校验与 held-out 切分 | 1 | — | 待填 |

## 附录 A. 训练配置建议（3090 24GB 可跑）

- 方式：QLoRA（4bit NF4 + LoRA r=16、alpha=32、dropout 0.05，作用于全部线性层）；框架 Unsloth 或 LLaMA-Factory。
- 超参：lr 1e-4（cosine）、warmup 3%、3 epochs、seq_len 4096（packing）、bf16、gradient checkpointing、per_device_bs 4 × grad_accum 4。
- 评估：held-out 上统计 tool_call 格式合法率、`send_to_arm_agent` 红线违反率（应为 0）、queue_status 响应正确率、打断续接质量人工评分；训练前后对比。
