# 双臂机器人 SFT 训练数据构造工作文档

> 总样本数：193 | Voice Agent: 110 | Arm Agent: 83 | 全部 JSON 有效

---

## 一、全局约束规则

以下规则适用于所有文件夹的所有样本：

### 1.1 消息格式（Qwen3 原生工具调用协议）
- **JSONL 格式**：每行一个完整的 JSON 对象（不换行、不美化），顶层为 `{"messages": [...], "tools": [...]}`
- **system 消息**：每个样本以 `role="system"` 开头（该侧 agent 的系统提示词），以 `role="assistant"` 结尾
- **tools 字段**：顶层 `tools` 数组携带该侧全部工具的 JSON schema，训练时由 Qwen3 chat template 渲染进 system 块的 `<tools>` 段
- **Tool call 格式**：**Qwen3 原生结构化协议**——assistant 消息的 `tool_calls` 字段为 `[{"type": "function", "function": {"name": "...", "arguments": {...}}}]`（`arguments` 为 JSON 对象），`content` 只放伴随文本；chat template 渲染为 `<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>`。禁止使用 `[tool_call: ...]` 等自定义括号协议
- **工具结果**：以单条 `role="tool"` 消息写回（content 为返回串）；chat template 渲染时自动包进 user 块的 `<tool_response>` 段，**不再做 tool/user 双角色手工回写**
- **queue_status 注入**：`queue_status` 以独立的 `role="user"` 消息形式插入——Voice 侧只跟在每条**人类** user 消息之后；Arm 侧跟在每条 tool 消息之后（空闲注入的首条任务消息除外）

### 1.2 双向通信对称性
- **Voice Agent** 下发任务时，`send_to_arm_agent()` 的 content 中必须包含"通过 send_to_voice_agent() 将结果返回给 voice agent"
- **Arm Agent** 上报结果/求助时，assistant content 必须显式命名 `send_to_voice_agent()`，禁止使用"上报"、"汇报"等泛称
- 两方消费对方消息时，必须显式命名 `get_message_from_arm_agent()` / `get_message_from_voice_agent()`

### 1.3 工具返回值引用
- assistant content 中引用工具返回值时，使用中文书名号「」包裹，避免 JSON 转义
- 6 个 Arm 工具的 **13 种返回值**必须全部覆盖

### 1.4 难度配比
- **4:1** 常规（easy）: 困难（hard），不设 5 条上限
- 常规和困难使用**同一个 prompt**，由工具返回值和 queue_status 的差异自然区分难度

---

## 二、Arm Agent 文件夹构造规则

Arm Agent 使用 6 个工具函数，按训练维度分为 A/B/C/D 四大类。

### 2.1 A 类：单工具基础（A1-A6）

每个工具一个子文件夹，训练模型掌握单个工具的调用与返回处理。

| 文件夹 | 工具函数 | 核心训练目标 |
|--------|---------|------------|
| A1_get_current_coordinates | `get_current_coordinates()` | 定位：获取当前坐标 |
| A2_move_to_coordinates | `move_to_coordinates(x,y,z)` | 移动：到达目标位置 |
| A3_grab_the_block | `grab_the_block(color)` | 抓取：按颜色夹取物块 |
| A4_release_the_block | `release_the_block()` | 释放：放下夹持的物块 |
| A5_send_to_voice_agent | `send_to_voice_agent(content)` | 通信：主动上报消息 |
| A6_get_message_from_voice_agent | `get_message_from_voice_agent()` | 通信：接收指令 |

**构造规则（常规 4 条）：**
- 用户给出单一任务，Arm 调用一个工具
- 工具返回**成功类**结果（如"成功到达"、"夹取成功"、"发送成功"）
- `queue_status` 全程 `empty`
- assistant content 显式命名工具函数名，如 `get_current_coordinates()返回「当前坐标为(0.5,0.5,0.2)」`
- 消息数：A1-A3, A5 为 5 条；A4 为 5 条；A6 为 14 条

**构造规则（困难 1 条）：**
- 工具返回**异常/边界**结果（如"未到达"、"夹取失败"、"当前没有新消息"）
- A4/A5/A6 困难引入 `queue_status=not empty`，需额外消费队列
- A4 困难必须包含 C4 强制路径：`release_the_block()` 失败 → `send_to_voice_agent("释放物块失败，请人用手直接取出物块。")`

**A6 特殊说明：**
- 常规 8 条，困难 2 条
- 仅当任务消息主动要求查看（或状态栏为 `not empty`）时才调用 get；状态栏为 `not empty` 时 get **必须**返回真实消息（`all_messages_from_voice_agent:...`），禁止出现"状态栏 not empty 却返回 `当前没有新消息`"的矛盾样本
- `当前没有新消息` 只出现在"任务消息主动要求查看且队列确实为空"的样本中（常规 1 条）：调用 get 后返回空 → 继续执行当前任务

---

### 2.2 B 类：任务链编排（B1-B3）

训练模型按固定顺序编排多个工具调用完成复杂任务。

| 文件夹 | 工具链 | 核心训练目标 |
|--------|-------|------------|
| B1_标准抓放链 | move→grab→move→release→send | 标准四步抓放流程 |
| B2_带定位链 | get→move→grab→move→release→send | 先定位再执行的完整流程 |
| B3_多物块连续 | (move→grab→move→release→send)×N | 多个物块顺序搬运 |

**构造规则（常规 4 条）：**
- 全链工具返回**均为成功**
- `queue_status` 全程 `empty`
- B1：4 步链 + 上报 = 17 条消息
- B2：5 步链（含定位）+ 上报 = 20 条消息
- B3：2 轮完整抓放链（10 步）+ 中间上报 + 最终上报 = 32 条消息
- assistant 在每步调用工具时，先解释再工具调用
- 最终上报必须显式通过 `send_to_voice_agent()` 完成

**构造规则（困难 1 条）：**
- 链中**某一工具返回异常**（如 B1 困难：首次 grab 失败，重试后成功）
- B3 困难：引入 `queue_status=not empty`，收到优先级变更指令，需调整执行顺序
- 消息数更多，包含异常处理分支

---

### 2.3 C 类：返回驱动决策（C1-C6）

训练模型**根据工具返回值决定下一步行为**——这是 Arm Agent 最核心的决策能力。

| 文件夹 | 异常返回值 | 驱动行为 |
|--------|----------|---------|
| C1_move未到达 | `未到达x,y,z，误差是{error}` | 调整参数重试 → 成功 or 求助 |
| C2_grab失败 | `有这种颜色的物块，但夹取物块失败` | 重试一次 → 成功 or 求助 |
| C3_grab不在场 | `没有这种颜色的物块，无法夹取` | 上报 voice agent 求助 → 等待新颜色 |
| C4_release失败 | `释放物块失败，请用手直接拿出来物块` | **强制**上报 voice agent 要求人工取出 |
| C5_release没夹 | `本身就没加起来物块` | 先执行 grab → 再 release |
| C6_get高速返回 | `我的坐标是x,y,z` | 判断坐标精度，决定是否重查 |

**构造规则（常规 4 条）：**
- 工具首次调用返回异常 → assistant 解析返回值决定行动 → 调用正确工具解决
- C1：move 未到达 → 重试 move（调整目标位置）
- C2：grab 失败 → 重试 grab
- C3：grab 不在场 → send_to_voice_agent 求助
- C4：release 失败 → **强制** send_to_voice_agent("释放物块失败，请人用手直接取出物块。")
- C5：release 没夹 → 先 grab 再 release
- C6：get 高速返回 → 再查一次 get_current_coordinates
- `queue_status` 全程 `empty`，8-11 条消息

**构造规则（困难 1-2 条）：**
- **相同 prompt 但首次 queue_status=not empty** → 需先消费队列
- 消费队列后才能继续处理异常
- C1/C2/C3 困难各 2 条（含"求助后接续"闭环样本：求助→voice 回复→继续执行）
- C4 困难 1 条：释放失败后队列有消息，先上报再消费
- 消息数 11-25 条

**C4 强制路径（所有语言模型的必学规则）：**
```
release_the_block() → 「释放物块失败，请用手直接拿出来物块」
→ send_to_voice_agent("释放物块失败，请人用手直接取出物块。")
```

---

### 2.4 D 类：异步通信（D1-D3）

训练模型在执行任务过程中**主动上报进度**和**上报后接续执行**。

| 文件夹 | 上报模式 | 核心训练目标 |
|--------|---------|------------|
| D1_上报结果单条 | 执行一步 → 上报一条 | 进度透明化通信 |
| D2_上报结果多条 | 执行多步 → 上报多条 | 多阶段进度追踪 |
| D3_上报后接续 | 上报完成 → 检测队列 → 接新任务 | 闭环双向通信 |

**构造规则（常规 4 条）：**
- D1：每完成一个子任务（移动/抓取/释放）→ 上报一次，4 条样本
- D2：全链执行中上报 2 次（中间进度 + 最终结果），4 条样本
- D3：全链执行完成 → 上报 → 检测到 `queue_status=not empty` → 消费 voice 消息 → 执行新任务 → 上报，4 条样本
- 上报前 assistant content 必须显式命名 `send_to_voice_agent()`
- D3 消费队列时显式命名 `get_message_from_voice_agent()`

**构造规则（困难 1 条）：**
- D1：包含异常工具返回（释放失败），上报求助信息
- D2：包含重试逻辑（grab 连续失败后求助）
- D3：更复杂的接续链条（上报→消费→执行→再上报→再消费→再执行），38 条消息

---

## 三、Voice Agent 文件夹构造规则

Voice Agent 不直接调用工具，而是**通过 send_to_arm_agent() 下发任务**、**通过 get_message_from_arm_agent() 获取反馈**。

### 3.1 骨架1：单轮直下

**场景**：用户清晰给出完整任务（颜色+位置）→ Voice 直接下发，不做澄清。

**构造规则（常规 8 条）：**
- 原 4 条 + 新增 4 条 not_empty 扩展
- 原 4 条：`queue_status=empty` → 直接 send_to_arm_agent
- 新增 4 条：`queue_status=not empty` → 先 get_message_from_arm_agent 消费 → 再 send_to_arm_agent
- send_to_arm_agent 的 content 必须包含"完成后通过 send_to_voice_agent() 将执行结果返回给 voice agent"
- 6 条消息（not_empty 版本多 2 条）

**构造规则（困难 2 条）：**
- 原 1 条 + 新增 1 条 not_empty 扩展
- 原困难：用户指令不精确（"那个红色的"），Arm 返回异常 → Voice 需判断并告知用户
- 新增困难：not_empty 后消费到 Arm 异常消息 → 将异常反馈给用户并建议替代方案

---

### 3.2 骨架2：澄清多轮

**场景**：用户指令不完整 → Voice 主动追问 → 用户补充 → 再下发。

5 个子场景：

| 子文件夹 | 缺失信息 | 澄清流程 |
|---------|---------|---------|
| 缺颜色 | 只说位置没颜色 | 问颜色 → 用户补 → 下发 |
| 缺位置 | 只说颜色没位置 | 问位置 → 用户补 → 下发 |
| 两者缺 | 颜色位置都没有 | 问两者 → 用户补 → 下发 |
| 2a_澄清被打断 | 刚开始澄清被中断 | 问一半被打断 → 用户直接补全 → 下发 |
| 2e_澄清中消费队列 | 澄清时队列有消息 | 收到指令 → 先消费 → 再澄清/下发 |

**构造规则（常规 8 条每个子场景）：**
- 原 4 条 + 新增 4 条 not_empty 扩展
- 原 4 条：用户指令完整后下发，全程 empty
- 新增 4 条：用户首次发话时 `queue_status=not empty` → 先消费 Arm 消息 → 再澄清/下发
- 2a 使用 `</interrupted>` 标签标记打断点
- 消息数 9-12 条

**构造规则（困难 2 条每个子场景）：**
- 原 1 条 + 新增 1 条 not_empty 扩展
- 困难特征：消费到**多条 Arm 消息**（需多次 get），或消费到的消息与用户指令**冲突/叠加**
- 例如：用户要黄色但 Arm 报告黄色夹取失败 → 需向用户建议替代方案

---

### 3.3 骨架3：结果转述

**场景**：Arm 返回任务结果 → Voice 转述给用户，并可能需要下发新任务。

3 个子场景：

| 子文件夹 | 消费内容 | Voice 行为 |
|---------|---------|-----------|
| 3a_基础转述单条 | 一条 Arm 消息 | 转述 → 结束 |
| 3b_多条混合转述 | 多条 Arm 消息（混合成功+异常） | 逐一转述 → 总结 |
| 3c_转述后接续 | 一条 Arm 结果 | 转述 → 检测队列 → 可能下发新任务 |

**构造规则（常规 4 条）：**
- 3a：检测到 queue not empty → get → 收到一条 Arm 成功/进度消息 → 转述给用户
- 3b：get → 收到多条 Arm 消息（all_messages_from_arm_agent:msg1;msg2;...）→ 逐一解读转述
- 3c：get → 转述 → 再次检测到队列消息 → 再次 get → 收到新任务请求 → 下发

**构造规则（困难 1 条）：**
- 3b/3c：消息中包含**异常信息** → Voice 在转述时需特别提示用户
- 转述时引用具体工具函数名（`grab_the_block()`、`move_to_coordinates()` 等）

---

### 3.4 骨架4：推理中中断

**场景**：Voice 正在输出时被用户打断（`</interrupted>` 标签），需处理中断并恢复。

2 个子场景：

| 子文件夹 | 中断后行为 | 核心训练目标 |
|---------|----------|------------|
| 4a_中断后直接接续 | 消费队列 → 直接继续回答 | 简单中断恢复 |
| 4b_中断后需调用工具 | 消费队列 → 调用工具查询 | 复杂中断恢复 |

**构造规则（常规 8 条）：**
- 原 4 条 + 新增 4 条 not_empty 扩展
- 流程：用户任务 → Voice 下发 → 正在输出 → `</interrupted>` 用户打断 → `queue_status=not empty` → 消费 Arm 消息 → 回复
- 4a：消费后直接回答用户打断的问题
- 4b：消费后需再调用 send_to_arm_agent 查询信息才能回答

**构造规则（困难 2 条）：**
- 原 1 条 + 新增 1 条 not_empty 扩展
- 困难特征：中断后消费到**异常消息** → 需结合异常和用户打断意图做复杂决策
- 例如：用户打断说"如果红色不行换黄色" → 消费到 Arm 报告"红色夹取失败" → 完美匹配 → 下发黄色任务

---

## 四、难度区分规则（常规 vs 困难）

**核心原则**：常规和困难使用**语义相同的 prompt**，通过以下维度自然区分难度：

### 4.1 工具返回值维度
| 维度 | 常规（Easy） | 困难（Hard） |
|------|------------|------------|
| move_to_coordinates | `成功到达x,y,z` | `未到达x,y,z，误差是{error}` |
| grab_the_block | `且夹取物块成功` | `但夹取物块失败` 或 `没有这种颜色的物块` |
| release_the_block | `成功释放物块` | `释放物块失败` 或 `本身就没加起来物块` |
| get_current_coordinates | `当前坐标为(x,y,z)` | `我的坐标是x,y,z`（高速模式） |
| get_message_from_voice_agent | `all_messages_from_voice_agent:...` | `当前没有新消息` 或多条混合消息 |

### 4.2 queue_status 维度
| 维度 | 常规（Easy） | 困难（Hard） |
|------|------------|------------|
| 队列状态 | 大部分全程 empty | 关键节点出现 not_empty |
| 消费轮数 | 0-1 次 get | 2+ 次 get（需多轮消费） |
| 消费内容 | 单条常规消息 | 多条消息 or 异常消息叠加 |

### 4.3 执行复杂度维度
| 维度 | 常规（Easy） | 困难（Hard） |
|------|------------|------------|
| 工具调用次数 | 1 次或标准链 | 更多（含重试、额外查询） |
| 消息总数 | 少（5-20 条） | 多（14-38 条） |
| 决策分支 | 无分支（直线执行） | 有分支（异常恢复、优先级变更） |
| 信息重叠 | 工具返回与用户意图一致 | 工具返回与用户意图冲突 |

### 4.4 具体对比示例（A4 release_the_block）
- **常规**：`请释放当前夹持的物块` → release → `成功释放物块` → 结束（5 条消息）
- **困难**：`释放机械臂上夹的物块` → release → `释放物块失败` → `queue_status=not empty` → send_to_voice_agent 求助 → get_message_from_voice_agent 等待指令（11 条消息）

---

## 五、工具返回值覆盖清单

以下 13 种返回值已全部覆盖（截至 2026-08-10）：

| # | 工具函数 | 返回值字符串 | 出现次数 |
|---|---------|------------|---------|
| 1 | get_current_coordinates | `当前坐标为(x,y,z)` | 17 |
| 2 | get_current_coordinates | `我的坐标是x,y,z` | 4 |
| 3 | move_to_coordinates | `成功到达x,y,z` | 46 |
| 4 | move_to_coordinates | `未到达x,y,z，误差是{error}` | 8 |
| 5 | grab_the_block | `没有这种颜色的物块，无法夹取` | 10 |
| 6 | grab_the_block | `有这种颜色的物块，且夹取物块成功` | 47 |
| 7 | grab_the_block | `有这种颜色的物块，但夹取物块失败` | 11 |
| 8 | release_the_block | `本身就没加起来物块` | 1 |
| 9 | release_the_block | `成功释放物块` | 41 |
| 10 | release_the_block | `释放物块失败，请用手直接拿出来物块` | 9 |
| 11 | send_to_voice_agent | `发送成功` | 145 |
| 12 | get_message_from_voice_agent | `当前没有新消息` | 6 |
| 13 | get_message_from_voice_agent | `all_messages_from_voice_agent:...` | 81 |

---

## 六、消息队列模型

```
Voice Agent                          Arm Agent
┌─────────────┐                     ┌─────────────┐
│ message_from│  ←── send_to_arm ── │ get_message  │
│ _arm_agent  │                     │ _from_voice  │
│   _queue    │                     │  _agent()    │
├─────────────┤                     ├─────────────┤
│ send_to_arm │  ── send_to_voice → │ message_from │
│  _agent()   │                     │ _voice_agent │
│             │                     │   _queue     │
├─────────────┤                     ├─────────────┤
│ get_message │                     │ send_to_     │
│ _from_arm   │                     │ voice_agent()│
│  _agent()   │                     │             │
└─────────────┘                     └─────────────┘

queue_status 注入时机:
  - Voice: 每条【人类】user 消息后（tool 结果后【不】注入）
  - Arm:   每条 tool 结果消息后（空闲注入的首条任务消息除外）
  - send_to_* 不改变发送方自己的 queue_status
```

**关键规则：**
- `send_to_voice_agent()` 和 `send_to_arm_agent()` **不会**改变发送方自己的 `queue_status`
- `get_message_from_*()` 消费后会清空对应队列 → 下一次 `queue_status` 变为 `empty`
- `queue_status=not empty` 在对方发来消息但在当前 agent 消费之前的任意时刻出现

---

## 七、文件目录结构

```
e:\训练数据\
├── arm_agent\
│   ├── A_单工具基础\
│   │   ├── A1_get_current_coordinates\  {常规,困难}\
│   │   ├── A2_move_to_coordinates\      {常规,困难}\
│   │   ├── A3_grab_the_block\           {常规,困难}\
│   │   ├── A4_release_the_block\        {常规,困难}\
│   │   ├── A5_send_to_voice_agent\      {常规,困难}\
│   │   └── A6_get_message_from_voice_agent\ {常规,困难}\
│   ├── B_任务链编排\
│   │   ├── B1_标准抓放链\  {常规,困难}\
│   │   ├── B2_带定位链\    {常规,困难}\
│   │   └── B3_多物块连续\  {常规,困难}\
│   ├── C_返回驱动决策\
│   │   ├── C1_move未到达\   {常规,困难}\
│   │   ├── C2_grab失败\     {常规,困难}\
│   │   ├── C3_grab不在场\   {常规,困难}\
│   │   ├── C4_release失败\  {常规,困难}\
│   │   ├── C5_release没夹\  {常规,困难}\
│   │   └── C6_get高速返回\  {常规,困难}\
│   └── D_异步通信\
│       ├── D1_上报结果单条\  {常规,困难}\
│       ├── D2_上报结果多条\  {常规,困难}\
│       └── D3_上报后接续\    {常规,困难}\
├── voice_agent\
│   ├── 骨架1_单轮直下\       {常规,困难}\
│   ├── 骨架2_澄清多轮\
│   │   ├── 缺颜色\           {常规,困难}\
│   │   ├── 缺位置\           {常规,困难}\
│   │   ├── 两者缺\           {常规,困难}\
│   │   ├── 2a_澄清被打断\    {常规,困难}\
│   │   └── 2e_澄清中消费队列\{常规,困难}\
│   ├── 骨架3_结果转述\
│   │   ├── 3a_基础转述单条\  {常规,困难}\
│   │   ├── 3b_多条混合转述\  {常规,困难}\
│   │   └── 3c_转述后接续\    {常规,困难}\
│   └── 骨架4_推理中中断\
│       ├── 4a_中断后直接接续\{常规,困难}\
│       └── 4b_中断后需调用工具\{常规,困难}\
└── api_of_embodied_tools.md    # 6 个 REST API 工具定义
```

---

*文档生成日期：2026-08-10 | 数据集版本：193 样本*
