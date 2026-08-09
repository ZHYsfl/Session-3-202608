我们要写 voice agent 的两个工具（生产/消费，用于和 arm agent 异步通信），为了联调方便，把这两个工具暴露成 restful api。restful api 的文档要包含接口是什么，输入和输出的字段各自是什么的详细说明，以及该接口的说明介绍等。

一个工具是 send_to_arm_agent(content: str) -> str，把一条消息追加到 message_from_voice_agent_queue 队尾（生产），供 arm agent 空闲时自动消费、忙碌时主动消费。返回 "发送成功"。注意：人的任务意图不明确（缺颜色、缺位置等关键信息）时不得调用，先通过对话跟人确认清楚再下发。

一个工具是 get_message_from_arm_agent() -> str，排空 message_from_arm_agent_queue（消费），把 arm agent 上报的进度/结果/求助等全部消息一次性取出，返回 "all_messages_from_arm_agent:消息1;消息2;..."；队列为空时返回 "当前没有新消息"。

---

# RESTful API 网关文档

## 1. 概述

为了方便联调，将上述 voice agent 的两个工具通过本地 RESTful API 网关对外暴露。

- 服务地址（Base URL）：`http://127.0.0.1:8001`
- 协议：HTTP + JSON（`Content-Type: application/json; charset=utf-8`）
- 所有接口均为同步阻塞调用：工具内部执行完毕后才返回响应，联调方无需轮询。
- 与具身工具网关（`http://127.0.0.1:8000`，见 `api_of_embodied_tools.md`）对接**同一个队列后端**（见 §3 队列与状态栏约定）。系统整体设计见 `async_dual_agent_system_design.md`。

### 通用响应结构

所有接口统一返回如下 JSON 结构：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | int | 网关状态码。`0` 表示网关调用成功（不代表工具内部业务一定成功，业务结果见 `result` 字段）；非 0 表示网关层失败（参数错误、内部异常等）。 |
| `result` | string \| null | 工具的原始返回字符串，原样透传。网关调用失败时为 `null`。 |
| `error` | string \| null | 网关层错误描述。成功时为 `null`。 |

HTTP 状态码约定：

| HTTP 状态码 | 含义 |
| --- | --- |
| 200 | 网关调用成功（`code=0`）。 |
| 400 | 请求参数缺失或格式错误（`code=400`，`error` 中有具体说明）。 |
| 500 | 网关或工具内部异常（`code=500`，`error` 中有具体说明）。 |

---

## 2. 接口详情

### 2.1 生产消息给 Arm Agent（send_to_arm_agent）

- **接口说明**：对应工具 `send_to_arm_agent(content: str) -> str`。把一条消息追加到 `message_from_voice_agent_queue` 队尾，供 arm agent 空闲时自动消费或忙碌时主动消费。**意图明确约束**：人的任务意图不明确（缺颜色、缺位置等关键信息，或人只是在闲聊/问进度）时不得调用本工具，先通过对话跟人确认清楚；任务下发后，人的变更/追加/取消请求在意图明确后即可发送。
- **URL**：`POST /api/v1/send_to_arm_agent`
- **输入**：JSON 请求体，字段如下。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `content` | string | 是 | 要发送给 arm agent 的消息全文，自然语言，如 `抓取 red 物块并放到 (1.0,2.0,3.0)。` |

- **输出**（HTTP 200）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | int | 固定为 `0`。 |
| `result` | string | 固定为 `发送成功`。 |
| `error` | null | 固定为 `null`。 |

- **调用示例**：

```bash
curl -X POST http://127.0.0.1:8001/api/v1/send_to_arm_agent \
  -H "Content-Type: application/json" \
  -d '{"content": "抓取 red 物块并放到 (1.0,2.0,3.0)。"}'
```

```json
{
  "code": 0,
  "result": "发送成功",
  "error": null
}
```

---

### 2.2 消费来自 Arm Agent 的消息（get_message_from_arm_agent）

- **接口说明**：对应工具 `get_message_from_arm_agent() -> str`。排空 `message_from_arm_agent_queue`，把 arm agent 上报的进度/结果/求助等全部消息一次性取出。**调用时机**：编排层随每条人类 user 消息注入 `<queue_status>empty/not empty</queue_status>` 状态栏，voice agent 仅在状态为 `not empty` 时调用本工具主动消费；消费结果作为 user 消息进入上下文后，由 voice agent 用语音转述给人。
- **URL**：`POST /api/v1/get_message_from_arm_agent`
- **输入**：无请求参数，无请求体。
- **输出**（HTTP 200）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | int | 固定为 `0`。 |
| `result` | string | 队列非空时为 `all_messages_from_arm_agent:消息1;消息2;...`（多条消息按入队顺序以英文分号 `;` 拼接）；队列为空时固定为 `当前没有新消息`。 |
| `error` | null | 固定为 `null`。 |

- **调用示例**：

```bash
curl -X POST http://127.0.0.1:8001/api/v1/get_message_from_arm_agent
```

```json
{
  "code": 0,
  "result": "all_messages_from_arm_agent:已到达物块位置;有这种颜色的物块，且夹取物块成功",
  "error": null
}
```

---

## 3. 队列与状态栏约定

- 系统有两条 FIFO 队列，由编排运行时持有，两个工具网关（8000/8001）对接同一后端：
  - `message_from_voice_agent_queue`（voice → arm 方向）：本网关的 `send_to_arm_agent` 生产，具身网关的 `get_message_from_voice_agent` 消费。
  - `message_from_arm_agent_queue`（arm → voice 方向）：具身网关的 `send_to_voice_agent` 生产，本网关的 `get_message_from_arm_agent` 消费。
- **状态栏注入（编排层职责，非工具）**：voice agent 侧，每一条人类 user 消息进入上下文时注入 `<queue_status>empty/not empty</queue_status>`，反映 `message_from_arm_agent_queue` 当时是否非空。
- **消费结果的消息形式**：`get_message_from_arm_agent` 的返回字符串以一条 user 消息进入上下文，形如 `all_messages_from_arm_agent:...`。
- 人优先原则：队列消息只能通过"状态栏感知 + 主动消费"进入上下文，不得直接插队打断当前推理。

---

## 4. 接口汇总

| 工具 | 方法 | 路径 | 输入 | 业务结果字段 |
| --- | --- | --- | --- | --- |
| `send_to_arm_agent` | POST | `/api/v1/send_to_arm_agent` | `content`（string，必填） | `result` = `发送成功` |
| `get_message_from_arm_agent` | POST | `/api/v1/get_message_from_arm_agent` | 无 | `result` = `all_messages_from_arm_agent:...` / `当前没有新消息` |
