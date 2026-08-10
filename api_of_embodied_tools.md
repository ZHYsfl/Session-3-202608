我们要写具身agent的四个工具，为了联调方便，把这四个工具暴露成restful api.restful api的文档要包含接口是什么，输入和输出的字段各自是什么的详细说明，以及该接口的说明介绍等。

一个工具是get_current_coordinates()->str,工具内部会检测自己的速度，如果速度小于某个阈值，那么工具会返回当前的坐标位置，如果速度大于等于该阈值，工具会返回“我的坐标是{x},{y},{z}”.

一个工具是move_to_coordinates(x:str,y:str,z:str) -> str，工具内部会有办法利用某种算法/网络到达指定的x,y,z处。每次到达后，工具内部应该有检查机制，如果误差小于某个阈值，那么返回“成功到达{x},{y},{z}”，否则返回“未到达{x},{y},{z}，误差是{error}”.

一个工具是grab_the_block(color:str) -> str,意思是抓取特定颜色的物块。我们在真机上只可能提供"yellow","red","white"三种颜色的物块，但是color本身可以随便输入str，如果输入的color不在我们的{"yellow","red","white"}里，那么工具直接返回"没有这种颜色的物块，无法夹取"，如果在，那么调用视觉摄像头，观察有哪些颜色的物块（我们有的时候不会把三种颜色的物块都放在机械臂前面），如果没有color这种颜色的物块，也返回“没有这种颜色的物块，无法夹取”。否则才尝试夹取，如果夹取成功（会有第二个第三视角机位的摄像头判断），返回“有这种颜色的物块，且夹取物块成功”，否则返回“有这种颜色的物块，但夹取物块失败”。（我们保证一次任务一个颜色的物块只有一个。）

一个工具是release_the_block() -> str,意识是放下当前抓取的物块，该工具内部会检测当前真机是否加起来了物块（比如可以用过第二机位摄像头，第一机位自感知等交叉验证），来返回“本身就没加起来物块”，或者的话，要释放物块，可能成功也可能不成功，工具本身也应该有检测机制，成功了就返回“成功释放物块”，否则返回“释放物块失败，请用手直接拿出来物块”

---

# RESTful API 网关文档

## 1. 概述

为了方便联调，将 arm agent（具身 agent）的六个工具通过本地 RESTful API 网关对外暴露：2.1–2.4 为上述四个具身执行工具，2.5–2.6 为异步双 agent 系统的跨 agent 通信工具（生产/消费，设计原理见 `async_dual_agent_system_design.md`）。

- 服务地址（Base URL）：`http://127.0.0.1:8000`
- 协议：HTTP + JSON（`Content-Type: application/json; charset=utf-8`）
- 所有接口均为同步阻塞调用：工具内部执行完毕后才返回响应，联调方无需轮询。
- 与语音工具网关（`http://127.0.0.1:8001`，见 `api_of_voice_tools.md`）对接**同一个队列后端**（见 §3 队列与状态栏约定）。

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

### 2.1 获取当前坐标

- **接口说明**：对应工具 `get_current_coordinates() -> str`。工具内部会检测自身速度：若速度小于阈值，返回当前的坐标位置；若速度大于等于阈值，返回 `我的坐标是{x},{y},{z}`。
- **URL**：`GET /api/v1/get_current_coordinates`
- **输入**：无请求参数，无请求体。
- **输出**（HTTP 200）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | int | 固定为 `0`。 |
| `result` | string | 工具返回的坐标字符串。低速时为当前坐标位置描述；速度大于等于阈值时为 `我的坐标是{x},{y},{z}`，其中 `{x}`、`{y}`、`{z}` 为当前坐标分量。 |
| `error` | null | 固定为 `null`。 |

- **调用示例**：

```bash
curl -X GET http://127.0.0.1:8000/api/v1/get_current_coordinates
```

```json
{
  "code": 0,
  "result": "我的坐标是1.0,2.0,3.0",
  "error": null
}
```

---

### 2.2 移动到指定坐标

- **接口说明**：对应工具 `move_to_coordinates(x: str, y: str, z: str) -> str`。工具内部利用某种算法/网络驱动机械臂到达指定的 `x,y,z` 处。每次到达后内部有检查机制：若实际位置与目标位置的误差小于阈值，返回 `成功到达{x},{y},{z}`；否则返回 `未到达{x},{y},{z}，误差是{error}`。
- **URL**：`POST /api/v1/move_to_coordinates`
- **输入**：JSON 请求体，字段如下（三个字段均必填，类型为字符串，内容为数值的字符串形式，例如 `"1.25"`）。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `x` | string | 是 | 目标位置的 x 坐标。 |
| `y` | string | 是 | 目标位置的 y 坐标。 |
| `z` | string | 是 | 目标位置的 z 坐标。 |

- **输出**（HTTP 200）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | int | 固定为 `0`。 |
| `result` | string | 工具返回的结果字符串。误差小于阈值时为 `成功到达{x},{y},{z}`；否则为 `未到达{x},{y},{z}，误差是{error}`，其中 `{error}` 为实际误差值。 |
| `error` | null | 固定为 `null`。 |

- **调用示例**：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/move_to_coordinates \
  -H "Content-Type: application/json" \
  -d '{"x": "1.0", "y": "2.0", "z": "3.0"}'
```

```json
{
  "code": 0,
  "result": "成功到达1.0,2.0,3.0",
  "error": null
}
```

- **参数错误示例**（缺少字段，HTTP 400）：

```json
{
  "code": 400,
  "result": null,
  "error": "缺少必填字段: z"
}
```

---

### 2.3 抓取指定颜色物块

- **接口说明**：对应工具 `grab_the_block(color: str) -> str`，抓取特定颜色的物块。处理逻辑如下：
  1. 真机上只有 `"yellow"`、`"red"`、`"white"` 三种颜色的物块，但 `color` 可输入任意字符串。若输入的 `color` 不在 `{"yellow","red","white"}` 中，直接返回 `没有这种颜色的物块，无法夹取`。
  2. 若颜色合法，调用视觉摄像头观察机械臂前方实际存在哪些颜色的物块（三种物块不一定都放在机械臂前面）。若没有该颜色的物块，同样返回 `没有这种颜色的物块，无法夹取`。
  3. 若物块存在，尝试夹取，并由第二个第三视角机位的摄像头判断结果：夹取成功返回 `有这种颜色的物块，且夹取物块成功`；否则返回 `有这种颜色的物块，但夹取物块失败`。
  4. 约定：一次任务中同一颜色的物块只有一个。

- **URL**：`POST /api/v1/grab_the_block`
- **输入**：JSON 请求体，字段如下。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `color` | string | 是 | 要抓取的物块颜色。可输入任意字符串，但仅 `"yellow"`、`"red"`、`"white"` 为合法颜色，其余值会直接返回无法夹取。 |

- **输出**（HTTP 200）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | int | 固定为 `0`。 |
| `result` | string | 工具返回的结果字符串，取值仅有三种：<br>1. `没有这种颜色的物块，无法夹取`（颜色非法，或摄像头未观察到该颜色物块）；<br>2. `有这种颜色的物块，且夹取物块成功`；<br>3. `有这种颜色的物块，但夹取物块失败`。 |
| `error` | null | 固定为 `null`。 |

- **调用示例**：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/grab_the_block \
  -H "Content-Type: application/json" \
  -d '{"color": "red"}'
```

```json
{
  "code": 0,
  "result": "有这种颜色的物块，且夹取物块成功",
  "error": null
}
```

- **参数错误示例**（缺少字段，HTTP 400）：

```json
{
  "code": 400,
  "result": null,
  "error": "缺少必填字段: color"
}
```

---

### 2.4 释放当前抓取的物块

- **接口说明**：对应工具 `release_the_block() -> str`，放下当前抓取的物块。处理逻辑如下：
  1. 工具内部先检测真机当前是否真的夹起了物块（可通过第二机位摄像头、第一机位自感知等交叉验证）。若未夹起物块，直接返回 `本身就没加起来物块`。
  2. 若确实夹有物块，则执行释放动作。释放可能成功也可能失败，工具自身带有检测机制：成功返回 `成功释放物块`；失败返回 `释放物块失败，请用手直接拿出来物块`。

- **URL**：`POST /api/v1/release_the_block`
- **输入**：无请求参数，无请求体。
- **输出**（HTTP 200）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | int | 固定为 `0`。 |
| `result` | string | 工具返回的结果字符串，取值仅有三种：<br>1. `本身就没加起来物块`（交叉验证确认当前未夹取物块）；<br>2. `成功释放物块`；<br>3. `释放物块失败，请用手直接拿出来物块`。 |
| `error` | null | 固定为 `null`。 |

- **调用示例**：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/release_the_block
```

```json
{
  "code": 0,
  "result": "成功释放物块",
  "error": null
}
```

---

### 2.5 生产消息给 Voice Agent（send_to_voice_agent）

- **接口说明**：对应工具 `send_to_voice_agent(content: str) -> str`。把一条消息追加到 `message_from_arm_agent_queue` 队尾，供 voice agent 消费后语音转述给人。arm agent 不直接对人输出，进度汇报、任务完成、异常求助（如 `release_the_block` 失败后请人用手取出）都通过本工具上报。
- **URL**：`POST /api/v1/send_to_voice_agent`
- **输入**：JSON 请求体，字段如下。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `content` | string | 是 | 要发送给 voice agent 的消息全文，自然语言，如 `已到达目标位置 (1.0,2.0,3.0) 并成功释放物块，任务完成。` |

- **输出**（HTTP 200）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | int | 固定为 `0`。 |
| `result` | string | 固定为 `发送成功`。 |
| `error` | null | 固定为 `null`。 |

- **调用示例**：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/send_to_voice_agent \
  -H "Content-Type: application/json" \
  -d '{"content": "已到达目标位置 (1.0,2.0,3.0) 并成功释放物块，任务完成。"}'
```

```json
{
  "code": 0,
  "result": "发送成功",
  "error": null
}
```

---

### 2.6 消费来自 Voice Agent 的消息（get_message_from_voice_agent）

- **接口说明**：对应工具 `get_message_from_voice_agent() -> str`。排空 `message_from_voice_agent_queue`，把 voice agent 转发的任务下发/变更/取消等全部消息一次性取出。**调用时机**（pics/4.png 左下角修正后的机制）：
  1. arm agent **空闲时**：由编排层触发本工具自动消费，新任务由此进入上下文（此时队列刚排空，注入的任务消息后**不追加状态栏**）。
  2. arm agent **忙碌时**（工具调用循环中）：编排层在每条 tool response 消息后追加一条 role 为 user、内容为 `<queue_status>empty/not empty</queue_status>` 的状态栏消息，然后让 LLM 继续推理；arm agent 看到 `not empty` 时调用本工具主动消费，从而在任务执行中途感知新指令。
- **URL**：`POST /api/v1/get_message_from_voice_agent`
- **输入**：无请求参数，无请求体。
- **输出**（HTTP 200）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | int | 固定为 `0`。 |
| `result` | string | 队列非空时为 `all_messages_from_voice_agent:消息1;消息2;...`（多条消息按入队顺序以英文分号 `;` 拼接）；队列为空时固定为 `当前没有新消息`。 |
| `error` | null | 固定为 `null`。 |

- **调用示例**：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/get_message_from_voice_agent
```

```json
{
  "code": 0,
  "result": "all_messages_from_voice_agent:用户改主意了，请改抓 yellow 物块;目标位置不变",
  "error": null
}
```

---

## 3. 队列与状态栏约定

- 系统有两条 FIFO 队列，由编排运行时持有，两个工具网关（8000/8001）对接同一后端：
  - `message_from_voice_agent_queue`（voice → arm 方向）：语音网关的 `send_to_arm_agent` 生产，本网关的 `get_message_from_voice_agent` 消费。
  - `message_from_arm_agent_queue`（arm → voice 方向）：本网关的 `send_to_voice_agent` 生产，语音网关的 `get_message_from_arm_agent` 消费。
- **状态栏注入（编排层职责，非工具）**：状态栏在两个 agent 侧统一为独立的 role=user 消息，内容为 `<queue_status>empty/not empty</queue_status>`。arm agent 忙碌时，每条 tool response 消息之后追加一条状态栏消息，反映 `message_from_voice_agent_queue` 当时是否非空，随后让 LLM 继续推理；arm agent 空闲自动消费时注入的任务消息后不追加状态栏（队列刚排空，恒为 empty）。当 tool response、user input、状态栏三者同时存在时，顺序固定为 tool response → user input → 状态栏。
- **消费结果的消息形式**：`get_message_from_voice_agent` 的返回字符串以一条 user 消息进入上下文，形如 `all_messages_from_voice_agent:...`。
- 人优先原则：队列消息只能通过"状态栏感知 + 主动消费"进入上下文，不得直接插队打断当前推理。

---

## 4. 接口汇总

| 工具 | 方法 | 路径 | 输入 | 业务结果字段 |
| --- | --- | --- | --- | --- |
| `get_current_coordinates` | GET | `/api/v1/get_current_coordinates` | 无 | `result` |
| `move_to_coordinates` | POST | `/api/v1/move_to_coordinates` | `x` / `y` / `z`（string，必填） | `result` |
| `grab_the_block` | POST | `/api/v1/grab_the_block` | `color`（string，必填） | `result` |
| `release_the_block` | POST | `/api/v1/release_the_block` | 无 | `result` |
| `send_to_voice_agent` | POST | `/api/v1/send_to_voice_agent` | `content`（string，必填） | `result` = `发送成功` |
| `get_message_from_voice_agent` | POST | `/api/v1/get_message_from_voice_agent` | 无 | `result` = `all_messages_from_voice_agent:...` / `当前没有新消息` |
