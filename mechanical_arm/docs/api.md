# Arm Skill Server REST API

本网关是 `docs/api_of_embodied_tool.md`（**Source of Truth**）的**实现**。接口路径、
字段、返回字符串、业务语义均逐字遵循该文档，未获负责人同意不修改。本文件记录实现
细节（格式化、错误分支、示例），便于联调。

- Base URL：`http://127.0.0.1:8000`
- 启动：`python scripts/run_api.py`
- 协议：HTTP + JSON（`Content-Type: application/json; charset=utf-8`）
- 全部为同步阻塞调用：工具执行完毕才返回。

## 通用响应结构

```json
{ "code": 0, "result": "...", "error": null }
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | int | `0` 网关成功；`400` 参数错误；`500` 内部异常 |
| `result` | string \| null | 工具的原始返回字符串，原样透传；失败时为 `null` |
| `error` | string \| null | 网关层错误描述；成功时为 `null` |

HTTP 状态码：`200` 成功；`400` 参数缺失/格式错误；`500` 工具或网关内部异常。

## 格式化约定（实现选择，已在 architecture_notes.md 记录，待确认）

- 坐标分量 `{x}` `{y}` `{z}`：`:.3f`（如 `0.180`）
- 误差 `{error}`：`:.4f`（如 `0.0123`）
- 协议只约束字符串形状，不约束小数位数。

## 1. 获取当前坐标

`GET /api/v1/get_current_coordinates`

- 输入：无。
- 返回：`result = "我的坐标是{x},{y},{z}"`。

示例：

```bash
curl -X GET http://127.0.0.1:8000/api/v1/get_current_coordinates
```

```json
{ "code": 0, "result": "我的坐标是0.180,0.000,0.170", "error": null }
```

## 2. 移动到指定坐标

`POST /api/v1/move_to_coordinates`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `x` | string | 是 | 目标 x 坐标（数值字符串） |
| `y` | string | 是 | 目标 y 坐标（数值字符串） |
| `z` | string | 是 | 目标 z 坐标（数值字符串） |

流程：目标投影到可达球面 → IK → 50Hz 闭环 → 真实测量误差与 `position_threshold_m`
（默认 0.02）比较。

- 误差 < 阈值：`result = "成功到达{x},{y},{z}"`
- 否则：`result = "未到达{x},{y},{z}，误差是{error}"`（含不可达 / 超时 / 取消分支，
  误差为真实测量值，不编造）

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/move_to_coordinates \
  -H "Content-Type: application/json" \
  -d '{"x": "0.20", "y": "-0.05", "z": "0.15"}'
```

```json
{ "code": 0, "result": "成功到达0.200,-0.050,0.150", "error": null }
```

## 3. 抓取指定颜色物块

`POST /api/v1/grab_the_block`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `color` | string | 是 | 可任意输入；仅 `yellow` / `red` / `white` 合法 |

流程（对应协议 §2.3）：

1. 颜色不合法 → `没有这种颜色的物块，无法夹取`
2. 场景无该颜色物块 → `没有这种颜色的物块，无法夹取`
3. 尝试夹取（移到上方 → 下降 → 闭夹爪 → 抬升）
   - 成功 → `有这种颜色的物块，且夹取物块成功`
   - 失败（不可达 / 未夹住 / 取消） → `有这种颜色的物块，但夹取物块失败`

Phase 1 物块状态为仿真 ground truth；Phase 2 起由视觉感知替换「是否存在」的判定。

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/grab_the_block \
  -H "Content-Type: application/json" \
  -d '{"color": "red"}'
```

```json
{ "code": 0, "result": "有这种颜色的物块，且夹取物块成功", "error": null }
```

## 4. 释放当前抓取的物块

`POST /api/v1/release_the_block`

- 未夹取物块 → `本身就没加起来物块`
- 释放成功 → `成功释放物块`
- 释放失败（如夹爪卡死，测试用 `set_gripper_fault` 模拟） → `释放物块失败，请用手直接拿出来物块`

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/release_the_block
```

```json
{ "code": 0, "result": "成功释放物块", "error": null }
```

## 5. 生产消息给 Voice Agent

`POST /api/v1/send_to_voice_agent`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `content` | string | 是 | 要发送给 voice agent 的消息全文 |

返回：`result = "发送成功"`。

## 6. 消费来自 Voice Agent 的消息

`POST /api/v1/get_message_from_voice_agent`

- 队列非空：`result = "all_messages_from_voice_agent:消息1;消息2;..."`（按入队顺序用英文分号 `;` 拼接）
- 队列为空：`result = "当前没有新消息"`

调用后队列被排空（一次取完）。

## 错误响应

### HTTP 400（参数缺失 / 格式错误）

```json
{ "code": 400, "result": null, "error": "缺少必填字段: z" }
```

其他 400 示例：`缺少必填字段: color`、`缺少必填字段: content`、
`字段 x 必须是数值字符串, 实际: 'abc'`。

### HTTP 500（网关 / 工具内部异常）

```json
{ "code": 500, "result": null, "error": "<服务端错误信息>" }
```

## 接口汇总

| 工具 | 方法 | 路径 |
| --- | --- | --- |
| `get_current_coordinates` | GET | `/api/v1/get_current_coordinates` |
| `move_to_coordinates` | POST | `/api/v1/move_to_coordinates` |
| `grab_the_block` | POST | `/api/v1/grab_the_block` |
| `release_the_block` | POST | `/api/v1/release_the_block` |
| `send_to_voice_agent` | POST | `/api/v1/send_to_voice_agent` |
| `get_message_from_voice_agent` | POST | `/api/v1/get_message_from_voice_agent` |
