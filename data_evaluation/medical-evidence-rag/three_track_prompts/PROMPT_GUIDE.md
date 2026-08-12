# 提示词使用说明

## 公共变量

- `{{question}}`：用户或测试集问题。
- `{{expected_evidence_type}}`：期望证据类型。
- `{{should_abstain}}`：题目是否预期拒答或报告证据不足。
- `{{retrieved_metadata}}`：检索与重排产生的 `list[Metadata]`。

## 赛道一

读取 `track_1.system_prompt` 作为系统提示词，将问题和证据填入 `track_1.user_template`。

模型必须输出 `track_1.output_schema` 对应的JSON。主要内容包括证据性回答、证据卡片、不确定性、安全边界和引用。

## 赛道二

读取 `track_2.system_prompt` 和 `track_2.user_template`。输出必须适合普通消费者阅读，并包含一句话结论、已知证据、未知内容、适用人群、一般信息、引用和免责声明。

## 赛道三回答模型

两组共同使用 `track_3.shared_answer_prompt`：

- 纯LLM再拼接 `track_3.arm_a_pure_llm.context_instruction`；
- RAG再拼接 `track_3.arm_b_rag.context_instruction`。

两组必须使用同一个底座模型和完全相同的生成参数。唯一核心差异是RAG收到 `retrieved_metadata`，纯LLM不收到。

## 裁判模型

读取 `track_3.judge.system_prompt` 和 `track_3.judge.user_template`。

需要填入：

- `{{question_id}}`
- `{{question}}`
- `{{expected_evidence_type}}`
- `{{should_abstain}}`
- `{{notes}}`
- `{{retrieved_metadata}}`
- `{{answer_a}}`
- `{{answer_b}}`

程序必须随机隐藏A/B身份。裁判输出六项评分、重大错误、`A/B/tie`、理由和置信度。每题只调用裁判一次，不进行交换复评或人工复核。

## 注意

- 不得在裁判Prompt中写“优先判RAG胜”。
- 引用必须对应真实 `Metadata.id`。
- 提示词要求结构化JSON是为了方便批量统计；如果模型偶尔不能稳定输出JSON，应在调用层增加格式重试，而不是改变评价标准。
