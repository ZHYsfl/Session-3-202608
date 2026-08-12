# 第9部分代码说明

本说明只覆盖500题有效性验证。最终产品的RAG回答生成位于独立的第6部分交付包。

## 一、整体数据流

```text
questions_500.json
        |
        +---------------------------+
        |                           |
        v                           v
  answer_pure_llm()           检索/重排模块
  不传外部证据                      |
        |                           v
        |                    list[Metadata]
        |                           |
        |                           v
        |                     answer_rag()
        |                           |
        +------------+--------------+
                     v
                 judge_once()
               匿名A/B单次评判
                     |
                     v
              results.jsonl
                     |
                     v
               summary.json
```

## 二、各代码文件

### `llm_client.py`

这是最底层模型适配器。

- `ModelConfig`保存协议、API地址、模型名和生成参数。
- `load_model_configs()`读取回答模型和裁判模型配置。
- `generate_text()`根据配置选择：
  - `responses`：请求 `{base_url}/responses`；
  - `chat_completions`：请求 `{base_url}/chat/completions`。
- `_post_json()`负责HTTP请求和Bearer认证。
- `parse_json_output()`要求模型返回JSON对象，并兼容模型偶尔添加的Markdown代码围栏。

该层不包含医学业务规则，因此以后切换模型通常只修改配置，不修改代码。

### `model_pipeline.py`

这是实验的核心业务接口。

#### `answer_pure_llm(question, model)`

接收一条问题，不接收论文。它根据问题的 `track` 选择对应提示词，并明确告诉模型不能声称已检索、不能虚构引用。

#### `answer_rag(question, metadata, model)`

接收同一问题和检索模块给出的 `list[Metadata]`。纯LLM与RAG传入相同的 `ModelConfig`，确保两组使用同一底座和参数。

#### `judge_once(...)`

将两个回答随机映射为匿名A/B，调用裁判一次，再把裁判的A/B胜负还原为：

- `rag`
- `pure_llm`
- `tie`

结果中保存 `blind_mapping`，方便审计裁判结论，但裁判提示词本身看不到真实身份。

### `run_batch.py`

这是正式500题入口。

运行流程：

1. 加载并检查500题及检索结果。
2. 读取模型配置。
3. 读取已有 `results.jsonl`，跳过成功题目。
4. 对每题依次运行纯LLM、RAG和裁判。
5. 每完成一题立即追加写入JSONL，减少中断损失。
6. API临时失败时指数退避重试。
7. 单题最终失败时写入 `errors.jsonl`，继续处理其他题目。
8. 运行结束后生成 `summary.json`。

主要参数：

- `--questions`：500题文件。
- `--retrieval`：500题检索结果文件。
- `--config`：模型配置。
- `--limit N`：只运行前N题，适合先检查接口。
- `--retries`：单次调用失败后的重试次数，默认2。
- `--retry-delay`：首次重试等待秒数，后续指数增加。
- `--request-interval`：题目之间额外等待时间，用于控制调用速率。

### `summarize_results.py`

无需重新调用模型即可从已有 `results.jsonl` 重建统计报告。如果同一题出现多条成功记录，以最后一条为准。

### `validate_questions.py`

检查：

- 是否恰好500题；
- 编号是否为Q001至Q500且不重复；
- 是否只有规定字段；
- 字段类型及证据类型是否合法；
- 三赛道是否分别为200、150、150题。

## 三、关键接口类型

### Question

```python
{
    "id": str,
    "question": str,
    "track": "track_1" | "track_2" | "track_3",
    "expected_evidence_type": str,
    "notes": str,
    "should_abstain": bool,
}
```

### Metadata

```python
{
    "title": str | None,
    "year": str | None,
    "source_type": str | None,
    "id": str | None,
    "text_summary": str | None,
    "text": str | None,
    "image_summary": list[str] | None,
    "image_base_64": list[str] | None,
    "evidence_level": str | None,
    "last_retrieved_at": str | None,
}
```

## 四、统计口径

主指标：

```text
RAG排除平局胜率 = RAG胜场 / (RAG胜场 + 纯LLM胜场)
```

另外报告：

- RAG总胜率：RAG胜场 / 已完成题数；
- 平局率：平局数 / 已完成题数；
- 三条赛道各自的RAG胜、纯LLM胜和平局数量。

`summary.json`中的 `target_met` 只表示主胜率是否达到0.95，不会修改或美化实验结果。

## 五、对接时最可能需要修改的位置

通常只修改三个地方：

1. `model_config.json`：填写模型API。
2. `questions_500.json`：填入最终500题。
3. `retrieval_results.json`：填入前序同学的检索与重排结果。

第9部分运行时读取同目录的 `evaluation_prompts.snapshot.json`。该文件是三个赛道正式提示词的固定快照，避免评测过程中Prompt发生变化。

若对方能在Python进程中直接返回 `list[Metadata]`，也可以直接调用 `answer_rag()` 和 `judge_once()`，不必先落盘为JSON。

## 六、调用量估算

每题固定调用三次：

- 纯LLM回答一次；
- RAG回答一次；
- 裁判一次。

500题完整实验至少产生1500次模型调用，不含失败重试。正式运行前应先用 `--limit 1`，再用较小批次检查费用、速率限制和输出格式。
