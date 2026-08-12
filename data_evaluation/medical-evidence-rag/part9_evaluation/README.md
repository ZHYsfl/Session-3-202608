# 第9部分：500题有效性评测

本包只负责证明最终RAG程序是否比同一底座模型仅依靠自身知识回答更有效。500题是固定测试集，不属于最终产品的用户问答功能。

## 对照设计

- Arm A：纯LLM，不接收论文或检索结果。
- Arm B：RAG，接收检索与重排后的 `list[Metadata]`。
- 两组使用同一回答模型、同一版本、同一温度、同一top_p和同一输出长度。
- 第三模型作为裁判，每题匿名评判一次，允许平局。
- 主胜率：`RAG胜场 / (RAG胜场 + 纯LLM胜场)`。
- 目标为95%，但程序必须如实报告结果，不能强迫裁判偏向RAG。

## 500题

三个赛道共同组成500题：

- track_1：200题；
- track_2：150题；
- track_3：150题。

本包已包含根据用户提供的外部数据包生成的 `questions_500.json`。如需换用其他同学提供的题库，也可以从 `questions_500.template.json` 重新填写。校验命令：

```powershell
python validate_questions.py questions_500.json
```

逐题来源和生成依据见 `question_source_map.json`；整体质量与限制见 `QUESTION_DATASET_REPORT.md`。

当前题库采用混合设计：300道通用型真实用户问题，加200道论文针对型专业问题。此前的全论文针对型版本保存在 `previous_paper_focused_version/`，仅供版本对比，不用于当前实验。

## 检索结果接口

前序模块按照 `retrieval_results.schema.json` 提供每题的 `list[Metadata]`。复制 `retrieval_results.template.json` 为 `retrieval_results.json` 并填充Q001至Q500。

## 模型配置

复制 `model_config.template.json` 为 `model_config.json`，填写：

- `answer_model`：纯LLM与RAG共同使用；
- `judge_model`：单次匿名评判模型。

密钥通过 `ANSWER_MODEL_API_KEY` 和 `JUDGE_MODEL_API_KEY` 环境变量传入。

## 运行

先试跑1题：

```powershell
python run_batch.py --questions questions_500.json --retrieval retrieval_results.json --config model_config.json --limit 1
```

确认后运行全部题目：

```powershell
python run_batch.py --questions questions_500.json --retrieval retrieval_results.json --config model_config.json
```

输出位于 `results/`：

- `results.jsonl`：每题两个回答和裁判结果；
- `errors.jsonl`：失败题目；
- `summary.json`：主胜率、平局率和分赛道统计。

程序支持断点续跑，会跳过已成功题目。

## 核心代码

- `model_pipeline.py`：`answer_pure_llm()`、`answer_rag()`、`judge_once()`。
- `run_batch.py`：500题批处理与断点续跑。
- `summarize_results.py`：结果统计。
- `evaluation_prompts.snapshot.json`：运行时使用的提示词快照。
- `questions_500.json`：基于真实数据生成的完整500题。
- `question_source_map.json`：每题来源映射和审计信息。
- `question_generation_report.json`：题库统计报告。
- `QUESTION_DATASET_REPORT.md`：生成方法、质量检查和限制。
- `CODE_EXPLANATION.md`：详细代码说明。
