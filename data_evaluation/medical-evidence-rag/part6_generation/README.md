# 第6部分：Generation最终产品模块

本包只负责最终产品的回答生成。用户提出问题后，前序检索与重排模块返回 `list[Metadata]`，本模块调用我们的RAG模型生成带引用回答。

这里没有500题、纯LLM对照或裁判模型；这些只属于第9部分的有效性验证。

## 唯一对外接口

```python
from generation import generate_rag_answer

answer = generate_rag_answer(
    question=user_question,
    retrieved_metadata=top_k_metadata,
    model=rag_model,
)
```

## 接入步骤

1. 复制 `model_config.template.json` 为 `model_config.json`。
2. 填写最终产品模型名称和API地址。
3. 在环境变量 `RAG_MODEL_API_KEY` 中放入密钥。
4. 将第5步重排结果 `list[Metadata]` 传入 `generate_rag_answer()`。

PowerShell设置密钥：

```powershell
$env:RAG_MODEL_API_KEY="你的密钥"
```

单题示例见 `run_generation_example.py`。输入字段规范见 `metadata.schema.json`。

## 文件说明

- `generation.py`：第6部分唯一业务接口。
- `llm_client.py`：模型API调用。
- `generation_prompt.json`：最终生成提示词。
- `model_config.template.json`：模型配置模板。
- `metadata.schema.json`：前序模块的输入接口。
- `run_generation_example.py`：调用示例。
