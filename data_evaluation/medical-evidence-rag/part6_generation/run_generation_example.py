from __future__ import annotations

import json
from pathlib import Path

from generation import generate_rag_answer
from llm_client import load_rag_model


ROOT = Path(__file__).resolve().parent
model = load_rag_model(str(ROOT / "model_config.json"))

# 前面同学的检索与重排结果直接放入此数组。
retrieved_metadata: list[dict] = []
result = generate_rag_answer("请输入真实用户问题", retrieved_metadata, model)
print(json.dumps(result, ensure_ascii=False, indent=2))
