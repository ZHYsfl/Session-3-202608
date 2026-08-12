"""Part 6 public interface: final product uses only the RAG answer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_client import ModelConfig, generate_text, parse_json_output


ROOT = Path(__file__).resolve().parent


def _render(template: str, question: str, metadata: list[dict[str, Any]]) -> str:
    return template.replace("{{question}}", question).replace(
        "{{retrieved_metadata}}", json.dumps(metadata, ensure_ascii=False, indent=2)
    )


def generate_rag_answer(
    question: str,
    retrieved_metadata: list[dict[str, Any]],
    model: ModelConfig,
) -> dict[str, Any]:
    """Generate the final product answer from a question and reranked evidence."""
    prompts = json.loads((ROOT / "generation_prompt.json").read_text(encoding="utf-8"))
    system = prompts["system_prompt"] + "\n\n输出JSON结构：\n" + json.dumps(
        prompts["output_schema"], ensure_ascii=False, indent=2
    )
    user = _render(prompts["user_template"], question, retrieved_metadata)
    return parse_json_output(generate_text(model, system, user))
