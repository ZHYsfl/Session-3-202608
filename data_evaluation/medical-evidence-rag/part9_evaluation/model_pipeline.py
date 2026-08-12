"""Three-track answer and judge entry points.

The retrieval team should pass list[Metadata] to answer_rag and judge_once.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from llm_client import ModelConfig, generate_text, parse_json_output


ROOT = Path(__file__).resolve().parent
PROMPTS_PATH = ROOT / "evaluation_prompts.snapshot.json"


def load_prompts() -> dict[str, Any]:
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def _fill(template: str, values: dict[str, Any]) -> str:
    result = template
    for key, value in values.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, indent=2)
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = ""
        else:
            rendered = str(value)
        result = result.replace("{{" + key + "}}", rendered)
    return result


def _track_key(question: dict[str, Any]) -> str:
    track = question["track"]
    if track not in {"track_1", "track_2", "track_3"}:
        raise ValueError(f"未知赛道: {track}")
    return track


def _answer_schema_prompt(schema: dict[str, Any]) -> str:
    return "\n\n必须只返回以下结构的JSON：\n" + json.dumps(
        schema, ensure_ascii=False, indent=2
    )


def answer_pure_llm(question: dict[str, Any], model: ModelConfig) -> dict[str, Any]:
    """Arm A: answer with model knowledge only, without retrieved metadata."""
    prompts = load_prompts()
    track = _track_key(question)

    if track == "track_3":
        track_prompt = prompts[track]
        system = (
            track_prompt["shared_answer_prompt"]
            + "\n\n"
            + track_prompt["arm_a_pure_llm"]["context_instruction"]
            + _answer_schema_prompt(track_prompt["answer_output_schema"])
        )
        user = _fill(track_prompt["arm_a_pure_llm"]["user_template"], question)
    else:
        track_prompt = prompts[track]
        # Tracks 1 and 2 use the same communication style, but receive no evidence.
        system = (
            track_prompt["system_prompt"]
            + "\n\n本次为纯LLM对照组，没有提供任何检索证据。不得声称已经检索，"
            "不得虚构引用；证据不足时应明确说明。"
            + _answer_schema_prompt(track_prompt["output_schema"])
        )
        user = (
            f"问题：{question['question']}\n"
            f"期望证据类型：{question['expected_evidence_type']}\n"
            f"应否拒答：{str(question['should_abstain']).lower()}"
        )

    return parse_json_output(generate_text(model, system, user))


def answer_rag(
    question: dict[str, Any], metadata: list[dict[str, Any]], model: ModelConfig
) -> dict[str, Any]:
    """Arm B: answer the same question with retrieved list[Metadata]."""
    prompts = load_prompts()
    track = _track_key(question)
    track_prompt = prompts[track]
    values = {**question, "retrieved_metadata": metadata}

    if track == "track_3":
        system = (
            track_prompt["shared_answer_prompt"]
            + "\n\n"
            + track_prompt["arm_b_rag"]["context_instruction"]
            + _answer_schema_prompt(track_prompt["answer_output_schema"])
        )
        user = _fill(track_prompt["arm_b_rag"]["user_template"], values)
    else:
        system = track_prompt["system_prompt"] + _answer_schema_prompt(
            track_prompt["output_schema"]
        )
        user = _fill(track_prompt["user_template"], values)

    return parse_json_output(generate_text(model, system, user))


def judge_once(
    question: dict[str, Any],
    metadata: list[dict[str, Any]],
    pure_answer: dict[str, Any],
    rag_answer: dict[str, Any],
    judge_model: ModelConfig,
    seed: int | None = None,
) -> dict[str, Any]:
    """Blindly judge one question once; randomly map systems to A/B."""
    prompts = load_prompts()
    judge_prompt = prompts["track_3"]["judge"]
    rng = random.Random(seed)
    rag_is_a = bool(rng.getrandbits(1))
    answer_a = rag_answer if rag_is_a else pure_answer
    answer_b = pure_answer if rag_is_a else rag_answer

    values = {
        "question_id": question["id"],
        "question": question["question"],
        "expected_evidence_type": question["expected_evidence_type"],
        "should_abstain": question["should_abstain"],
        "notes": question["notes"],
        "retrieved_metadata": metadata,
        "answer_a": answer_a,
        "answer_b": answer_b,
    }
    system = judge_prompt["system_prompt"] + _answer_schema_prompt(
        judge_prompt["output_schema"]
    )
    user = _fill(judge_prompt["user_template"], values)
    result = parse_json_output(generate_text(judge_model, system, user))

    anonymous_winner = result.get("winner")
    if anonymous_winner == "tie":
        actual_winner = "tie"
    elif anonymous_winner == "A":
        actual_winner = "rag" if rag_is_a else "pure_llm"
    elif anonymous_winner == "B":
        actual_winner = "pure_llm" if rag_is_a else "rag"
    else:
        raise ValueError(f"裁判winner字段无效: {anonymous_winner!r}")

    result["actual_winner"] = actual_winner
    result["blind_mapping"] = {"A": "rag" if rag_is_a else "pure_llm", "B": "pure_llm" if rag_is_a else "rag"}
    return result
