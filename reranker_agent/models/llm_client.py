"""Provider-independent, structured-output LLM client."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class LLMResponseError(RuntimeError):
    """Raised when an LLM response cannot be validated as strict JSON."""


class LLMClient(ABC):
    @abstractmethod
    def complete_json(
        self, prompt: str, response_model: type[ResponseT]
    ) -> ResponseT:
        """Return a response validated against ``response_model``."""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise LLMResponseError("The model response did not contain a JSON object")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMResponseError("The model returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise LLMResponseError("The model response must be a JSON object")
    return value


class OpenAICompatibleLLMClient(LLMClient):
    """Works with OpenAI, Qwen DashScope, and compatible chat APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60,
        max_retries: int = 2,
        json_mode: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("An API key is required for the selected LLM provider")
        self.model = model
        self.temperature = temperature
        self.attempts = max_retries + 1
        self.json_mode = json_mode
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def complete_json(
        self, prompt: str, response_model: type[ResponseT]
    ) -> ResponseT:
        last_error: Exception | None = None
        retry_prompt = prompt
        for _ in range(self.attempts):
            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a careful academic search reranker. "
                                "Return one strict JSON object and no markdown."
                            ),
                        },
                        {"role": "user", "content": retry_prompt},
                    ],
                }
                if self.json_mode:
                    request["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(
                    **request,
                )
                content = response.choices[0].message.content or ""
                return response_model.model_validate(_extract_json(content))
            except (ValidationError, LLMResponseError, IndexError) as exc:
                last_error = exc
                retry_prompt = (
                    prompt
                    + "\n\nYour previous response was invalid. Return only JSON "
                    + f"matching this schema: {response_model.model_json_schema()}"
                )
        raise LLMResponseError(f"Could not obtain valid structured output: {last_error}")


def _marked_json(prompt: str, marker: str) -> Any:
    pattern = rf"{re.escape(marker)}\s*(.+?)(?=\n[A-Z_]+_JSON:|\nOUTPUT_SCHEMA:|\Z)"
    match = re.search(pattern, prompt, flags=re.DOTALL)
    if not match:
        raise LLMResponseError(f"Mock client could not find prompt marker {marker}")
    return json.loads(match.group(1).strip())


def _query_from_prompt(prompt: str) -> str:
    match = re.search(r"QUERY:\s*(.+?)(?=\n[A-Z_]+_JSON:)", prompt, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _tokens(value: Any) -> set[str]:
    text = json.dumps(value, ensure_ascii=False).lower() if not isinstance(value, str) else value.lower()
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text))


def _heuristic_score(query: str, paper: dict[str, Any]) -> float:
    """Deterministic offline scorer used only by the mock provider."""
    query_tokens = _tokens(query)
    title_tokens = _tokens(paper.get("title") or "")
    body_tokens = _tokens(
        {
            "summary": paper.get("text_summary"),
            "images": paper.get("image_summary"),
            "evidence": paper.get("evidence_level"),
        }
    )
    title_overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1)
    body_overlap = len(query_tokens & body_tokens) / max(len(query_tokens), 1)
    evidence_bonus = 0.4 if paper.get("evidence_level") else 0.0
    score = 1.0 + 6.0 * title_overlap + 3.0 * body_overlap + evidence_bonus
    return round(min(10.0, score), 3)


class MockLLMClient(LLMClient):
    """Offline deterministic implementation for examples, CI, and smoke tests."""

    def complete_json(
        self, prompt: str, response_model: type[ResponseT]
    ) -> ResponseT:
        fields = response_model.model_fields
        query = _query_from_prompt(prompt)
        if {"paper_id", "score", "reason"}.issubset(fields):
            paper = _marked_json(prompt, "PAPER_JSON:")
            score = _heuristic_score(query, paper)
            payload = {
                "paper_id": paper["id"],
                "score": score,
                "reason": f"Offline lexical relevance estimate: {score:.3f}/10.",
            }
        elif {"winner", "reason"}.issubset(fields):
            paper_a = _marked_json(prompt, "PAPER_A_JSON:")
            paper_b = _marked_json(prompt, "PAPER_B_JSON:")
            score_a = _heuristic_score(query, paper_a)
            score_b = _heuristic_score(query, paper_b)
            winner = paper_a if score_a >= score_b else paper_b
            payload = {
                "winner": winner["id"],
                "reason": (
                    f"Offline comparison selected {winner['id']} "
                    f"({score_a:.3f} vs {score_b:.3f})."
                ),
            }
        elif {"ranking", "reason"}.issubset(fields):
            papers = _marked_json(prompt, "CANDIDATES_JSON:")
            scored = [(_heuristic_score(query, paper), index, paper) for index, paper in enumerate(papers)]
            scored.sort(key=lambda item: (-item[0], item[1]))
            payload = {
                "ranking": [item[2]["id"] for item in scored],
                "reason": "Offline lexical ordering for this sliding window.",
            }
        else:
            raise LLMResponseError("Unsupported response schema for mock provider")
        return response_model.model_validate(payload)


def build_llm_client(config: dict[str, Any]) -> LLMClient:
    provider = str(config.get("provider", "mock")).lower()
    if provider == "mock":
        return MockLLMClient()
    if provider not in {"openai", "qwen", "compatible"}:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    env_name = str(
        config.get("api_key_env")
        or ("QWEN_API_KEY" if provider == "qwen" else "OPENAI_API_KEY")
    )
    api_key = os.getenv(env_name, "")
    base_url = config.get("base_url")
    if provider == "qwen" and not base_url:
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return OpenAICompatibleLLMClient(
        api_key=api_key,
        model=str(config.get("model", "gpt-4.1-mini")),
        base_url=base_url,
        temperature=float(config.get("temperature", 0.0)),
        timeout=float(config.get("timeout", 60)),
        max_retries=int(config.get("max_retries", 2)),
        json_mode=bool(config.get("json_mode", True)),
    )
