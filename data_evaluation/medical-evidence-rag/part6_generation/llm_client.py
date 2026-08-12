"""Standard-library client for Responses or OpenAI-compatible Chat Completions."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    protocol: str
    base_url: str
    model: str
    api_key_env: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 2000
    timeout_seconds: int = 120


class ModelAPIError(RuntimeError):
    pass


def load_rag_model(path: str) -> ModelConfig:
    with open(path, encoding="utf-8") as stream:
        return ModelConfig(**json.load(stream)["rag_model"])


def _post(url: str, key: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ModelAPIError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ModelAPIError(f"无法连接模型API: {exc.reason}") from exc


def generate_text(config: ModelConfig, system_prompt: str, user_prompt: str) -> str:
    key = os.environ.get(config.api_key_env)
    if not key:
        raise ModelAPIError(f"环境变量{config.api_key_env}未设置")
    base = config.base_url.rstrip("/")
    if config.protocol == "responses":
        response = _post(
            f"{base}/responses",
            key,
            {
                "model": config.model,
                "instructions": system_prompt,
                "input": user_prompt,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_output_tokens": config.max_output_tokens,
                "store": False,
            },
            config.timeout_seconds,
        )
        if isinstance(response.get("output_text"), str):
            return response["output_text"]
        parts = [
            content["text"]
            for item in response.get("output", [])
            for content in item.get("content", [])
            if content.get("type") == "output_text" and isinstance(content.get("text"), str)
        ]
        if parts:
            return "".join(parts)
        raise ModelAPIError("Responses API响应中没有output_text")
    if config.protocol == "chat_completions":
        response = _post(
            f"{base}/chat/completions",
            key,
            {
                "model": config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": config.max_output_tokens,
            },
            config.timeout_seconds,
        )
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelAPIError("Chat Completions响应格式不正确") from exc
    raise ValueError("protocol只能是responses或chat_completions")


def parse_json_output(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        cleaned = "\n".join(lines).strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ModelAPIError("模型输出必须是JSON对象")
    return value
