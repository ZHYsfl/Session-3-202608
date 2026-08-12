"""Unified model API used by the three-track experiment.

No third-party Python package is required. API keys are read only from
environment variables named in model_config.json.
"""

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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        return cls(**value)


class ModelAPIError(RuntimeError):
    pass


def load_model_configs(path: str) -> tuple[ModelConfig, ModelConfig]:
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    return (
        ModelConfig.from_dict(data["answer_model"]),
        ModelConfig.from_dict(data["judge_model"]),
    )


def _post_json(url: str, api_key: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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


def _responses_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
    if not parts:
        raise ModelAPIError("Responses API响应中没有output_text")
    return "".join(parts)


def generate_text(config: ModelConfig, system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise ModelAPIError(
            f"环境变量{config.api_key_env}未设置；请把API密钥放入该环境变量"
        )

    base_url = config.base_url.rstrip("/")
    if config.protocol == "responses":
        payload = {
            "model": config.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_output_tokens": config.max_output_tokens,
            "store": False,
        }
        response = _post_json(
            f"{base_url}/responses", api_key, payload, config.timeout_seconds
        )
        return _responses_output_text(response)

    if config.protocol == "chat_completions":
        payload = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_output_tokens,
        }
        response = _post_json(
            f"{base_url}/chat/completions", api_key, payload, config.timeout_seconds
        )
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelAPIError("Chat Completions响应格式不正确") from exc

    raise ValueError("protocol只能是responses或chat_completions")


def parse_json_output(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ModelAPIError(f"模型没有返回合法JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelAPIError("模型JSON输出必须是对象")
    return value
