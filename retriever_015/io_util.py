from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from metadata import Metadata


def _as_metadata_list(raw: Any) -> list[Metadata]:
    if raw is None:
        return []
    if isinstance(raw, list):
        out: list[Metadata] = []
        for i, item in enumerate(raw):
            if isinstance(item, Metadata):
                out.append(item)
            elif isinstance(item, dict):
                out.append(Metadata.from_dict(item))
            else:
                raise TypeError(f"metadatas[{i}] must be dict or Metadata, got {type(item)!r}")
        return out
    raise TypeError(f"metadatas must be a list, got {type(raw)!r}")


def _extract_meta_list(payload: Any) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("metadatas", "papers", "items", "data"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        raise ValueError(
            "JSON object must contain a list under 'metadatas', 'papers', 'items', or 'data'"
        )
    raise TypeError(f"Unsupported JSON root type: {type(payload)!r}")


def load_metadatas(source: str | Path | list | dict) -> list[Metadata]:
    """
    从下列来源加载 list[Metadata]：
    - list[dict] / list[Metadata]
    - dict 含 metadatas/papers/...
    - .json 文件路径（数组或上述对象）
    - .jsonl 文件路径
    - '-' 或 path 为 stdin 时读标准输入 JSON
    - JSON 字符串（以 '[' 或 '{' 开头）
    """
    if isinstance(source, list):
        return _as_metadata_list(source)
    if isinstance(source, dict):
        return _as_metadata_list(_extract_meta_list(source))

    if not isinstance(source, (str, Path)):
        raise TypeError(f"Unsupported source type: {type(source)!r}")

    text: str
    s = str(source).strip()
    if s == "-":
        import sys

        text = sys.stdin.read()
    elif s.startswith("[") or s.startswith("{"):
        text = s
    else:
        path = Path(s)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        if path.suffix.lower() == ".jsonl":
            items: list[dict] = []
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise TypeError(f"{path}:{line_no} must be a JSON object")
                items.append(obj)
            return _as_metadata_list(items)
        text = path.read_text(encoding="utf-8")

    payload = json.loads(text)
    return _as_metadata_list(_extract_meta_list(payload))


def dump_metadatas(
    items: list[Metadata],
    path: str | Path | None = None,
) -> list[dict]:
    """序列化为 list[dict]；若 path 给定则写入 JSON（UTF-8）。path='-' 写到 stdout。"""
    data = [m.to_dict() if isinstance(m, Metadata) else Metadata.from_dict(m).to_dict() for m in items]
    if path is None:
        return data
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if str(path) == "-":
        print(text)
    else:
        Path(path).write_text(text + "\n", encoding="utf-8")
    return data


def load_rank_request(source: str | Path | dict) -> tuple[list[Metadata], str, int]:
    """
    加载一包请求：{ "query": str, "k1": int, "metadatas": [...] }。
    返回 (metadatas, query, k1)。
    """
    if isinstance(source, dict):
        payload = source
    else:
        path = Path(source) if str(source) != "-" else None
        if path is None:
            import sys

            text = sys.stdin.read()
        else:
            text = path.read_text(encoding="utf-8")
        payload = json.loads(text)

    if not isinstance(payload, dict):
        raise TypeError("rank request must be a JSON object")

    query = payload.get("query")
    if not isinstance(query, str):
        raise ValueError("rank request requires string field 'query'")

    k1 = payload.get("k1", 10)
    if not isinstance(k1, int) or isinstance(k1, bool):
        raise ValueError("rank request field 'k1' must be an int")

    metas = _as_metadata_list(_extract_meta_list(payload))
    return metas, query, k1


def parse_rank_request(request: dict) -> tuple[list[Metadata], str, int]:
    """解析内存中的 request dict。"""
    return load_rank_request(request)
