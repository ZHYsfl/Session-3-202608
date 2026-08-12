"""
轻量级索引清单管理
====================
用于记录：
- 哪些 paper_id 已入库
- 每个 paper_id 对应哪些 chunk_ids
- 支持增量更新、删除、热更新

避免全量加载 split_blocks 到内存。
"""
from __future__ import annotations

import json
import os
from typing import Any

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "chroma_db", "index_manifest.json")


def _load_manifest() -> dict[str, Any]:
    if not os.path.exists(MANIFEST_PATH):
        return {}
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_manifest(manifest: dict[str, Any]):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def is_paper_indexed(paper_id: str) -> bool:
    """检查 paper_id 是否已在索引清单中。"""
    manifest = _load_manifest()
    return paper_id in manifest


def clear_manifest():
    """清空索引清单（用于重建向量库时同步）。"""
    _save_manifest({})


_INDEX_VERSION_KEY = "__index_version__"


def get_index_version() -> str | None:
    """获取当前索引版本号。"""
    manifest = _load_manifest()
    version = manifest.get(_INDEX_VERSION_KEY)
    return version if version else None


def set_index_version(version: str):
    """设置索引版本号。"""
    manifest = _load_manifest()
    manifest[_INDEX_VERSION_KEY] = version
    _save_manifest(manifest)


def add_paper_chunks(paper_id: str, chunk_ids: list[str]):
    """记录 paper_id → chunk_ids 映射。"""
    manifest = _load_manifest()
    manifest[paper_id] = {
        "chunk_ids": chunk_ids,
    }
    _save_manifest(manifest)


def remove_paper(paper_id: str) -> list[str]:
    """从清单中移除 paper，返回需要删除的 chunk_ids。"""
    manifest = _load_manifest()
    info = manifest.pop(paper_id, {})
    _save_manifest(manifest)
    return info.get("chunk_ids", [])


def list_indexed_paper_ids() -> set[str]:
    """列出所有已索引的 paper_id。"""
    manifest = _load_manifest()
    return {k for k in manifest.keys() if k != _INDEX_VERSION_KEY}


def get_paper_chunk_ids(paper_id: str) -> list[str]:
    """获取指定 paper 的所有 chunk_ids。"""
    manifest = _load_manifest()
    return manifest.get(paper_id, {}).get("chunk_ids", [])
