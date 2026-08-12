"""
BGE-M3 Embedding — SiliconFlow API 封装
- 主动节流（避免触发 429）
- 自动重试
- 注意：本模块只负责生成向量，不管理 SplitBlock
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from openai import OpenAI

from config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
)

if TYPE_CHECKING:
    pass

_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

# 主动节流: ~30 批次/分钟 ≈ 2s/批次，留有余量避免 TPM 限流
THROTTLE_SEC = 1.0
_last_call = 0.0


def _throttle():
    global _last_call
    now = time.time()
    wait = THROTTLE_SEC - (now - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def embed_batch(texts: list[str], max_retries: int = 5) -> list[list[float]]:
    """批量获取 embedding，主动节流 + 429 重试。空字符串返回零向量。"""
    if not texts:
        return []

    results: list[list[float]] = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        # 记录空串位置
        empty_idx = [j for j, t in enumerate(batch) if not t or not t.strip()]
        clean_batch = [t if t and t.strip() else " " for t in batch]

        batch_vecs: list[list[float]] = []
        for attempt in range(max_retries):
            try:
                _throttle()  # 主动限速
                resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=clean_batch)
                batch_vecs = [list(d.embedding) for d in resp.data]
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RateLimit" in err_str:
                    wait = min(3**attempt + 2, 60)
                    print(f"  ⏳ 限流(429)，等待 {wait}s 重试...", flush=True)
                    time.sleep(wait)
                elif attempt < max_retries - 1:
                    wait = min(2**attempt, 10)
                    print(f"  ⚠️  {e}，等待 {wait}s 重试...", flush=True)
                    time.sleep(wait)
                else:
                    raise

        # 空串置零
        dim = len(batch_vecs[0]) if batch_vecs else 0
        for j in empty_idx:
            batch_vecs[j] = [0.0] * dim
        results.extend(batch_vecs)

    return results


def embed_single(text: str) -> list[float]:
    """单条 embedding"""
    return embed_batch([text])[0]


def embed_query(query: str) -> np.ndarray:
    """查询 embedding → numpy 数组"""
    vec = embed_single(query)
    return np.array(vec, dtype=np.float32)
