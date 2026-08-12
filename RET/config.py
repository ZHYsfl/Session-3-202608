"""
全局配置
- 优先从 .env 文件加载环境变量（override=True 允许 .env 覆盖系统环境变量）
- 所有敏感信息（API key）必须从 .env / 环境变量读取，config.py 不再硬编码
- HTTP_PROXY 默认走 127.0.0.1:7890
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载 .env（优先使用项目根目录或 RET 目录下的 .env）
_dotenv_paths = [
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]
for _dotenv_path in _dotenv_paths:
    if _dotenv_path.exists():
        load_dotenv(dotenv_path=_dotenv_path, override=True)
        break


def _env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


# ─── DeepSeek 官方 API（默认 v4 flash）─────────────────────
DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_CHAT_MODEL = _env("DEEPSEEK_CHAT_MODEL", "deepseek-v4-flash")

# ─── SiliconFlow API（embedding 与备用 chat）─────────────────────
SILICONFLOW_API_KEY = _env("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = _env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

# Embedding 默认复用 SiliconFlow
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(_env("EMBEDDING_DIM", "1024"))
EMBEDDING_BATCH_SIZE = int(_env("EMBEDDING_BATCH_SIZE", "16"))  # API 批大小（保守，避免 TPM 限流）
EMBEDDING_BASE_URL = _env("EMBEDDING_BASE_URL", SILICONFLOW_BASE_URL)
EMBEDDING_API_KEY = _env("EMBEDDING_API_KEY", SILICONFLOW_API_KEY)

# ─── 外部检索 API ────────────────────────────────────────
METASO_API_KEY = _env("METASO_API_KEY")
PUBMED_API_KEY = _env("PUBMED_API_KEY")

# ─── HTTP 代理（梯子）────────────────────────────────────────
HTTP_PROXY = _env("HTTP_PROXY", "http://127.0.0.1:7890")
HTTPS_PROXY = _env("HTTPS_PROXY", HTTP_PROXY)
NO_PROXY = _env("NO_PROXY", "localhost,127.0.0.1")


def _apply_proxy():
    """应用 HTTP 代理到当前进程环境变量。"""
    os.environ.setdefault("HTTP_PROXY", HTTP_PROXY or "")
    os.environ.setdefault("HTTPS_PROXY", HTTPS_PROXY or "")
    os.environ.setdefault("NO_PROXY", NO_PROXY or "")


_apply_proxy()

# ─── ChromaDB ───────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CHROMA_COLLECTION_NAME = _env("CHROMA_COLLECTION_NAME", "papers")

# ─── 文本分块 ───────────────────────────────────────────────
CHUNK_SIZE = int(_env("CHUNK_SIZE", "512"))         # tokens
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "50"))    # tokens
# 已改用真实 BGE-M3 tokenizer（见 bge_tokenizer.py），此处保留编码名仅作兼容
TOKENIZER_ENCODING = _env("TOKENIZER_ENCODING", "cl100k_base")
CHUNKER_VERSION = _env("CHUNKER_VERSION", "v2.0")
INDEX_VERSION = _env("INDEX_VERSION", f"{EMBEDDING_MODEL}:{CHUNKER_VERSION}")

# ─── 检索 ───────────────────────────────────────────────────
DEFAULT_TOP_K = int(_env("DEFAULT_TOP_K", "10"))
RRF_K = int(_env("RRF_K", "60"))

# ─── source_type 清洗映射 ───────────────────────────────────
# 有些 source_type 是截断的句子，需要清洗
SOURCE_TYPE_BLACKLIST = {
    "the global obesity epidemic has relentlessly",
    "obesity reviews an official journal of the international association",
}

# ─── 并发限制（外部 API）────────────────────────────────────
DEEPSEEK_MAX_CONCURRENCY = int(_env("DEEPSEEK_MAX_CONCURRENCY", "12"))
QWEN_MAX_CONCURRENCY = int(_env("QWEN_MAX_CONCURRENCY", "6"))
