"""
全局配置
"""
import os

# ─── SiliconFlow API ────────────────────────────────────────
SILICONFLOW_API_KEY = "sk-esyilubtqzobjaylaraitkyaclmbjgzkijhdkwyzsfzwmtpp"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 16  # API 批处理大小（保守，避免 TPM 限流）

# ─── ChromaDB ───────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CHROMA_COLLECTION_NAME = "papers"

# ─── 文本分块 ───────────────────────────────────────────────
CHUNK_SIZE = 512       # tokens
CHUNK_OVERLAP = 50     # tokens
# tiktoken encoding for BGE-M3: 模型用 XLM-Roberta tokenizer，
# 这里用 cl100k_base 做近似，中文实际倍率约 1.5-2 chars/token
TOKENIZER_ENCODING = "cl100k_base"

# ─── 检索 ───────────────────────────────────────────────────
DEFAULT_TOP_K = 10
RRF_K = 60
# 多路信号的 RRF 不做加权，保持等权

# ─── source_type 清洗映射 ───────────────────────────────────
# 有些 source_type 是截断的句子，需要清洗
SOURCE_TYPE_BLACKLIST = {
    "the global obesity epidemic has relentlessly",
    "obesity reviews an official journal of the international association",
}
