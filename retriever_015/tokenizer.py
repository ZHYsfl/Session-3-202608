from __future__ import annotations

import re

import jieba

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)

# 英文停用词（粗排用）；保留实词与数字
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
        "of",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "their",
        "how",
        "why",
        "where",
        "as",
        "used",  # 避免 "are used for" 类虚匹配
    }
)


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def tokenize(text: str) -> list[str]:
    """中英混合分词：含中文时用 jieba，否则按空白/标点切分并小写。"""
    if not text or not text.strip():
        return []

    text = text.strip().lower()

    if _has_cjk(text):
        tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
        return [t for t in tokens if _TOKEN_RE.fullmatch(t) or any(c.isalnum() for c in t)]

    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def content_tokens(text: str) -> list[str]:
    """分词、过滤停用词，并做极简词干归一（单复数）。"""
    out: list[str] = []
    for t in tokenize(text):
        if t in _STOPWORDS or len(t) <= 1:
            continue
        # 简单复数归一：transformers -> transformer
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out
