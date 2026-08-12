"""
BGE-M3 tokenizer wrapper.

Loads the real tokenizer from HuggingFace (`BAAI/bge-m3/tokenizer.json`)
using `huggingface_hub.hf_hub_download` and `tokenizers.Tokenizer`.

If the download or initialization fails, falls back to a simple character-count
estimator so the rest of the pipeline keeps working offline.
"""
from __future__ import annotations

import os
from typing import Any

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


CACHE_DIR = os.path.join(os.path.dirname(__file__), "hf_cache")


class _FallbackTokenizer:
    """Minimal fallback when the real tokenizer cannot be loaded."""

    @staticmethod
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return max(len(text.split()), len(text) // 2)

    @staticmethod
    def encode(text: str) -> list[str]:
        if not text:
            return []
        return text.split()

    @staticmethod
    def decode(tokens: list[str]) -> str:
        if not tokens:
            return ""
        return " ".join(str(t) for t in tokens)


class _TokenizerWrapper:
    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        if text is None:
            return 0
        return len(self._tokenizer.encode(str(text)).ids)

    def encode(self, text: str) -> list[str]:
        if text is None:
            return []
        encoded = self._tokenizer.encode(str(text))
        return encoded.tokens

    def decode(self, tokens: list[str]) -> str:
        if not tokens:
            return ""
        # Tokenizer.decode expects token ids or a single string; tokens() returns
        # strings like "▁hello".  We can join them directly as a best-effort decode.
        return "".join(tokens).replace("▁", " ").strip()


def _load_bge_tokenizer() -> Any:
    try:
        tokenizer_path = hf_hub_download(
            repo_id="BAAI/bge-m3",
            filename="tokenizer.json",
            cache_dir=CACHE_DIR,
            local_files_only=False,
        )
        tokenizer = Tokenizer.from_file(tokenizer_path)
        return _TokenizerWrapper(tokenizer)
    except Exception as exc:
        print(f"[bge_tokenizer] Failed to load BAAI/bge-m3 tokenizer: {exc}")
        print("[bge_tokenizer] Falling back to simple token estimator.")
        return _FallbackTokenizer()


_tokenizer = _load_bge_tokenizer()


def count_tokens(text: str) -> int:
    """Return the number of tokens for the given text."""
    return _tokenizer.count_tokens(text)


def encode(text: str) -> list[str]:
    """Encode text into a list of token strings."""
    return _tokenizer.encode(text)


def decode(tokens: list[str]) -> str:
    """Decode a list of token strings back into text."""
    return _tokenizer.decode(tokens)


if __name__ == "__main__":
    sample = "Hello world, this is a test sentence for BGE-M3 tokenizer."
    print(f"count: {count_tokens(sample)}")
    print(f"encode: {encode(sample)}")
    print(f"decode: {decode(encode(sample))}")
