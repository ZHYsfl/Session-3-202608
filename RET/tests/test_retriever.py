"""Retriever smoke tests."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 先用临时目录替换 ChromaDB 持久化路径，避免测试与真实库互相干扰
import config
config.CHROMA_PERSIST_DIR = tempfile.mkdtemp(prefix="ret_test_chroma_")

from models import Metadata
from retriever import build_bm25_index, hybrid_retrieve, invalidate_bm25_cache


def _make_meta(paper_id: str, title: str, text: str, year: str = "2024", source_type: str = "Test") -> Metadata:
    return Metadata(
        id=paper_id,
        title=title,
        year=year,
        source_type=source_type,
        text_summary=text[:100],
        text=text,
    )


def test_bm25_and_hybrid_retrieve():
    invalidate_bm25_cache()
    metas = [
        _make_meta("p1", "Diabetes management guidelines", "Diabetes management is important for patients."),
        _make_meta("p2", "Hypertension treatment", "Hypertension treatment includes lifestyle changes."),
        _make_meta("p3", "Diabetes and cardiovascular risk", "Diabetes increases cardiovascular risk significantly."),
    ]
    build_bm25_index(metas)
    results = hybrid_retrieve("diabetes management", metadata_list=metas, top_k=5)
    assert results
    ids = [r["paper_id"] for r in results]
    assert "p1" in ids


def test_year_filter():
    invalidate_bm25_cache()
    metas = [
        _make_meta("p1", "Old diabetes paper", "Diabetes text", year="2010"),
        _make_meta("p2", "Recent diabetes paper", "Diabetes text", year="2023"),
    ]
    build_bm25_index(metas)
    results = hybrid_retrieve("diabetes", metadata_list=metas, top_k=5, year_filter=2020)
    assert all(int(r["year"]) >= 2020 for r in results if r.get("year"))


def test_source_type_filter():
    invalidate_bm25_cache()
    metas = [
        _make_meta("p1", "Paper in Nature", "Text", source_type="Nature"),
        _make_meta("p2", "Paper in JAMA", "Text", source_type="JAMA"),
    ]
    build_bm25_index(metas)
    results = hybrid_retrieve("paper", metadata_list=metas, top_k=5, source_type_filter="JAMA")
    assert all(r.get("source_type") == "JAMA" for r in results)
