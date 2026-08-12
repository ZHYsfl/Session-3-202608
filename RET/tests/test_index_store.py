"""Index store smoke tests."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import index_store as store

# 使用临时 manifest 文件，避免污染生产索引清单
store.MANIFEST_PATH = tempfile.mktemp(prefix="index_manifest_test_", suffix=".json")


def test_version_roundtrip():
    store.clear_manifest()
    assert store.get_index_version() is None
    store.set_index_version("v1.0")
    assert store.get_index_version() == "v1.0"
    store.set_index_version("BAAI/bge-m3:v2.0")
    assert store.get_index_version() == "BAAI/bge-m3:v2.0"


def test_paper_chunks_roundtrip():
    store.clear_manifest()
    store.add_paper_chunks("pmid_123", ["pmid_123_text_0", "pmid_123_title"])
    assert store.is_paper_indexed("pmid_123")
    assert store.list_indexed_paper_ids() == {"pmid_123"}
    assert store.get_paper_chunk_ids("pmid_123") == ["pmid_123_text_0", "pmid_123_title"]


def test_version_key_not_in_paper_ids():
    store.clear_manifest()
    store.set_index_version("v1")
    store.add_paper_chunks("p1", ["p1_t"])
    assert store.list_indexed_paper_ids() == {"p1"}
