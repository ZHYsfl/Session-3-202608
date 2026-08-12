# Metadata Coarse Ranking

对去重后的论文 `Metadata` 列表做粗排序：按用户问题相关度返回 top-k。

## Install

```bash
pip install -r requirements.txt
```

## Python API

```python
from coarse_rank import coarse_rank, load_metadatas, dump_metadatas
from metadata import Metadata

# A) 已有 Metadata 对象
results = coarse_rank(metadatas, query="Does metformin improve outcomes?", k1=5)

# B) 从 JSON / JSONL / list[dict] 读入
metas = load_metadatas("papers.json")
results = coarse_rank(metas, query="...", k1=10)
dump_metadatas(results, "ranked.json")

# C) 一包请求 dict
from coarse_rank import coarse_rank_from_request
results = coarse_rank_from_request({
    "query": "...",
    "k1": 5,
    "metadatas": [{"id": "1", "title": "...", "text_summary": "..."}],
})
```

`Metadata` 字段：`title`, `year`, `source_type`, `id`, `text_summary`, `text`,
`image_summary` (`list[str]|None`), `image_base_64`, `evidence_level`, `last_retrieved_at`。

也可用 `Metadata.from_dict(d)` / `m.to_dict()` 做序列化。

## CLI

```bash
# 一包请求（推荐）
python cli.py --request examples/sample_request.json --output ranked.json

# 打印到 stdout
python cli.py --request examples/sample_request.json --output -

# 分离 input + query
python cli.py --input papers.json --query "What are the adverse effects of statins?" --k1 5 --output ranked.json

# 调试分数分解
python cli.py --request examples/sample_request.json --debug --output -
```

JSON 输入支持：

- 数组：`[ {...}, {...} ]`
- 对象：`{ "metadatas": [ ... ] }`（也认 `papers` / `items` / `data`）
- JSONL：每行一个对象

请求文件格式：

```json
{
  "query": "user question",
  "k1": 10,
  "metadatas": [ { "id": "...", "title": "..." } ]
}
```

## Tests

用例数据在独立文件中：

- `tests/cases_round1.py` — case 1–14
- `tests/cases_round2.py` — case 15–25（医学）

```bash
# 全量（遇失败即停）
python tests/run_tests.py

# 全量软运行（汇总失败）
python tests/run_tests.py soft

# 医学 Round 2
python tests/run_tests.py medical
```

兼容入口：`python testexamples.py` / `python testexamples.py medical`。

## Layout

| 文件 | 作用 |
|------|------|
| `coarse_rank.py` | 主 API |
| `cli.py` | 命令行 |
| `io_util.py` | JSON/JSONL 读写 |
| `metadata.py` | 数据结构 |
| `retrievers.py` / `medical_concepts.py` / `tokenizer.py` | 打分 |
| `tests/` | 用例与 runner |
| `examples/sample_request.json` | CLI 示例 |
