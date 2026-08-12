# Rerank Agent

独立的论文候选重排序模块。它接收 Retrieval 阶段产生的 `Query + top-k1 Metadata`，通过 Pointwise、Pairwise 或 Sliding-window Listwise LLM Agent 输出 top-k2 完整 metadata。

完整的系统分层、LangGraph 流程、数据安全边界和目录说明见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 关键设计

- 三种方法都通过 LangGraph 编排“LLM 判断 → 全局排序/融合”工作流。
- `top_k1`、`top_k2`、Listwise 的 `window_size` 与 `stride` 均可动态配置。
- LLM 只会看到 `id/title/year/source_type/text_summary/image_summary/evidence_level`。
- `text` 和 `image_base_64` 仅随最终 metadata 返回，绝不会进入 prompt。
- LLM 响应由 Pydantic 严格校验；错误 JSON 会自动重试。
- 默认 `mock` provider 是确定性的离线实现，便于开箱验证与 CI，不代表生产排序质量。

## 安装与运行

建议使用 Python 3.10+：

```bash
cd reranker_agent
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

python main.py --method pointwise
python main.py --method pairwise --top-k1 6 --top-k2 3
python main.py --method listwise --window-size 5 --stride 2
```

默认读取 `data/example_candidates.json`，可用 `--input your_input.json` 替换。CLI 参数 `--query`、`--top-k1`、`--top-k2` 会覆盖输入 JSON 中对应值。

## 项目结构

```text
reranker_agent/
├── main.py                 # CLI 入口和完整 Metadata 回填
├── config.yaml             # LLM 与排序配置
├── requirements.txt        # 项目依赖
├── README.md               # 使用说明
├── ARCHITECTURE.md         # 架构与模块设计
├── schemas/                # Pydantic 数据结构
├── models/                 # 统一 LLM Client
├── reranker/               # 三种 LangGraph 排序方法及融合算法
├── evaluation/             # 检索评估指标
├── data/                   # 示例输入
└── tests/                  # 自动化测试
```

## 输入与输出

输入 JSON：

```json
{
  "query": "用户问题",
  "top_k1": 50,
  "top_k2": 10,
  "candidates": [{"id": "paper_001", "title": "..."}]
}
```

若 metadata 的 `id` 缺失，系统会生成 `generated_paper_0001`；重复 ID 会直接报错，避免 LLM 比较结果歧义。输入可以包含多于 `top_k1` 篇论文，系统只处理前 `top_k1` 篇。

输出 JSON：

```json
{
  "method": "pointwise",
  "top_k2": 1,
  "results": [
    {
      "metadata": {"id": "paper_001", "title": "..."},
      "rank": 1,
      "score": 9.5,
      "reason": "..."
    }
  ]
}
```

## 使用真实 LLM

### OpenAI

将 `config.yaml` 的 `provider` 改为 `openai`，设置模型，并在环境中提供 Key：

```powershell
$env:OPENAI_API_KEY="..."
python main.py --method pointwise
```

### Qwen

```yaml
llm:
  provider: qwen
  model: qwen-plus
  api_key_env: QWEN_API_KEY
```

Qwen 默认使用 DashScope 的 OpenAI-compatible 地址。其他兼容服务选择 `compatible`，并配置 `base_url`、`model` 和 `api_key_env`。若服务端不支持 Chat Completions 的 JSON object response format，将 `json_mode` 设为 `false`；客户端仍会解析并严格校验返回 JSON。

## 方法说明

- **Pointwise**：每篇调用一次，分数范围为 0–10，稳定降序。
- **Pairwise**：生成全部 `k1*(k1-1)/2` 个 pair，以 win count 得到全局顺序，成本为 O(k1²)。
- **Listwise**：确保 `0 < stride <= window_size`，生成覆盖所有候选的滑动窗口，再使用 RRF（或配置为 Borda）融合。融合分数归一化到 0–10。

同分时保留 Retrieval 输入顺序，保证结果可复现。

## Python API

```python
from models.llm_client import build_llm_client
from reranker import PointwiseReranker
from schemas.metadata import Metadata

client = build_llm_client({"provider": "mock"})
reranker = PointwiseReranker(client)
results = reranker.rerank("network intrusion detection", [Metadata(id="p1")], 1)
```

统一的 reranker 返回值为 `[{"id", "rank", "score", "reason"}]`；CLI 再将原始 metadata 合并进最终输出。

## Evaluation

```python
from evaluation import RankingEvaluator

scores = RankingEvaluator().evaluate(
    predicted=["p2", "p1", "p3"],
    ground_truth=[
        {"paper_id": "p1", "relevance": 3},
        {"paper_id": "p2", "relevance": 1},
    ],
    ks=[1, 3],
)
```

提供 Precision@K、Recall@K、单查询 reciprocal rank（多查询取均值即 MRR）与 graded NDCG@K。相关性 `relevance > 0` 视为 Precision/Recall/MRR 的相关文档。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖三种工作流、滑动窗口覆盖、四项评估指标，以及 `text/image_base_64` 不进入 prompt 的安全约束。
