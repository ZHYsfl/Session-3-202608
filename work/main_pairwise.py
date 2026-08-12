"""两两方案主程序：演示 k1 个文件 -> 一个接口 -> k2 个文件。

唯一接口：literature_pairwise.pairwise_rank(records, k2) -> list[dict]
  - 传入变量：k1 条记录（list[dict]，每条对应一个 JSON 文件）
  - 返回变量：k2 条记录（list[dict]，按获胜场次降序）

本主程序只做两件外围的事：
  1. 把 in 目录下 k1 个 JSON 文件读成 records 列表；
  2. 把接口返回的 k2 条记录写成 k2 个 JSON 文件到 out 目录。
两两对比逻辑全部在接口函数内部，主程序不碰。

用法：
    python main_pairwise.py --k2 3 --query "metformin diabetes"
    python main_pairwise.py --judge llm --model gpt-4.1-mini   # 线上模型当裁判
    python main_pairwise.py --judge ollama --model qwen2.5:7b  # 本地 Ollama 当裁判
    python main_pairwise.py --workers 4                        # 4 进程并行两两对比
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from literature_pairwise import pairwise_rank


def load_records(in_dir: Path) -> list[dict]:
    """读取目录下所有 .json 文件，拼成 k1 条记录列表。

    每个文件可以是单条记录 dict，也可以是记录列表；
    文件名只用于回写，不影响记录内容。
    """
    records: list[dict] = []
    for path in sorted(in_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            records.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict):
            records.append(data)
        else:
            raise ValueError(f"{path} 不是合法的记录 JSON（应为 dict 或 list）")
    return records


def save_records(records: list[dict], out_dir: Path) -> None:
    """把 k2 条记录写成 k2 个独立 JSON 文件，文件名带名次。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, rec in enumerate(records, start=1):
        name = rec.get("id") or rec.get("title") or f"record_{rank}"
        safe = "".join(c for c in str(name) if c not in r'\/:*?"<>|') or "record"
        path = out_dir / f"{rank:03d}_{safe}.json"
        path.write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  [{rank:>2}] {safe}")


def build_compare(judge: str, model: str, query: str | None) -> Any:
    """按 --judge 构造裁判函数；rule 返回 None，走接口内置规则裁判。"""
    if judge == "rule":
        return None
    if judge == "llm":
        try:
            from openai import OpenAI
            from llm_judge import make_llm_compare, set_compare_query
        except ImportError:
            sys.exit("线上模型需要安装 openai 包：pip install openai，并配置 OPENAI_API_KEY")
        set_compare_query(query)
        return make_llm_compare(OpenAI(), model=model)
    if judge == "ollama":
        from llm_judge_v2 import make_llm_compare
        return make_llm_compare(model=model, query=query)
    raise ValueError(f"未知 judge：{judge}")


def main() -> None:
    parser = argparse.ArgumentParser(description="两两方案：k1 个文件 -> k2 个文件")
    parser.add_argument("--in-dir", default="data/in", help="k1 个 JSON 文件所在目录")
    parser.add_argument("--out-dir", default="data/out_pairwise", help="k2 个 JSON 文件输出目录")
    parser.add_argument("--k2", type=int, default=3, help="输出条数")
    parser.add_argument("--query", default=None, help="检索主题（相关性评分用）")
    parser.add_argument("--judge", choices=["rule", "llm", "ollama"], default="rule",
                        help="裁判方式：rule=内置规则，llm=线上模型，ollama=本地 Ollama")
    parser.add_argument("--model", default=None,
                        help="模型名（llm 默认 gpt-4.1-mini，ollama 默认 qwen2.5:7b）")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行度（默认自动按 CPU 核数；1=串行）")
    args = parser.parse_args()

    records = load_records(Path(args.in_dir))
    print(f"读取到 k1 = {len(records)} 条记录")

    model = args.model or ("qwen2.5:7b" if args.judge == "ollama" else "gpt-4.1-mini")
    compare = build_compare(args.judge, model, args.query)
    print(f"裁判方式：{args.judge}" + (f"（{model}）" if args.judge != "rule" else ""))

    # ---- 唯一接口：传入 k1 条，返回 k2 条 ----
    picked = pairwise_rank(
        records, args.k2, compare=compare, query=args.query, workers=args.workers
    )
    # ------------------------------------------

    print(f"输出 k2 = {len(picked)} 条记录（按获胜场次降序）：")
    save_records(picked, Path(args.out_dir))
    print(f"已写入目录：{args.out_dir}")


if __name__ == "__main__":
    main()
