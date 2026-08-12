#!/usr/bin/env python3
"""Command-line interface for Metadata coarse ranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python cli.py` from repo root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coarse_rank import coarse_rank, coarse_rank_from_request  # noqa: E402
from io_util import dump_metadatas, load_metadatas, load_rank_request  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Coarse-rank paper Metadata by query (BM25 + keyword aspects + quality).",
    )
    p.add_argument(
        "--request",
        help="Single JSON file with {query, k1, metadatas}. Overrides --input/--query/--k1.",
    )
    p.add_argument(
        "--input",
        "-i",
        help="Metadata JSON/JSONL path, or '-' for stdin JSON array.",
    )
    p.add_argument("--query", "-q", help="User query string.")
    p.add_argument("--k1", "-k", type=int, default=None, help="Return top-k1 (default: 10, or value in --request).")
    p.add_argument(
        "--output",
        "-o",
        help="Write ranked Metadata JSON array to path, or '-' for stdout.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print score decomposition to stderr.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.request:
        metas, query, k1 = load_rank_request(args.request)
        if args.query:
            query = args.query
        if args.k1 is not None:
            k1 = args.k1
        results = coarse_rank(metas, query, k1, debug=args.debug)
    else:
        if not args.input:
            build_parser().error("either --request or --input is required")
        if not args.query:
            build_parser().error("--query is required when using --input")
        metas = load_metadatas(args.input)
        k1 = 10 if args.k1 is None else args.k1
        results = coarse_rank(metas, args.query, k1, debug=args.debug)

    if args.output:
        dump_metadatas(results, args.output)
    else:
        for i, m in enumerate(results, start=1):
            print(f"{i}. id={m.id} title={m.title}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
