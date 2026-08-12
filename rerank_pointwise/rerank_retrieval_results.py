#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rerank retrieval_results.json per question with local Ollama listwise."""

import argparse
import json
import os
import sys
import time


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    import rerank_listwise as rl

    for key, value in rl.DEFAULTS.items():
        cfg.setdefault(key, value)
    if not isinstance(cfg.get("fields"), dict):
        cfg["fields"] = dict(rl.DEFAULT_FIELDS)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True,
                        help="path to questions_500.json")
    parser.add_argument("--input", required=True,
                        help="coarse retrieval_results.json")
    parser.add_argument("--config", default="rerank_config.ollama.json")
    parser.add_argument("--output", default="reranked_retrieval_results.json")
    parser.add_argument("--details", default="work/rerank_details.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="only rerank the first N questions")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    import rerank_listwise as rl

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    query_map = {q["id"]: q["question"] for q in questions}

    with open(args.input, encoding="utf-8") as f:
        coarse = json.load(f)
    results = coarse["results"]

    cfg = load_config(args.config)
    order = list(results.keys())
    if args.limit:
        order = order[: args.limit]

    reranked = {}
    details = {}
    t0 = time.time()
    for i, qid in enumerate(order, start=1):
        items = results[qid]
        query = query_map.get(qid) or cfg.get("default_query") or ""
        if not query:
            raise SystemExit("cannot determine query for %s" % qid)
        wrapped = [(item, idx) for idx, item in enumerate(items)]
        ordered, reasons = rl.listwise_sort(wrapped, query, cfg)
        reranked[qid] = [item for item, _ in ordered]
        details[qid] = {
            "method": "listwise",
            "query": query,
            "reasons": reasons,
        }
        print(
            "[%d/%d] %s reranked %d docs, %.1fs elapsed"
            % (i, len(order), qid, len(reranked[qid]), time.time() - t0),
            flush=True,
        )

    output = dict(coarse)
    output["results"] = reranked
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("saved %s" % args.output, flush=True)

    os.makedirs(os.path.dirname(args.details) or ".", exist_ok=True)
    with open(args.details, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print("saved %s" % args.details, flush=True)


if __name__ == "__main__":
    main()
