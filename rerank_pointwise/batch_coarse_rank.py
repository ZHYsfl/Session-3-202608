#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate coarse-ranked retrieval_results.json with retriever_015."""

import argparse
import glob
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True,
                        help="path to questions_500.json")
    parser.add_argument("--data", required=True,
                        help="directory containing RET/data/*.json")
    parser.add_argument("--output", default="retrieval_results.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N questions")
    parser.add_argument("--k1", type=int, default=20,
                        help="top-k per question returned by coarse rank")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "retriever_015"),
        os.path.join(os.path.dirname(here), "retriever_015"),
    ]
    retriever_dir = next((c for c in candidates if os.path.isdir(c)), candidates[0])
    sys.path.insert(0, retriever_dir)
    from metadata import Metadata
    from coarse_rank import coarse_rank

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    if args.limit:
        questions = questions[: args.limit]

    metadatas = []
    for path in sorted(glob.glob(os.path.join(args.data, "*.json"))):
        with open(path, encoding="utf-8") as f:
            metadatas.append(Metadata.from_dict(json.load(f)))
    print("loaded %d metadatas" % len(metadatas), flush=True)

    results = {}
    t0 = time.time()
    for i, question in enumerate(questions, start=1):
        qid = question["id"]
        query = question["question"]
        top = coarse_rank(metadatas, query, args.k1)
        results[qid] = [m.to_dict() for m in top]
        print(
            "[%d/%d] %s: %d docs, %.1fs elapsed"
            % (i, len(questions), qid, len(top), time.time() - t0),
            flush=True,
        )

    output = {
        "version": "0.1.0",
        "description": "coarse ranked retrieval results (retriever_015)",
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("saved %s" % args.output, flush=True)


if __name__ == "__main__":
    main()
