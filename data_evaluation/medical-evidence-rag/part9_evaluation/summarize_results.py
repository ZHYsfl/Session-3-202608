"""Regenerate summary.json from an existing results.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_batch import summarize


parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path)
parser.add_argument("--output", type=Path, default=Path("summary.json"))
args = parser.parse_args()

summary = summarize(args.results)
args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
