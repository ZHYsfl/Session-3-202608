#!/usr/bin/env python3
"""Merge external medical dictionary rows into medical_concepts.json.

Supported input formats:
  JSON list:
    [{"concept": "hypertension", "terms": ["高血压", "血压升高"], "is_disease": true}]

  CSV:
    concept,term,is_disease
    hypertension,高血压,true
    hypertension,血压升高,true
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("records", data.get("data", []))
    if not isinstance(data, list):
        raise ValueError("JSON input must be a list, or an object containing records/data")
    return data


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_terms(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.replace("；", ";").replace("，", ";").replace(",", ";").split(";") if part.strip()]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是", "疾病"}


def load_concepts(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return {"concept_groups": {}, "disease_concepts": [], "scope_conflict_rules": []}


def merge_rows(config: Dict[str, Any], rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    groups = config.setdefault("concept_groups", {})
    diseases = set(config.setdefault("disease_concepts", []))
    for row in rows:
        concept = str(row.get("concept") or row.get("id") or row.get("name") or "").strip()
        if not concept:
            continue
        terms = list(as_terms(row.get("terms") or row.get("term") or row.get("synonyms") or row.get("aliases") or row.get("name")))
        if not terms:
            continue
        existing = list(groups.get(concept, []))
        seen = set(existing)
        for term in terms:
            if term not in seen:
                existing.append(term)
                seen.add(term)
        groups[concept] = existing
        if truthy(row.get("is_disease") or row.get("disease")):
            diseases.add(concept)
    config["disease_concepts"] = sorted(diseases)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge external medical terms into medical_concepts.json.")
    parser.add_argument("--concepts", default=Path(__file__).with_name("medical_concepts.json"), type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--format", choices=("json", "csv"), default=None)
    args = parser.parse_args()

    input_format = args.format or args.input.suffix.lower().lstrip(".")
    rows = read_json(args.input) if input_format == "json" else read_csv(args.input)
    config = merge_rows(load_concepts(args.concepts), rows)
    args.concepts.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"concepts": len(config["concept_groups"]), "diseases": len(config["disease_concepts"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
