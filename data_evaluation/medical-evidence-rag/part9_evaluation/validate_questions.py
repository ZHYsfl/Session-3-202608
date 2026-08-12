"""Validate the three-track 500-question JSON interface.

Usage:
    python validate_questions.py questions_500.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


EXPECTED_TOTAL = 500
EXPECTED_ALLOCATION = {"track_1": 200, "track_2": 150, "track_3": 150}
REQUIRED_FIELDS = {
    "id",
    "question",
    "track",
    "expected_evidence_type",
    "notes",
    "should_abstain",
}
ALLOWED_EVIDENCE_TYPES = {
    "guideline",
    "systematic_review",
    "RCT",
    "observational",
    "mixed",
    "other",
    "insufficient",
}


def fail(messages: list[str]) -> None:
    for message in messages:
        print(f"ERROR: {message}")
    raise SystemExit(1)


def validate(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions")
    errors: list[str] = []

    if not isinstance(questions, list):
        fail(["questions必须是数组"])

    if len(questions) != EXPECTED_TOTAL:
        errors.append(f"题目数量应为500，实际为{len(questions)}")

    ids: list[str] = []
    tracks: Counter[str] = Counter()
    for index, item in enumerate(questions, start=1):
        location = f"questions[{index - 1}]"
        if not isinstance(item, dict):
            errors.append(f"{location}必须是对象")
            continue

        missing = REQUIRED_FIELDS - item.keys()
        extra = item.keys() - REQUIRED_FIELDS
        if missing:
            errors.append(f"{location}缺少字段: {sorted(missing)}")
        if extra:
            errors.append(f"{location}包含未约定字段: {sorted(extra)}")

        question_id = item.get("id")
        if not isinstance(question_id, str):
            errors.append(f"{location}.id必须是字符串")
        else:
            ids.append(question_id)
            expected_id = f"Q{index:03d}"
            if question_id != expected_id:
                errors.append(f"{location}.id应为{expected_id}，实际为{question_id}")

        question = item.get("question")
        if not isinstance(question, str) or len(question.strip()) < 5:
            errors.append(f"{location}.question不能为空且至少5个字符")

        track = item.get("track")
        if track not in EXPECTED_ALLOCATION:
            errors.append(f"{location}.track无效: {track!r}")
        else:
            tracks[track] += 1

        evidence_type = item.get("expected_evidence_type")
        if evidence_type not in ALLOWED_EVIDENCE_TYPES:
            errors.append(f"{location}.expected_evidence_type无效: {evidence_type!r}")

        if not isinstance(item.get("notes"), str):
            errors.append(f"{location}.notes必须是字符串")
        if not isinstance(item.get("should_abstain"), bool):
            errors.append(f"{location}.should_abstain必须是布尔值")

    if len(ids) != len(set(ids)):
        errors.append("题目id存在重复")

    for track, expected_count in EXPECTED_ALLOCATION.items():
        actual_count = tracks[track]
        if actual_count != expected_count:
            errors.append(f"{track}应有{expected_count}题，实际为{actual_count}题")

    if data.get("total_expected") != EXPECTED_TOTAL:
        errors.append("total_expected必须为500")
    if data.get("allocation") != EXPECTED_ALLOCATION:
        errors.append(f"allocation必须为{EXPECTED_ALLOCATION}")

    if errors:
        fail(errors)

    abstain_count = sum(bool(item["should_abstain"]) for item in questions)
    print("VALID")
    print(f"total={len(questions)}")
    print("allocation=" + json.dumps(dict(tracks), ensure_ascii=False, sort_keys=True))
    print(f"should_abstain={abstain_count}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_questions.py <questions.json>")
        raise SystemExit(2)
    validate(Path(sys.argv[1]))
