"""Run the complete pure-LLM vs RAG vs judge experiment.

The program appends one JSON record per completed question and supports resume.
Failed questions are recorded separately and retried on the next run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from llm_client import ModelAPIError, load_model_configs
from model_pipeline import answer_pure_llm, answer_rag, judge_once


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}第{line_number}行不是合法JSON") from exc
            if record.get("status") == "completed":
                result.add(record["question_id"])
    return result


def stable_seed(question_id: str) -> int:
    digest = hashlib.sha256(question_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def retry_call(
    action: Callable[[], dict[str, Any]], retries: int, base_delay: float
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return action()
        except (ModelAPIError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(base_delay * (2**attempt))
    assert last_error is not None
    raise last_error


def validate_inputs(dataset: dict[str, Any], retrieval: dict[str, Any]) -> None:
    questions = dataset.get("questions")
    if not isinstance(questions, list) or len(questions) != 500:
        raise ValueError("正式运行要求questions数组恰好包含500题")
    question_ids = [item.get("id") for item in questions]
    if len(set(question_ids)) != 500:
        raise ValueError("500题的id必须唯一")
    expected = {"track_1": 200, "track_2": 150, "track_3": 150}
    actual = Counter(item.get("track") for item in questions)
    if dict(actual) != expected:
        raise ValueError(f"赛道数量应为{expected}，实际为{dict(actual)}")
    results = retrieval.get("results")
    if not isinstance(results, dict):
        raise ValueError("检索文件必须包含results对象")
    missing = [question_id for question_id in question_ids if question_id not in results]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"检索结果缺少{len(missing)}题，例如: {preview}")
    invalid = [key for key in question_ids if not isinstance(results[key], list)]
    if invalid:
        raise ValueError(f"每题检索结果必须是list[Metadata]，错误题号: {invalid[:10]}")


def summarize(results_path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if results_path.exists():
        with results_path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
    # If a question was completed more than once, the latest completed record wins.
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("status") == "completed":
            latest[record["question_id"]] = record

    wins = Counter(record["judgment"]["actual_winner"] for record in latest.values())
    rag_wins = wins["rag"]
    pure_wins = wins["pure_llm"]
    ties = wins["tie"]
    decisive = rag_wins + pure_wins
    total = len(latest)

    by_track: dict[str, Counter[str]] = defaultdict(Counter)
    for record in latest.values():
        by_track[record["track"]][record["judgment"]["actual_winner"]] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "completed_questions": total,
        "rag_wins": rag_wins,
        "pure_llm_wins": pure_wins,
        "ties": ties,
        "rag_decisive_win_rate": rag_wins / decisive if decisive else None,
        "rag_overall_win_rate": rag_wins / total if total else None,
        "tie_rate": ties / total if total else None,
        "target": 0.95,
        "target_met": (rag_wins / decisive >= 0.95) if decisive else None,
        "by_track": {track: dict(counts) for track, counts in sorted(by_track.items())},
        "primary_metric_definition": "rag_wins / (rag_wins + pure_llm_wins)",
    }


def run(args: argparse.Namespace) -> None:
    dataset = load_json(args.questions)
    retrieval = load_json(args.retrieval)
    validate_inputs(dataset, retrieval)
    answer_model, judge_model = load_model_configs(str(args.config))
    done = completed_ids(args.results)

    questions = dataset["questions"]
    if args.limit is not None:
        questions = questions[: args.limit]

    for question in questions:
        question_id = question["id"]
        if question_id in done:
            print(f"SKIP {question_id} 已完成")
            continue
        metadata = retrieval["results"][question_id]
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            pure_answer = retry_call(
                lambda: answer_pure_llm(question, answer_model),
                args.retries,
                args.retry_delay,
            )
            rag_answer = retry_call(
                lambda: answer_rag(question, metadata, answer_model),
                args.retries,
                args.retry_delay,
            )
            judgment = retry_call(
                lambda: judge_once(
                    question,
                    metadata,
                    pure_answer,
                    rag_answer,
                    judge_model,
                    seed=stable_seed(question_id),
                ),
                args.retries,
                args.retry_delay,
            )
            record = {
                "status": "completed",
                "question_id": question_id,
                "track": question["track"],
                "question": question,
                "retrieved_metadata": metadata,
                "pure_llm_answer": pure_answer,
                "rag_answer": rag_answer,
                "judgment": judgment,
                "answer_model": answer_model.model,
                "judge_model": judge_model.model,
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(args.results, record)
            print(f"DONE {question_id} winner={judgment['actual_winner']}")
        except Exception as exc:
            error_record = {
                "status": "failed",
                "question_id": question_id,
                "track": question.get("track"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(args.errors, error_record)
            print(f"FAILED {question_id}: {exc}")
        if args.request_interval > 0:
            time.sleep(args.request_interval)

    summary = summarize(args.results)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行500题纯LLM与RAG盲评实验")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results/results.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("results/errors.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/summary.json"))
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--request-interval", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="先试跑前N题")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
