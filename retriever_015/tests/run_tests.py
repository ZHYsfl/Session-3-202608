"""Test runner for coarse ranking (cases live in cases_round1/2)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coarse_rank import coarse_rank  # noqa: E402
from metadata import Metadata  # noqa: E402
from tests.cases_round1 import ROUND1_CASES  # noqa: E402
from tests.cases_round2 import ROUND2_CASES  # noqa: E402

ALL_TEST_CASES = ROUND1_CASES + ROUND2_CASES
MEDICAL_ROUND2_CASES = ROUND2_CASES


def _result_ids(results: list[Metadata]) -> list[str | None]:
    return [m.id for m in results]


def _print_scores(case: dict) -> None:
    from retrievers import rank_indices, score_all

    breakdowns = score_all(case["metadatas"], case["query"])
    ranked = rank_indices(breakdowns)
    by_idx = {b.index: b for b in breakdowns}
    print(f"  scores for {case['name']}:")
    for rank, idx in enumerate(ranked, start=1):
        b = by_idx[idx]
        print(
            f"    {rank}. {b.id} final={b.final:.4f} "
            f"rel={b.relevance:.4f} topic={b.topic:.4f} "
            f"aspect={b.aspect:.4f} joint={b.joint:.4f} constr={b.constraint:.4f} "
            f"qual={b.quality:.4f} img={b.image_score:.3f}"
        )


def _assert_case(case: dict) -> None:
    name = case["name"]
    results = coarse_rank(case["metadatas"], case["query"], case["k1"])
    ids = _result_ids(results)
    expect = case.get("expect", {})

    try:
        if "len" in expect:
            assert len(results) == expect["len"], (
                f"{name}: len={len(results)} expected={expect['len']} ids={ids}"
            )

        if "first_id" in expect and results:
            assert results[0].id == expect["first_id"], (
                f"{name}: first={results[0].id} expected={expect['first_id']} ids={ids}"
            )

        for need in expect.get("contains_ids", []):
            assert need in ids, f"{name}: missing id={need} in {ids}"

        for bad in expect.get("excludes_ids", []):
            assert bad not in ids, f"{name}: unexpected id={bad} in {ids}"

        id_pos = {rid: i for i, rid in enumerate(ids)}
        for earlier, later in expect.get("order_before", []):
            assert earlier in id_pos, f"{name}: order_before missing {earlier} in {ids}"
            assert later in id_pos, f"{name}: order_before missing {later} in {ids}"
            assert id_pos[earlier] < id_pos[later], (
                f"{name}: expected {earlier} before {later}, got {ids}"
            )

        for m in results:
            assert m.last_retrieved_at, f"{name}: last_retrieved_at empty for id={m.id}"
            assert m.last_retrieved_at != "2026-08-01", (
                f"{name}: last_retrieved_at not updated for id={m.id}: {m.last_retrieved_at}"
            )
            datetime.fromisoformat(m.last_retrieved_at)
    except AssertionError:
        if case.get("metadatas") and case.get("query"):
            _print_scores(case)
        raise


def run_all_tests(*, soft: bool = False) -> int:
    """Run all cases. soft=True continues after failures and returns failure count."""
    passed = 0
    failed: list[tuple[str, str]] = []
    for case in ALL_TEST_CASES:
        try:
            _assert_case(case)
            ids = _result_ids(coarse_rank(case["metadatas"], case["query"], case["k1"]))
            print(f"PASS {case['name']}: {ids}")
            passed += 1
        except AssertionError as e:
            if not soft:
                raise
            failed.append((case["name"], str(e)))
            print(f"FAIL {case['name']}: {e}")
    print(f"\n{passed}/{len(ALL_TEST_CASES)} passed.")
    if failed:
        print("failures:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
    return len(failed)


def run_medical_round2(*, debug: bool = True) -> int:
    """Round-2 medical tests; continue on failure; return failure count."""
    from retrievers import rank_indices, score_all

    passed = 0
    failed: list[tuple[str, str, list]] = []

    for case in MEDICAL_ROUND2_CASES:
        name = case["name"]
        results = coarse_rank(case["metadatas"], case["query"], case["k1"], debug=False)
        ids = _result_ids(results)

        breakdowns = score_all(case["metadatas"], case["query"])
        ranked = rank_indices(breakdowns)
        by_idx = {b.index: b for b in breakdowns}

        print(f"\n===== {name} =====")
        print(f"query: {case['query']}")
        print(f"ranked ids (top-{case['k1']}): {ids}")
        if debug:
            print("score decomposition (all candidates):")
            for rank, idx in enumerate(ranked, start=1):
                b = by_idx[idx]
                print(
                    f"  {rank}. {b.id} final={b.final:.4f} "
                    f"rel={b.relevance:.4f} (field_rel={b.rel:.4f} topic={b.topic:.4f} "
                    f"aspect={b.aspect:.4f} joint={b.joint:.4f} constr={b.constraint:.4f}) "
                    f"qual={b.quality:.4f} (ev={b.evidence:.4f} src={b.source_bonus:.4f}) "
                    f"fields[t={b.title_score:.3f} s={b.summary_score:.3f} "
                    f"x={b.text_score:.3f} i={b.image_score:.3f}]"
                )

        try:
            _assert_case(case)
            print(f"PASS {name}: {ids}")
            passed += 1
        except AssertionError as e:
            msg = str(e)
            print(f"FAIL {name}: {msg}")
            failed.append((name, msg, ids))

    print("\n========== MEDICAL ROUND2 SUMMARY ==========")
    print(f"passed={passed}/{len(MEDICAL_ROUND2_CASES)}")
    if failed:
        print("failures:")
        for name, msg, ids in failed:
            print(f"  - {name}: {ids}")
            print(f"    {msg}")
    else:
        print("All medical round-2 cases passed.")
    return len(failed)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"medical", "round2", "15-25"}:
        raise SystemExit(run_medical_round2(debug=True))
    soft = len(sys.argv) > 1 and sys.argv[1] == "soft"
    raise SystemExit(run_all_tests(soft=soft))
