"""Soft-run all tests and print medical score decompositions for key cases."""
from __future__ import annotations

from coarse_rank import coarse_rank
from retrievers import rank_indices, score_all
from tests.run_tests import (
    ALL_TEST_CASES,
    MEDICAL_ROUND2_CASES,
    _assert_case,
    _result_ids,
)


def run_group(cases: list, label: str) -> tuple[int, list]:
    passed = 0
    failed: list = []
    for case in cases:
        name = case["name"]
        ids = _result_ids(coarse_rank(case["metadatas"], case["query"], case["k1"]))
        try:
            _assert_case(case)
            print(f"PASS {name}: {ids}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {ids}")
            print(f"  {e}")
            failed.append((name, str(e), ids))
    print(f"--- {label}: {passed}/{len(cases)} ---\n")
    return passed, failed


def dump_scores(names: list[str]) -> None:
    by_name = {c["name"]: c for c in ALL_TEST_CASES}
    for name in names:
        case = by_name[name]
        breakdowns = score_all(case["metadatas"], case["query"])
        ranked = rank_indices(breakdowns)
        by_idx = {b.index: b for b in breakdowns}
        print(f"===== {name} =====")
        print(f"query: {case['query']}")
        for rank, idx in enumerate(ranked, start=1):
            b = by_idx[idx]
            print(
                f"  {rank}. {b.id} final={b.final:.4f} "
                f"relevance={b.relevance:.4f} topic={b.topic:.4f} "
                f"aspect={b.aspect:.4f} joint={b.joint:.4f} constr={b.constraint:.4f} "
                f"img={b.image_score:.3f} quality={b.quality:.4f} "
                f"(ev={b.evidence:.2f} src={b.source_bonus:.2f})"
            )
        print()


if __name__ == "__main__":
    c14 = ALL_TEST_CASES[:14]
    p1, f1 = run_group(c14, "Case 1-14")
    p2, f2 = run_group(MEDICAL_ROUND2_CASES, "Case 15-25")
    total = len(c14) + len(MEDICAL_ROUND2_CASES)
    print(f"=== Overall PASS: {p1 + p2}/{total} ===")
    print("\n=== Score analysis (key medical cases) ===\n")
    dump_scores(
        [
            "case_16_drug_adverse_effects",
            "case_17_risk_factor",
            "case_19_medical_abbreviation",
            "case_20_medical_synonym",
            "case_21_population_specific",
            "case_24_medical_image_summary",
            "case_25_multi_factor_medical_query",
        ]
    )
