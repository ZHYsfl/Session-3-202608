"""Backward-compatible entry: cases and runners live under tests/. """

from __future__ import annotations

from tests.run_tests import (  # noqa: F401
    ALL_TEST_CASES,
    MEDICAL_ROUND2_CASES,
    _assert_case,
    _print_scores,
    _result_ids,
    run_all_tests,
    run_medical_round2,
)

# Re-export round case lists for older scripts
from tests.cases_round1 import ROUND1_CASES  # noqa: F401
from tests.cases_round2 import ROUND2_CASES  # noqa: F401

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in {"medical", "round2", "15-25"}:
        run_medical_round2(debug=True)
    else:
        run_all_tests()
