"""Fast smoke tests for the R-006 falsification-experiment helpers.

These run under the repo's standard `pytest tests/` collection and pin the
pure helper logic used by the stress experiment (the experiment entry points
themselves remain long-running scripts invoked via `__main__`).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_EXPERIMENT_PATH = (
    Path(__file__).resolve().parent.parent / "experiments" / "r006_stress_test_falsification.py"
)


def _load_experiment_module():
    spec = importlib.util.spec_from_file_location("r006_experiment", _EXPERIMENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def r006():
    return _load_experiment_module()


def test_extract_ast_numeric_constants_finds_threshold(r006) -> None:
    code = (
        "def f(re):\n"
        "    if re > 2300.0:\n"
        "        return 0.3164 / re ** 0.25\n"
        "    return 64.0 / re\n"
    )
    # Only comparison-bound constants are extracted (the helper's documented
    # purpose: seed probes at guard thresholds). The 64.0 denominator is a
    # BinOp operand, not a Compare bound, so it is intentionally excluded.
    assert r006.extract_ast_numeric_constants(code) == [2300.0]


def test_extract_ast_numeric_constants_handles_negation_and_dedup(r006) -> None:
    code = (
        "def g(x):\n"
        "    if x < -1.5:\n"
        "        return -1.5\n"
        "    if x == 2:\n"
        "        return 2\n"
        "    return 0\n"
    )
    # Deduplicated, sorted ascending; comparison constants only (return values are not Compare nodes).
    assert r006.extract_ast_numeric_constants(code) == [-1.5, 2.0]


def test_extract_ast_numeric_constants_empty_on_no_comparisons(r006) -> None:
    assert r006.extract_ast_numeric_constants("def h(x):\n    return x + 1\n") == []


def test_canonical_friction_factor_threshold_semantics(r006) -> None:
    """The laminar/turbulent guard under audit: mutant must diverge exactly past the threshold."""
    th = 2300.0
    canonical = (lambda re, _th=th: 0.3164 / re**0.25 if re > _th else 64.0 / re)
    mutant = lambda re: 64.0 / re  # noqa: E731 - intentional inline mutant definition

    below, at, above = 100.0, 2300.0, 3000.0
    assert canonical(below) == pytest.approx(mutant(below))
    assert canonical(at) == pytest.approx(mutant(at))  # strict '>' boundary
    assert canonical(above) != pytest.approx(mutant(above))
