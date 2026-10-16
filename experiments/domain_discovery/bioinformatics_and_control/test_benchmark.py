"""Tests for Metagenomic BWT & Closed-Loop Bioreactor Benchmark Suite."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from experiments.domain_discovery.bioinformatics_and_control.benchmark import (
    MetagenomicPipeline,
    BioreactorThermalController,
    simulate_closed_loop_bioreactor,
)


@pytest.fixture
def catalog_concepts() -> list[dict]:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    cat_file = repo_root / "data" / "domains" / "bioinformatics_and_control.json"
    assert cat_file.exists(), f"Catalog not found at {cat_file}"
    with open(cat_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_catalog_integrity(catalog_concepts: list[dict]) -> None:
    """Verify that bioinformatics_and_control domain contains all 42 valid concepts."""
    assert len(catalog_concepts) == 42
    names = {c["name"] for c in catalog_concepts}
    assert len(names) == 42

    required_keys = {"name", "param_types", "return_type", "summary", "pseudocode", "invariants", "pitfalls", "source_code", "domain"}
    for c in catalog_concepts:
        assert required_keys.issubset(c.keys()), f"Missing keys in {c.get('name')}"
        assert c["domain"] == "general"
        assert len(c["invariants"]) >= 2
        assert len(c["pitfalls"]) >= 1


def test_bwt_roundtrip_inversion() -> None:
    """Verify Burrows-Wheeler Transform achieves exact bit-for-bit lossless string reconstruction."""
    pipe = MetagenomicPipeline()
    test_cases = [
        "BANANA",
        "GATTACA",
        "ACGTGCATGCAT",
        "AAAAAA",
        "CAGTCAGTCAGT",
        "T",
    ]
    for seq in test_cases:
        bwt_str, orig_idx = pipe.compress_read_bwt(seq)
        recovered = pipe.decompress_read_bwt(bwt_str, orig_idx)
        assert recovered == seq, f"Failed roundtrip for {seq}: got {recovered}"


def test_kmer_frequency_spectrum() -> None:
    """Verify k-mer frequency table computes exact sliding window counts."""
    pipe = MetagenomicPipeline()
    seq = "ACGTACGT"
    # 3-mers: ACG (2), CGT (2), GTA (1), TAC (1)
    # Total 3-mers: 8 - 3 + 1 = 6
    kmers = pipe.analyze_read_composition(seq, k=3)
    assert kmers.get("ACG") == 2
    assert kmers.get("CGT") == 2
    assert kmers.get("GTA") == 1
    assert kmers.get("TAC") == 1
    assert sum(kmers.values()) == 6


def test_needleman_wunsch_alignment_score() -> None:
    """Verify Needleman-Wunsch dynamic programming computes globally optimal alignment."""
    pipe = MetagenomicPipeline()
    # Matching sequences
    score_identical, _, _ = pipe.align_to_reference("ATGC", "ATGC", match=2, mismatch=-1, gap=-1)
    assert score_identical == 8  # 4 * 2

    # Divergent sequences
    score_diff, a1, a2 = pipe.align_to_reference("GCATGCG", "GATTACA", match=1, mismatch=-1, gap=-1)
    assert score_diff == 0
    assert len(a1) == len(a2)


def test_closed_loop_bioreactor_temperature_settling() -> None:
    """Verify Kalman-PID controller settles bioreactor temperature within +-0.05 C of target."""
    sim = simulate_closed_loop_bioreactor(n_steps=120, target_temp=37.0, initial_temp=25.0)
    assert sim["settled"] is True
    assert sim["final_error"] <= 0.05
    assert abs(sim["final_temp"] - 37.0) <= 0.05


def test_pid_anti_windup_prevents_saturation() -> None:
    """Verify actuator output is strictly bounded and integral does not wind up indefinitely."""
    controller = BioreactorThermalController(
        target_temp=37.0,
        min_power=-10.0,
        max_power=10.0,
        initial_temp=10.0,  # large initial error of 27 degrees
    )
    # Perform 20 steps with severe persistent error
    for _ in range(20):
        out = controller.step(raw_sensor_measurement=10.0, dt=1.0)
        assert -10.0 <= out["control_power"] <= 10.0
