"""Tests for Resilient Infrastructure Max-Flow Min-Cut Benchmark Suite."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from experiments.domain_discovery.graph_and_combinatorial.benchmark import (
    ResilientInfrastructureOptimizer,
    run_synthetic_benchmark,
)


@pytest.fixture
def catalog_concepts() -> list[dict]:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    cat_file = repo_root / "data" / "domains" / "graph_and_combinatorial.json"
    assert cat_file.exists(), f"Catalog not found at {cat_file}"
    with open(cat_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_catalog_integrity(catalog_concepts: list[dict]) -> None:
    """Verify that graph_and_combinatorial domain contains all 42 valid concepts."""
    assert len(catalog_concepts) == 42
    names = {c["name"] for c in catalog_concepts}
    assert len(names) == 42

    required_keys = {"name", "param_types", "return_type", "summary", "pseudocode", "invariants", "pitfalls", "source_code", "domain"}
    for c in catalog_concepts:
        assert required_keys.issubset(c.keys()), f"Missing keys in {c.get('name')}"
        assert c["domain"] == "general"
        assert len(c["invariants"]) >= 2
        assert len(c["pitfalls"]) >= 1


def test_dijkstra_latencies() -> None:
    """Verify Dijkstra computes exact shortest path latencies satisfying the triangle inequality."""
    opt = ResilientInfrastructureOptimizer(n_nodes=4)
    opt.add_edge(0, 1, latency=4.0)
    opt.add_edge(0, 2, latency=1.0)
    opt.add_edge(2, 1, latency=2.0)
    opt.add_edge(1, 3, latency=1.0)
    opt.add_edge(2, 3, latency=5.0)

    res = opt.evaluate_resilience(source=0, sink=3)
    dists = res["min_latencies_from_source"]

    # 0->2 (1.0), 2->1 (2.0) => dist[1] = 3.0 (shorter than direct 4.0)
    assert dists[0] == 0.0
    assert dists[1] == 3.0
    assert dists[2] == 1.0
    assert dists[3] == 4.0  # 0->2->1->3: 1 + 2 + 1 = 4.0


def test_dinic_max_flow_bottleneck() -> None:
    """Verify Dinic max-flow reaches exact min-cut capacity on diamond bottleneck network."""
    opt = ResilientInfrastructureOptimizer(n_nodes=4)
    # Source: 0, Sink: 3
    opt.add_edge(0, 1, capacity=10.0)
    opt.add_edge(0, 2, capacity=10.0)
    opt.add_edge(1, 2, capacity=2.0)
    opt.add_edge(1, 3, capacity=4.0)
    opt.add_edge(2, 3, capacity=9.0)

    res = opt.evaluate_resilience(source=0, sink=3)
    # Max flow should be 4 (via 1->3) + 9 (via 2->3 with 2 coming from 0 and 1) = 13.0
    assert res["max_flow_throughput"] == 13.0


def test_articulation_points_cut_vertices() -> None:
    """Verify Tarjan's algorithm flags bridge nodes as critical single-points-of-failure."""
    # Graph: 0-1, 1-2, 2-3 (line of 4 nodes). Cut vertices are 1 and 2.
    opt = ResilientInfrastructureOptimizer(n_nodes=4)
    opt.add_edge(0, 1, bidirectional=True)
    opt.add_edge(1, 2, bidirectional=True)
    opt.add_edge(2, 3, bidirectional=True)

    res = opt.evaluate_resilience(source=0, sink=3)
    cut_nodes = res["critical_cut_vertices"]
    assert 1 in cut_nodes
    assert 2 in cut_nodes


def test_dag_topological_sort() -> None:
    """Verify Kahn's algorithm produces valid dependency ordering where prerequisites precede dependents."""
    opt = ResilientInfrastructureOptimizer(n_nodes=5)
    deps = [
        [1, 2],  # 0 precedes 1 and 2
        [3],     # 1 precedes 3
        [3],     # 2 precedes 3
        [4],     # 3 precedes 4
        [],      # 4 is terminal
    ]
    order = opt.schedule_service_mesh_stages(deps)
    assert len(order) == 5
    pos = {node: i for i, node in enumerate(order)}
    for u, targets in enumerate(deps):
        for v in targets:
            assert pos[u] < pos[v], f"Node {u} should precede {v} in ordering {order}"


def test_knapsack_hardening() -> None:
    """Verify 0/1 knapsack dynamic programming computes globally optimal value."""
    opt = ResilientInfrastructureOptimizer(n_nodes=1)
    costs = [10, 20, 30]
    benefits = [60, 100, 120]
    budget = 50

    # Optimal choice: cost 20 + 30 = 50 -> benefit 100 + 120 = 220
    gain = opt.allocate_hardening_budget(costs, benefits, budget)
    assert gain == 220


def test_run_synthetic_benchmark() -> None:
    """End-to-end integration test of the synthetic infrastructure benchmark."""
    bench_result = run_synthetic_benchmark()
    assert "resilience" in bench_result
    assert "deployment_order" in bench_result
    assert "optimal_hardening_gain" in bench_result

    res = bench_result["resilience"]
    assert res["max_flow_throughput"] == 30.0
    assert 3 in res["critical_cut_vertices"]
    assert 6 in res["critical_cut_vertices"]
    assert len(bench_result["deployment_order"]) == 5
    assert bench_result["optimal_hardening_gain"] == 115
