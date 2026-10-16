"""Pytest test suite for Distributed Systems Replicated CRDT State Store Benchmark."""

import json
from pathlib import Path
import sys
import pytest

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from experiments.domain_discovery.distributed_systems.benchmark import (
    ReplicatedCRDTCluster,
    ReplicatedStateNode,
    simulate_partitioned_cluster_reconciliation,
    vector_clock_increment,
    vector_clock_merge,
    crdt_pn_counter_merge,
    crdt_lww_element_set_merge,
    token_bucket_refill_drain,
    consistent_hash_ring_lookup,
)


def test_consistent_hash_routing():
    """Verify deterministic routing of keys to cluster nodes on hash ring."""
    cluster = ReplicatedCRDTCluster(["node-1", "node-2", "node-3"], vnodes_per_node=5)
    assert len(cluster.ring_keys) == 15
    assert len(cluster.ring_nodes) == 15

    # Routing is deterministic: same key always maps to same replica
    node_a = cluster.route_key("user:1001:profile")
    node_b = cluster.route_key("user:1001:profile")
    assert node_a.node_id == node_b.node_id

    # Multiple distinct keys map to valid nodes in the cluster
    mapped_nodes = {cluster.route_key(f"doc:{i}").node_id for i in range(30)}
    assert mapped_nodes.issubset({"node-1", "node-2", "node-3"})
    assert len(mapped_nodes) > 1, "Keys should distribute across multiple nodes"


def test_token_bucket_rate_limiting():
    """Verify ingress rate limiting behavior with burst capacity and time refill."""
    node = ReplicatedStateNode("replica-0", rate=5.0, capacity=10.0)

    # 1. Burst up to capacity (10 tokens at t=0.0)
    for _ in range(10):
        assert node.admit_request(now=0.0, cost=1.0) is True

    # 11th request at t=0.0 should be rejected (bucket drained)
    assert node.admit_request(now=0.0, cost=1.0) is False

    # 2. Advance time by 1.0 second: refills 5 tokens
    assert node.admit_request(now=1.0, cost=1.0) is True
    assert node.admit_request(now=1.0, cost=1.0) is True
    assert node.admit_request(now=1.0, cost=1.0) is True
    assert node.admit_request(now=1.0, cost=1.0) is True
    # 5th token in second 1.0
    assert node.admit_request(now=1.0, cost=1.0) is True
    # 6th should fail
    assert node.admit_request(now=1.0, cost=1.0) is False


def test_replicated_crdt_partition_reconciliation():
    """Verify that split-brain network partition mutations converge to identical state."""
    node_ids = ["node-A", "node-B", "node-C", "node-D"]
    partition_a = ["node-A", "node-B"]
    partition_b = ["node-C", "node-D"]

    events_a = [
        {"node_id": "node-A", "add": "doc_alpha", "ts": 10.0, "inc": 5},
        {"node_id": "node-B", "add": "doc_beta", "ts": 12.0, "inc": 10},
        {"node_id": "node-A", "rem": "doc_gamma", "ts": 15.0, "dec": 2},
    ]

    events_b = [
        {"node_id": "node-C", "add": "doc_gamma", "ts": 11.0, "inc": 8},  # Added at 11, but removed in A at 15
        {"node_id": "node-D", "add": "doc_delta", "ts": 14.0, "inc": 3},
        {"node_id": "node-C", "rem": "doc_alpha", "ts": 8.0, "dec": 1},   # Removed at 8, but added in A at 10
    ]

    result = simulate_partitioned_cluster_reconciliation(
        node_ids=node_ids,
        partition_a=partition_a,
        partition_b=partition_b,
        events_a=events_a,
        events_b=events_b,
    )

    assert result["converged"] is True

    # Net counter: increments = (5 + 10) + (8 + 3) = 26; decrements = 2 + 1 = 3; net = 23
    assert result["final_counter"] == 23

    # LWW set resolution:
    # doc_alpha: added @ 10, removed @ 8 -> LIVE
    # doc_beta: added @ 12 -> LIVE
    # doc_gamma: added @ 11, removed @ 15 -> REMOVED
    # doc_delta: added @ 14 -> LIVE
    expected_live = ["doc_alpha", "doc_beta", "doc_delta"]
    assert sorted(result["final_elements"]) == expected_live

    # Vector clock reflects mutations across all 4 nodes
    assert result["final_vector_clock"]["node-A"] >= 2
    assert result["final_vector_clock"]["node-B"] >= 1
    assert result["final_vector_clock"]["node-C"] >= 2
    assert result["final_vector_clock"]["node-D"] >= 1


def test_vector_clock_causality():
    """Verify vector clock component-wise lattice merge."""
    vc1 = {"n1": 3, "n2": 1, "n3": 0}
    vc2 = {"n1": 2, "n2": 4, "n4": 2}
    merged = vector_clock_merge(vc1, vc2)
    assert merged == {"n1": 3, "n2": 4, "n3": 0, "n4": 2}


def test_composed_concepts_registered_in_catalog():
    """Verify that all 6 composed building blocks exist in distributed_systems.json."""
    cat_path = repo_root / "data" / "domains" / "distributed_systems.json"
    assert cat_path.is_file()
    with open(cat_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    catalog_names = {c["name"] for c in concepts}
    required = {
        "vector_clock_increment",
        "vector_clock_merge",
        "crdt_pn_counter_merge",
        "crdt_lww_element_set_merge",
        "token_bucket_refill_drain",
        "consistent_hash_ring_lookup",
    }
    assert required.issubset(catalog_names), f"Missing concepts: {required - catalog_names}"
