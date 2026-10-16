"""Replicated CRDT State Store Benchmark.

Composes 6 verified OpenCommons concepts from the distributed_systems domain:
1. `vector_clock_increment`: Causal logical timestamp tracking across distributed events.
2. `vector_clock_merge`: Component-wise maximum lattice join of replica version vectors.
3. `crdt_pn_counter_merge`: Commutative, associative, idempotent merge of positive/negative counter tallies.
4. `crdt_lww_element_set_merge`: Last-Write-Wins conflict-free replicated element set reconciliation.
5. `token_bucket_refill_drain`: Leak-free stateful rate limiting for replica ingress requests.
6. `consistent_hash_ring_lookup`: Deterministic monotonic node partitioning via binary ring bisection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable


def _load_domain_functions() -> dict[str, Callable[..., Any]]:
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "distributed_systems.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    env: dict[str, Any] = {"math": math}
    for c in concepts:
        code_obj = compile(c["source_code"], f"<{c['name']}>", "exec")
        exec(code_obj, env)
    return {c["name"]: env[c["name"]] for c in concepts}


_FUNCS = _load_domain_functions()
vector_clock_increment = _FUNCS["vector_clock_increment"]
vector_clock_merge = _FUNCS["vector_clock_merge"]
crdt_pn_counter_merge = _FUNCS["crdt_pn_counter_merge"]
crdt_lww_element_set_merge = _FUNCS["crdt_lww_element_set_merge"]
token_bucket_refill_drain = _FUNCS["token_bucket_refill_drain"]
consistent_hash_ring_lookup = _FUNCS["consistent_hash_ring_lookup"]


class ReplicatedStateNode:
    """A distributed replica node maintaining vector clocks, CRDTs, and rate limiting."""

    def __init__(self, node_id: str, rate: float = 10.0, capacity: float = 20.0):
        self.node_id = node_id
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = 0.0

        # CRDT state
        self.vector_clock: dict[str, int] = {node_id: 0}
        self.p_counter: dict[str, int] = {}
        self.n_counter: dict[str, int] = {}
        self.lww_adds: dict[str, float] = {}
        self.lww_rems: dict[str, float] = {}

    def admit_request(self, now: float, cost: float = 1.0) -> bool:
        """Admit or reject ingress request via token bucket rate limiter."""
        allowed, new_tokens, new_last = token_bucket_refill_drain(
            self.tokens, self.last_refill, now, self.rate, self.capacity, cost
        )
        self.tokens = new_tokens
        self.last_refill = new_last
        return allowed

    def apply_local_event(
        self,
        item_add: str | None = None,
        item_rem: str | None = None,
        inc_amount: int = 0,
        dec_amount: int = 0,
        timestamp: float = 0.0,
    ) -> None:
        """Record a local state mutation: advances vector clock and mutates CRDT state."""
        self.vector_clock = vector_clock_increment(self.vector_clock, self.node_id)
        if inc_amount > 0:
            self.p_counter[self.node_id] = self.p_counter.get(self.node_id, 0) + inc_amount
        if dec_amount > 0:
            self.n_counter[self.node_id] = self.n_counter.get(self.node_id, 0) + dec_amount
        if item_add is not None:
            if item_add not in self.lww_adds or timestamp > self.lww_adds[item_add]:
                self.lww_adds[item_add] = timestamp
        if item_rem is not None:
            if item_rem not in self.lww_rems or timestamp > self.lww_rems[item_rem]:
                self.lww_rems[item_rem] = timestamp

    def get_counter_value(self) -> int:
        """Compute the net scalar sum of the local PN-counter."""
        return sum(self.p_counter.values()) - sum(self.n_counter.values())

    def get_live_elements(self) -> list[str]:
        """Query surviving elements in LWW set."""
        return crdt_lww_element_set_merge(self.lww_adds, self.lww_rems, {}, {})

    def merge_remote_peer(self, peer: ReplicatedStateNode) -> None:
        """Commutatively join state with a remote replica peer."""
        # 1. Merge vector clocks
        self.vector_clock = vector_clock_merge(self.vector_clock, peer.vector_clock)

        # 2. Merge PN counter
        new_p, new_n, _ = crdt_pn_counter_merge(
            self.p_counter, self.n_counter, peer.p_counter, peer.n_counter
        )
        self.p_counter = new_p
        self.n_counter = new_n

        # 3. Merge LWW element set
        for elem, ts in peer.lww_adds.items():
            if elem not in self.lww_adds or ts > self.lww_adds[elem]:
                self.lww_adds[elem] = ts
        for elem, ts in peer.lww_rems.items():
            if elem not in self.lww_rems or ts > self.lww_rems[elem]:
                self.lww_rems[elem] = ts


class ReplicatedCRDTCluster:
    """A consistent-hashed cluster of replicated CRDT state nodes."""

    def __init__(self, node_ids: list[str], vnodes_per_node: int = 3):
        self.nodes: dict[str, ReplicatedStateNode] = {nid: ReplicatedStateNode(nid) for nid in node_ids}
        self.ring_keys: list[int] = []
        self.ring_nodes: list[str] = []

        # Construct hash ring using standard 32-bit integer hashes
        for nid in node_ids:
            for vn in range(vnodes_per_node):
                # Deterministic integer token on ring
                token = (hash(f"{nid}:vnode:{vn}") & 0x7FFFFFFF)
                self.ring_keys.append(token)
                self.ring_nodes.append(nid)

        # Sort ring pairs by token
        paired = sorted(zip(self.ring_keys, self.ring_nodes))
        self.ring_keys = [k for k, _ in paired]
        self.ring_nodes = [n for _, n in paired]

    def route_key(self, key: str) -> ReplicatedStateNode:
        """Route a key to its primary replica node on the consistent hash ring."""
        key_hash = hash(key) & 0x7FFFFFFF
        target_node_id = consistent_hash_ring_lookup(self.ring_keys, self.ring_nodes, key_hash)
        return self.nodes[target_node_id]

    def full_mesh_sync(self) -> None:
        """Execute anti-entropy gossip pass across all replica pairs in the cluster."""
        node_list = list(self.nodes.values())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                node_list[i].merge_remote_peer(node_list[j])
                node_list[j].merge_remote_peer(node_list[i])


def simulate_partitioned_cluster_reconciliation(
    node_ids: list[str],
    partition_a: list[str],
    partition_b: list[str],
    events_a: list[dict[str, Any]],
    events_b: list[dict[str, Any]],
) -> dict[str, Any]:
    """Simulates split-brain mutations across two network partitions and tests convergence.

    1. Applies events_a to nodes in partition_a while isolated from partition_b.
    2. Applies events_b to nodes in partition_b while isolated from partition_a.
    3. Heals partition by executing anti-entropy sync across all nodes.
    4. Asserts and returns final converged state across all nodes.
    """
    cluster = ReplicatedCRDTCluster(node_ids)

    # 1. Mutate partition A
    for ev in events_a:
        nid = ev["node_id"]
        if nid in partition_a:
            cluster.nodes[nid].apply_local_event(
                item_add=ev.get("add"),
                item_rem=ev.get("rem"),
                inc_amount=ev.get("inc", 0),
                dec_amount=ev.get("dec", 0),
                timestamp=ev.get("ts", 0.0),
            )

    # Sync within partition A
    nodes_a = [cluster.nodes[nid] for nid in partition_a]
    for i in range(len(nodes_a)):
        for j in range(i + 1, len(nodes_a)):
            nodes_a[i].merge_remote_peer(nodes_a[j])
            nodes_a[j].merge_remote_peer(nodes_a[i])

    # 2. Mutate partition B
    for ev in events_b:
        nid = ev["node_id"]
        if nid in partition_b:
            cluster.nodes[nid].apply_local_event(
                item_add=ev.get("add"),
                item_rem=ev.get("rem"),
                inc_amount=ev.get("inc", 0),
                dec_amount=ev.get("dec", 0),
                timestamp=ev.get("ts", 0.0),
            )

    # Sync within partition B
    nodes_b = [cluster.nodes[nid] for nid in partition_b]
    for i in range(len(nodes_b)):
        for j in range(i + 1, len(nodes_b)):
            nodes_b[i].merge_remote_peer(nodes_b[j])
            nodes_b[j].merge_remote_peer(nodes_b[i])

    # 3. Heal partition: full mesh anti-entropy reconciliation
    cluster.full_mesh_sync()

    # 4. Check convergence
    first_node = list(cluster.nodes.values())[0]
    expected_counter = first_node.get_counter_value()
    expected_elements = sorted(first_node.get_live_elements())
    expected_clock = dict(first_node.vector_clock)

    converged = True
    for node in cluster.nodes.values():
        if node.get_counter_value() != expected_counter:
            converged = False
        if sorted(node.get_live_elements()) != expected_elements:
            converged = False
        if node.vector_clock != expected_clock:
            converged = False

    return {
        "converged": converged,
        "final_counter": expected_counter,
        "final_elements": expected_elements,
        "final_vector_clock": expected_clock,
    }
