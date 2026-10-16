"""Resilient Infrastructure Max-Flow Min-Cut & Topology Hardening Benchmark.

Composes 6 verified OpenCommons concepts from the graph_and_combinatorial domain:
1. `dijkstra_shortest_path_costs`: Minimum latency propagation from network root.
2. `tarjan_strongly_connected_components`: Strong-connectivity failure domain partitioning.
3. `articulation_points_cut_vertices`: Single-point-of-failure vulnerability detection.
4. `dinic_level_graph_max_flow`: Ingress-to-egress maximum throughput evaluation.
5. `topological_sort_kahn_indegree`: Deterministic DAG service deployment sequencing.
6. `knapsack_01_dp_optimal_value`: Optimal security hardening under fixed capital budget.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable


def _load_domain_functions() -> dict[str, Callable[..., Any]]:
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "graph_and_combinatorial.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    env: dict[str, Any] = {"math": math}
    for c in concepts:
        code_obj = compile(c["source_code"], f"<{c['name']}>", "exec")
        exec(code_obj, env)
    return {c["name"]: env[c["name"]] for c in concepts}


_FUNCS = _load_domain_functions()
dijkstra_shortest_path_costs = _FUNCS["dijkstra_shortest_path_costs"]
tarjan_strongly_connected_components = _FUNCS["tarjan_strongly_connected_components"]
articulation_points_cut_vertices = _FUNCS["articulation_points_cut_vertices"]
dinic_level_graph_max_flow = _FUNCS["dinic_level_graph_max_flow"]
topological_sort_kahn_indegree = _FUNCS["topological_sort_kahn_indegree"]
knapsack_01_dp_optimal_value = _FUNCS["knapsack_01_dp_optimal_value"]


class ResilientInfrastructureOptimizer:
    """Network infrastructure reliability, capacity, and scheduling optimizer."""

    def __init__(self, n_nodes: int, node_labels: list[str] | None = None) -> None:
        self.n_nodes = max(0, int(n_nodes))
        self.labels = node_labels or [f"node_{i}" for i in range(self.n_nodes)]
        self.adj_unweighted: list[list[int]] = [[] for _ in range(self.n_nodes)]
        self.adj_weighted: list[list[tuple[int, float]]] = [[] for _ in range(self.n_nodes)]
        self.capacity_matrix: list[list[float]] = [[0.0 for _ in range(self.n_nodes)] for _ in range(self.n_nodes)]

    def add_edge(self, u: int, v: int, latency: float = 1.0, capacity: float = 10.0, bidirectional: bool = False) -> None:
        """Adds a directed or bidirectional link between infrastructure nodes."""
        if 0 <= u < self.n_nodes and 0 <= v < self.n_nodes:
            self.adj_unweighted[u].append(v)
            self.adj_weighted[u].append((v, float(latency)))
            self.capacity_matrix[u][v] += float(capacity)
            if bidirectional:
                self.adj_unweighted[v].append(u)
                self.adj_weighted[v].append((u, float(latency)))
                self.capacity_matrix[v][u] += float(capacity)

    def evaluate_resilience(self, source: int, sink: int) -> dict[str, Any]:
        """Evaluates throughput, single points of failure, latencies, and connectivity domains."""
        # 1. Dijkstra Latencies
        latencies = dijkstra_shortest_path_costs(self.n_nodes, self.adj_weighted, source)

        # 2. Strongly Connected Failure Domains (Tarjan SCC)
        sccs = tarjan_strongly_connected_components(self.n_nodes, self.adj_unweighted)

        # 3. Single-Points-of-Failure (Tarjan Cut-Vertices)
        cut_vertices = articulation_points_cut_vertices(self.n_nodes, self.adj_unweighted)

        # 4. Max Ingress-to-Egress Throughput (Dinic Max-Flow)
        max_throughput = dinic_level_graph_max_flow(self.n_nodes, self.capacity_matrix, source, sink)

        return {
            "source": source,
            "sink": sink,
            "min_latencies_from_source": latencies,
            "strongly_connected_components": sccs,
            "critical_cut_vertices": sorted(cut_vertices),
            "max_flow_throughput": max_throughput,
        }

    def schedule_service_mesh_stages(self, service_dependencies: list[list[int]]) -> list[int]:
        """Determines valid non-cyclical deployment order for service components via Kahn's algorithm."""
        n_services = len(service_dependencies)
        return topological_sort_kahn_indegree(n_services, service_dependencies)

    def allocate_hardening_budget(
        self,
        candidate_costs: list[int],
        vulnerability_reductions: list[int],
        budget_limit: int,
    ) -> int:
        """Computes optimal security hardening gain achievable within monetary budget constraints."""
        return knapsack_01_dp_optimal_value(candidate_costs, vulnerability_reductions, budget_limit)


def run_synthetic_benchmark() -> dict[str, Any]:
    """Runs end-to-end benchmark on realistic multi-tier cloud datacenter topology."""
    # 8-node topology:
    # 0: Ingress Edge Gateway
    # 1: Security Firewall Tier A
    # 2: Security Firewall Tier B
    # 3: Core Service Mesh Load Balancer
    # 4: Microservice Worker Pool 1
    # 5: Microservice Worker Pool 2
    # 6: Database Primary Node
    # 7: Analytics Egress Sink
    opt = ResilientInfrastructureOptimizer(
        n_nodes=8,
        node_labels=["EdgeGW", "FW_A", "FW_B", "CoreLB", "Worker1", "Worker2", "DB_Primary", "EgressSink"]
    )

    opt.add_edge(0, 1, latency=2.0, capacity=20.0)
    opt.add_edge(0, 2, latency=3.0, capacity=30.0)
    opt.add_edge(1, 3, latency=1.5, capacity=25.0)
    opt.add_edge(2, 3, latency=2.5, capacity=15.0)
    opt.add_edge(3, 4, latency=1.0, capacity=20.0)
    opt.add_edge(3, 5, latency=1.2, capacity=20.0)
    opt.add_edge(4, 6, latency=1.0, capacity=15.0)
    opt.add_edge(5, 6, latency=1.0, capacity=15.0)
    opt.add_edge(6, 7, latency=3.0, capacity=40.0)

    # Cross redundant link
    opt.add_edge(1, 2, latency=1.0, capacity=10.0, bidirectional=True)

    resilience = opt.evaluate_resilience(source=0, sink=7)

    # Service dependency DAG for 5 microservices:
    # Service 0 -> 1, 2; Service 1 -> 3; Service 2 -> 3; Service 3 -> 4
    service_deps = [
        [1, 2],  # Service 0 deploys before 1 and 2
        [3],     # Service 1 deploys before 3
        [3],     # Service 2 deploys before 3
        [4],     # Service 3 deploys before 4
        [],      # Service 4 has no dependents
    ]
    deploy_order = opt.schedule_service_mesh_stages(service_deps)

    # Hardening options: cost ($k) vs risk reduction points
    costs = [10, 15, 20, 25, 30]
    benefits = [24, 30, 45, 55, 70]
    budget = 50
    optimal_hardening = opt.allocate_hardening_budget(costs, benefits, budget)

    return {
        "resilience": resilience,
        "deployment_order": deploy_order,
        "optimal_hardening_gain": optimal_hardening,
    }
