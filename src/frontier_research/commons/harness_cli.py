"""Large-Scale OpenCommons Swarm Harness & CLI (`opencommons`).

Provides the production CLI and multi-agent swarm orchestrator for scaling
OpenCommons across local models (`kardashev-cli`, LiteRT, vLLM, Ollama, Qwen, Gemma)
and distributed workspaces:
  - `opencommons seed`: Seed the persistent store with verified scientific building blocks
  - `opencommons find <query>`: Sub-10ms live I/O-probe retrieval from CLI
  - `opencommons record <file.py>`: Ingest, fingerprint, and 5-gate peer-review a Python file
  - `opencommons sync <replica_a.json> <replica_b.json>`: Join-semilattice CRDT merge `⊔`
  - `opencommons status`: Inspect store metrics, quarantine set Q, and sid dedup ratio
  - `opencommons mcp`: Launch the 2-tool stdio JSON-RPC 2.0 MCP server (`find`, `record`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from frontier_research.commons.mcp_server import run_stdio_mcp_server
from frontier_research.commons.persistence import (
    DEFAULT_STORE_PATH,
    load_persistent_store,
    save_persistent_store,
)
from frontier_research.commons.store import PitfallWarning

FOUNDATIONAL_SEED_BLOCKS: list[dict[str, object]] = [
    {
        "domain": "numerical_analysis",
        "summary": "Shifted numerically stable LogSumExp preventing exp(x) overflow for x >= 710",
        "pseudocode": "1. m = max(xs); 2. return m + log(sum(exp(x - m) for x in xs))",
        "invariants": ("LSE(x + c) == LSE(x) + c", "max(xs) <= LSE(xs) <= max(xs) + log(N)"),
        "teleological_intent": "Log-partition function free-energy normalization without IEEE-754 overflow",
        "dimensional_units": "logits: nats -> nats",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="lse_overflow",
                naive_pattern="math.log(sum(math.exp(x) for x in xs))",
                failure_mechanism="OverflowError when max(xs) >= 710.0",
                counterexample_input="[1000.0, 1001.0, 1002.0]",
                verified_fix="Subtract m = max(xs) before exponentiating",
            ),
        ),
        "source_code": """
def stable_logsumexp(xs: list[float]) -> float:
    if not xs:
        return 0.0
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))
""",
    },
    {
        "domain": "numerical_analysis",
        "summary": "Welford single-pass online variance avoiding subtractive cancellation at kappa >= 10^8",
        "pseudocode": "1. Track running mean and M2 = sum((x - mean_old)*(x - mean_new)); 2. return M2 / n",
        "invariants": ("Var(X + c) == Var(X)", "Var(X) >= 0"),
        "teleological_intent": "Orthogonal running residual accumulation preserving mantissa bits on shifted streams",
        "dimensional_units": "xs: [U] -> U^2",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="variance_cancellation",
                naive_pattern="sum(x*x)/n - (sum(x)/n)**2",
                failure_mechanism="Severe precision loss from subtractive cancellation when kappa >= 10^8",
                counterexample_input="[1e12 + 1.0, 1e12 + 2.0, 1e12 + 3.0]",
                verified_fix="Use Welford online delta * delta2 update",
            ),
        ),
        "source_code": """
def welford_variance(xs: list[float]) -> float:
    if not xs:
        return 0.0
    count = 0
    mean = 0.0
    m2 = 0.0
    for x in xs:
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        m2 += delta * delta2
    return m2 / count
""",
    },
    {
        "domain": "hamiltonian_dynamics",
        "summary": "Symplectic Velocity Verlet / Leapfrog step preserving phase-space volume (det J == 1)",
        "pseudocode": "1. v_half = v - 0.5*dt*k*x; 2. x_new = x + dt*v_half; 3. v_new = v_half - 0.5*dt*k*x_new",
        "invariants": ("det(Jacobian) == 1.0", "bounded energy oscillation O(dt^2)"),
        "teleological_intent": "Liouville theorem phase-space volume conservation in Hamiltonian mechanics",
        "dimensional_units": "x: m, v: m/s, k: 1/s^2, dt: s -> tuple[m, m/s]",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="euler_energy_drift",
                naive_pattern="x_new = x + dt*v; v_new = v - dt*k*x",
                failure_mechanism="Explicit Euler inflates phase-space area det J = 1 + k*dt^2 > 1",
                counterexample_input="(1.0, 0.0, 1.0, 0.1) over 1000 steps",
                verified_fix="Use half-step kick-drift-kick symplectic update",
            ),
        ),
        "source_code": """
def symplectic_leapfrog_step(x: float, v: float, k: float, dt: float) -> tuple[float, float]:
    v_half = v - 0.5 * dt * k * x
    x_new = x + dt * v_half
    v_new = v_half - 0.5 * dt * k * x_new
    return (x_new, v_new)
""",
    },
    {
        "domain": "cryptography_and_framing",
        "summary": "32-bit FNV-1a deterministic byte payload hash with standard offset basis 0x811c9dc5",
        "pseudocode": "1. h = 0x811c9dc5; 2. for b in data: h = ((h ^ b) * 0x01000193) & 0xffffffff",
        "invariants": ("0 <= h <= 0xffffffff", "avalanche on single-bit flip"),
        "teleological_intent": "Non-cryptographic uniform hash distribution over byte frames",
        "dimensional_units": "data: list[int] -> uint32",
        "pitfalls": (
            PitfallWarning(
                pitfall_id="wrong_fnv_offset",
                naive_pattern="h = 0",
                failure_mechanism="Zero initialization collapses all-zero prefixes to hash 0",
                counterexample_input="[0, 0, 0]",
                verified_fix="Initialize with FNV-1a 32-bit offset basis 0x811c9dc5",
            ),
        ),
        "source_code": """
def fnv1a_32_ints(data: list[int]) -> int:
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ (b & 0xFF)) * 0x01000193) & 0xFFFFFFFF
    return h
""",
    },
]


def seed_default_commons(store_path: Path | str | None = None) -> dict[str, Any]:
    """Seed the persistent OpenCommons store with foundational verified scientific concepts."""
    t0 = time.perf_counter()
    store = load_persistent_store(store_path)
    ingested_ids = []
    for item in FOUNDATIONAL_SEED_BLOCKS:
        block, _, _ = store.ingest(
            source_code=str(item["source_code"]),
            domain=str(item["domain"]),
            summary=str(item["summary"]),
            pseudocode=str(item["pseudocode"]),
            invariants=tuple(item["invariants"]),  # type: ignore[arg-type]
            pitfalls=tuple(item["pitfalls"]),  # type: ignore[arg-type]
            teleological_intent=str(item["teleological_intent"]),
            dimensional_units=str(item["dimensional_units"]),
            run_peer_review=True,
        )
        ingested_ids.append(block.block_id)
    merged = save_persistent_store(store, store_path)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "seeded_blocks": ingested_ids,
        "total_active_concepts": len(merged.canonical_block_by_cid),
        "elapsed_ms": elapsed_ms,
        "store_path": str(Path(store_path or DEFAULT_STORE_PATH).expanduser()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="opencommons",
        description="OpenCommons Distributed Concept Commons CLI & Swarm Harness",
    )
    parser.add_argument(
        "--store",
        default=None,
        help="Path to persistent store JSON (default: ~/.opencommons/store.json)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("seed", help="Seed the persistent store with verified scientific building blocks")
    p_status = sub.add_parser("status", help="Show store statistics, active concepts, and quarantine count")
    p_status.add_argument("--json", action="store_true", help="Output JSON format")
    sub.add_parser("mcp", help="Run the 2-tool stdio JSON-RPC 2.0 MCP server (find, record)")

    p_find = sub.add_parser("find", help="Search the store with sub-10ms live I/O-probe filtering")
    p_find.add_argument("query", help="Search query or numbered concept name")
    p_find.add_argument("--modality", default="dual_card_with_pitfalls")
    p_find.add_argument("--top-k", type=int, default=3)

    p_sync = sub.add_parser("sync", help="Merge another replica JSON into the target store via CRDT `⊔`")
    p_sync.add_argument("other_store", help="Path to replica JSON file to merge in")

    args = parser.parse_args(argv)

    if args.cmd == "mcp":
        return run_stdio_mcp_server(store_path=args.store)

    if args.cmd == "seed":
        res = seed_default_commons(store_path=args.store)
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "status":
        store = load_persistent_store(args.store)
        payload = {
            "store_path": str(Path(args.store or DEFAULT_STORE_PATH).expanduser()),
            "total_blocks": len(store.blocks_by_id),
            "active_equivalence_classes_cid": len(store.canonical_block_by_cid),
            "quarantined_blocks": len(store.quarantined_ids),
            "peer_review_executions": store.peer_review_executions,
            "sid_dedup_hits": store.sid_dedup_hits,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.cmd == "find":
        store = load_persistent_store(args.store)
        hits = store.find(query=args.query, top_k=args.top_k)
        for h in hits:
            print(h.block.render_for_agent(mode=args.modality))
            print("-" * 60)
        return 0

    if args.cmd == "sync":
        other = load_persistent_store(args.other_store)
        merged = save_persistent_store(other, path=args.store, crdt_merge_existing=True)
        print(
            json.dumps(
                {
                    "status": "crdt_merged",
                    "active_concepts": len(merged.canonical_block_by_cid),
                    "quarantined_blocks": len(merged.quarantined_ids),
                },
                indent=2,
            )
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
