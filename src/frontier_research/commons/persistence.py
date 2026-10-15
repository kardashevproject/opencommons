"""Persistent Disk Serialization & Multi-Node CRDT Gossip Synchronization for OpenCommons.

Enables large-scale multi-agent swarms and `agy-local` local harnesses to persist
their shared Concept Commons to `~/.opencommons/store.json` (or any project path)
and merge distributed replica files via the join-semilattice CRDT operator `⊔`
(Theorem 1) with atomic file-lock safety.
"""

from __future__ import annotations

from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from frontier_research.commons.store import ConceptBlock, OpenCommonsStore, PitfallWarning

DEFAULT_STORE_PATH = Path(
    os.environ.get("OPENCOMMONS_STORE_PATH", "~/.opencommons/store.json")
).expanduser()


def serialize_block(block: ConceptBlock) -> dict[str, Any]:
    """Convert a ConceptBlock (including PitfallWarnings) to a JSON-serializable dict."""
    d = asdict(block)
    d["invariants"] = list(block.invariants)
    d["param_types"] = list(block.param_types)
    d["dependencies"] = list(block.dependencies)
    d["pitfalls"] = [asdict(p) for p in block.pitfalls]
    return d


def deserialize_block(data: dict[str, Any]) -> ConceptBlock:
    """Reconstruct a ConceptBlock from a JSON dict."""
    pitfalls = tuple(PitfallWarning(**p) for p in data.get("pitfalls", []))
    return ConceptBlock(
        block_id=data["block_id"],
        cid=data["cid"],
        sid=data["sid"],
        fp=data["fp"],
        name=data["name"],
        domain=data.get("domain", "general"),
        summary=data.get("summary", ""),
        pseudocode=data.get("pseudocode", ""),
        invariants=tuple(data.get("invariants", [])),
        source_code=data["source_code"],
        param_types=tuple(data.get("param_types", [])),
        return_type=data.get("return_type", "Any"),
        dependencies=tuple(data.get("dependencies", [])),
        pitfalls=pitfalls,
        teleological_intent=data.get("teleological_intent", ""),
        dimensional_units=data.get("dimensional_units", ""),
        trust_tier=data.get("trust_tier", "probe_verified"),
        contributor_agent=data.get("contributor_agent", "agent_0"),
        wave=int(data.get("wave", 1)),
        created_at_ms=float(data.get("created_at_ms", 0.0)),
    )


def store_to_dict(store: OpenCommonsStore) -> dict[str, Any]:
    """Serialize an entire OpenCommonsStore replica state S = (B, Q, C) to a dict."""
    blocks = [
        serialize_block(store.blocks_by_id[bid])
        for bid in sorted(store.blocks_by_id.keys())
    ]
    return {
        "schema_version": "1.0.0",
        "blocks": blocks,
        "quarantined_ids": sorted(store.quarantined_ids),
        "peer_review_executions": store.peer_review_executions,
        "sid_dedup_hits": store.sid_dedup_hits,
    }


def store_from_dict(payload: dict[str, Any]) -> OpenCommonsStore:
    """Reconstruct an OpenCommonsStore from a serialized replica dictionary."""
    store = OpenCommonsStore()
    q_set = set(payload.get("quarantined_ids", []))
    for b_dict in payload.get("blocks", []):
        if b_dict.get("block_id") in q_set:
            b_dict = dict(b_dict)
            b_dict["trust_tier"] = "quarantined"
        block = deserialize_block(b_dict)
        store._block_id_by_sid[block.sid] = block.block_id
        store._insert_block(block)
    store.quarantined_ids.update(q_set)
    store.peer_review_executions = int(payload.get("peer_review_executions", 0))
    store.sid_dedup_hits = int(payload.get("sid_dedup_hits", 0))
    return store


def load_persistent_store(path: Path | str | None = None) -> OpenCommonsStore:
    """Load an OpenCommonsStore from disk (or return an empty store if not yet created)."""
    target = Path(path).expanduser().resolve() if path else DEFAULT_STORE_PATH.resolve()
    if not target.is_file():
        return OpenCommonsStore()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        return store_from_dict(raw)
    except Exception:
        return OpenCommonsStore()


def save_persistent_store(
    store: OpenCommonsStore,
    path: Path | str | None = None,
    crdt_merge_existing: bool = True,
) -> OpenCommonsStore:
    """Atomically save an OpenCommonsStore to disk, merging with any concurrent updates via CRDT `⊔`."""
    target = Path(path).expanduser().resolve() if path else DEFAULT_STORE_PATH.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            final_store = store
            if crdt_merge_existing and target.is_file():
                try:
                    on_disk = store_from_dict(json.loads(target.read_text(encoding="utf-8")))
                    final_store = on_disk.merge(store)
                    final_store.peer_review_executions = (
                        on_disk.peer_review_executions + store.peer_review_executions
                    )
                    final_store.sid_dedup_hits = on_disk.sid_dedup_hits + store.sid_dedup_hits
                except Exception:
                    final_store = store

            serialized = json.dumps(store_to_dict(final_store), indent=2, sort_keys=True)
            fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".oc_tmp_", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                    tmp_f.write(serialized + "\n")
                os.replace(tmp_name, target)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            return final_store
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def seed_default_scientific_primitives(store: OpenCommonsStore) -> int:
    """Seed an OpenCommonsStore instance in-place with the 4 foundational scientific primitives."""
    from frontier_research.commons.harness_cli import FOUNDATIONAL_SEED_BLOCKS

    count = 0
    for item in FOUNDATIONAL_SEED_BLOCKS:
        store.ingest(
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
        count += 1
    return count

