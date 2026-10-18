#!/usr/bin/env python3
"""OpenCommons Concept Catalog Ingestion and Synchronization Tool.

Reads concept catalogs (consolidated data/opencommons_expansion_v1.json or
per-domain data/domains/*.json), verifies schema conformance, AST non-triviality,
and security defense checks, and synchronizes concepts into the local OpenCommons
store (~/.opencommons/store.json) using atomic file writes and G-Set CRDT merge
semantics with POSIX advisory locking.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Dynamic Kardashev-CLI Module Shim Setup
# ---------------------------------------------------------------------------

def _setup_kardashev_shim() -> None:
    """Ensure kardashev_cli submodules can be imported cleanly without external deps."""
    if "kardashev_cli" in sys.modules and hasattr(sys.modules["kardashev_cli"], "__path__"):
        return

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "kardashev-cli" / "src" / "kardashev_cli"
        if candidate.is_dir():
            cli_mod = types.ModuleType("kardashev_cli")
            cli_mod.__path__ = [str(candidate)]
            sys.modules["kardashev_cli"] = cli_mod
            return
        candidate_sibling = parent.parent / "kardashev-cli" / "src" / "kardashev_cli"
        if candidate_sibling.is_dir():
            cli_mod = types.ModuleType("kardashev_cli")
            cli_mod.__path__ = [str(candidate_sibling)]
            sys.modules["kardashev_cli"] = cli_mod
            return


_setup_kardashev_shim()

try:
    from kardashev_cli.opencommons import (
        compute_structural_sid,
        fingerprint_function,
        generate_canonical_probes,
        synthesize_ast_guided_probes,
    )
    from kardashev_cli.opencommons_defense import evaluate_proposal_safety
    from kardashev_cli.opencommons_sandbox import run_sandboxed_probes
except ImportError:
    compute_structural_sid = None  # type: ignore[assignment]
    fingerprint_function = None  # type: ignore[assignment]
    generate_canonical_probes = None  # type: ignore[assignment]
    synthesize_ast_guided_probes = None  # type: ignore[assignment]
    evaluate_proposal_safety = None  # type: ignore[assignment]
    run_sandboxed_probes = None  # type: ignore[assignment]



# ---------------------------------------------------------------------------
# Default Paths and Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STORE_PATH = Path(
    os.environ.get("OPENCOMMONS_STORE_PATH", "~/.opencommons/store.json")
).expanduser()
DEFAULT_CATALOG_PATH = REPO_ROOT / "data" / "opencommons_expansion_v1.json"
DEFAULT_DOMAINS_GLOB = str(REPO_ROOT / "data" / "domains" / "*.json")


# ---------------------------------------------------------------------------
# Catalog Loading & Parsing
# ---------------------------------------------------------------------------

def find_catalog_files(catalog_arg: str | None = None) -> list[Path]:
    """Resolve one or more catalog file paths from command-line argument or default locations."""
    if catalog_arg:
        expanded_arg = os.path.expanduser(catalog_arg)
        matched_globs = glob.glob(expanded_arg)
        if matched_globs:
            files = [Path(p).resolve() for p in sorted(matched_globs) if Path(p).is_file()]
            if files:
                return files

        p = Path(expanded_arg).resolve()
        if p.is_dir():
            return sorted(p.glob("*.json"))
        elif p.is_file():
            return [p]
        else:
            return []

    # Default order: consolidated catalog first, otherwise domain catalogs
    if DEFAULT_CATALOG_PATH.is_file():
        return [DEFAULT_CATALOG_PATH]

    domain_files = [Path(p).resolve() for p in sorted(glob.glob(DEFAULT_DOMAINS_GLOB))]
    return domain_files


def load_catalog_items(file_path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Load concept proposals or blocks from a JSON catalog file.

    Returns:
        (items_list, success_flag)
    """
    if not file_path.is_file():
        return [], False
    try:
        content = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to parse JSON from {file_path}: {exc}", file=sys.stderr)
        return [], False

    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)], True
    elif isinstance(content, dict):
        if "blocks" in content and isinstance(content["blocks"], list):
            return [b for b in content["blocks"] if isinstance(b, dict)], True
        elif "concepts" in content and isinstance(content["concepts"], list):
            return [c for c in content["concepts"] if isinstance(c, dict)], True
        elif "proposals" in content and isinstance(content["proposals"], list):
            return [p for p in content["proposals"] if isinstance(p, dict)], True
    return [], False


# ---------------------------------------------------------------------------
# Verification Logic
# ---------------------------------------------------------------------------

def check_ast_non_triviality(source_code: str) -> tuple[bool, str | None]:
    """Verify AST non-triviality invariants (statements >= 4, node count >= 25, pure function)."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return False, f"AST SyntaxError: {exc}"

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False, "AST requires exactly one top-level FunctionDef"

    fn_def = tree.body[0]
    stmt_count = len(fn_def.body)
    if stmt_count < 4:
        return False, f"AST body statement count too low ({stmt_count} < 4)"

    node_count = sum(1 for _ in ast.walk(fn_def))
    if node_count < 25:
        return False, f"AST node count too low ({node_count} < 25)"

    for node in ast.walk(fn_def):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef)):
            return False, f"Forbidden AST node type: {type(node).__name__}"

    return True, None


def check_invariants_and_pitfalls(item: dict[str, Any]) -> tuple[bool, str | None]:
    """Verify presence and schema of invariants (>= 2) and pitfalls (>= 1)."""
    invariants = item.get("invariants")
    if not isinstance(invariants, (list, tuple)) or len(invariants) < 2:
        return False, f"Requires >= 2 invariants, found {len(invariants) if isinstance(invariants, (list, tuple)) else 0}"

    for i, inv in enumerate(invariants):
        if not isinstance(inv, str) or not inv.strip():
            return False, f"Invariant index {i} must be non-empty string"

    pitfalls = item.get("pitfalls")
    if not isinstance(pitfalls, (list, tuple)) or len(pitfalls) < 1:
        return False, f"Requires >= 1 pitfalls, found {len(pitfalls) if isinstance(pitfalls, (list, tuple)) else 0}"

    required_pitfall_keys = ("naive_pattern", "failure_mechanism", "counterexample_input", "verified_fix")
    for idx, pf in enumerate(pitfalls):
        if not isinstance(pf, dict):
            return False, f"Pitfall index {idx} must be a dict"
        for key in required_pitfall_keys:
            val = pf.get(key)
            if not isinstance(val, str) or not val.strip():
                return False, f"Pitfall index {idx} missing non-empty '{key}'"

    return True, None


def verify_concept_item(
    item: dict[str, Any],
    run_deep_sandbox: bool = False,
) -> tuple[bool, str | None]:
    """Run verification pipeline on a concept proposal.

    If run_deep_sandbox is False: FAST verification (AST non-triviality >= 4 stmts, >= 25 nodes,
    single pure FunctionDef, invariants >= 2, pitfalls >= 1, L1-L6 defense).
    If run_deep_sandbox is True: FULL verification (all above plus L7 out-of-process sandbox determinism,
    non-crashing execution, and zero in-place argument mutation).
    """
    # 1. AST non-triviality check
    source_code = item.get("source_code", "")
    ok, err = check_ast_non_triviality(source_code)
    if not ok:
        return False, f"AST Quality: {err}"

    # 2. Invariants & Pitfalls check
    ok, err = check_invariants_and_pitfalls(item)
    if not ok:
        return False, f"Contracts: {err}"

    # 3. L1-L6 (and L7 if run_deep_sandbox) Deterministic Defense via evaluate_proposal_safety
    if evaluate_proposal_safety is not None:
        try:
            verdict = evaluate_proposal_safety(item, skip_sandbox=(not run_deep_sandbox))
            if not verdict.is_accepted:
                return (
                    False,
                    f"Defense {verdict.failure_layer} [{verdict.failure_rule_id}]: {verdict.failure_detail}",
                )
        except Exception as exc:
            return False, f"Defense evaluator exception: {exc}"

    # 4. Out-of-process Sandbox L7 argument mutation & execution check
    if run_deep_sandbox and run_sandboxed_probes is not None:
        try:
            name = item.get("name", "concept")
            param_types = tuple(item.get("param_types", []))
            probes: list[tuple[Any, ...]] = []
            if generate_canonical_probes is not None:
                probes.extend(generate_canonical_probes(param_types, n_probes=16))
            if synthesize_ast_guided_probes is not None:
                ast_probes = synthesize_ast_guided_probes(source_code, n_probes=16)
                if ast_probes:
                    probes.extend(ast_probes)
            if not probes:
                probes = [()]

            sb_res = run_sandboxed_probes(
                source_code=source_code,
                func_name=name,
                probes=probes,
            )
            if not sb_res.success or not sb_res.is_deterministic:
                return False, f"Sandbox L7 failure: code={sb_res.error_code}, err={sb_res.error_message}"
            if any(pr.mutated_args for pr in sb_res.probe_results):
                return False, "Sandbox L7 failure: argument mutation detected in-place"
        except Exception as exc:
            return False, f"Sandbox runner exception: {exc}"

    return True, None


# ---------------------------------------------------------------------------
# Block Normalization & Store Conversion
# ---------------------------------------------------------------------------

def normalize_to_store_block(
    item: dict[str, Any],
    fallback_trust_tier: str = "probe_verified",
) -> dict[str, Any]:
    """Convert a concept proposal or raw block dict into standard OpenCommons store block format."""
    source_code = str(item.get("source_code", ""))
    name = str(item.get("name", "unnamed_concept"))

    # Compute or retrieve structural SID
    if "sid" in item and item["sid"]:
        sid = str(item["sid"])
    elif compute_structural_sid is not None:
        try:
            sid, _, _, _, _ = compute_structural_sid(source_code)
        except Exception:
            sid_digest = hashlib.sha256(source_code.encode("utf-8")).hexdigest()[:16]
            sid = f"sid:{sid_digest}"
    else:
        sid_digest = hashlib.sha256(source_code.encode("utf-8")).hexdigest()[:16]
        sid = f"sid:{sid_digest}"

    # Compute or retrieve block_id
    if "block_id" in item and item["block_id"]:
        block_id = str(item["block_id"])
    else:
        sid_suffix = sid.split(":", 1)[1][:8] if ":" in sid else sid[:8]
        block_id = f"{name}@{sid_suffix}"

    # Compute or retrieve behavioral fp and cid
    if "fp" in item and item["fp"]:
        fp = str(item["fp"])
    elif fingerprint_function is not None:
        try:
            fp_res = fingerprint_function(source_code)
            fp = fp_res.fp
        except Exception:
            fp = f"fp_struct:{sid.split(':', 1)[1] if ':' in sid else sid}"
    else:
        fp = f"fp_struct:{sid.split(':', 1)[1] if ':' in sid else sid}"

    if "cid" in item and item["cid"]:
        cid = str(item["cid"])
    else:
        fp_suffix = fp.split(":", 1)[1] if ":" in fp else fp
        cid = f"cid:{fp_suffix}"

    # Normalize pitfalls with pitfall_id
    normalized_pitfalls: list[dict[str, str]] = []
    for idx, pf in enumerate(item.get("pitfalls", [])):
        if isinstance(pf, dict):
            pid = str(pf.get("pitfall_id") or "")
            if not pid:
                pat = str(pf.get("naive_pattern", ""))
                cin = str(pf.get("counterexample_input", ""))
                h = hashlib.sha256(f"{pat}:{cin}".encode("utf-8")).hexdigest()[:8]
                pid = f"pf_{h}"
            normalized_pitfalls.append({
                "pitfall_id": pid,
                "naive_pattern": str(pf.get("naive_pattern", "")),
                "failure_mechanism": str(pf.get("failure_mechanism", "")),
                "counterexample_input": str(pf.get("counterexample_input", "")),
                "verified_fix": str(pf.get("verified_fix", "")),
            })

    return {
        "block_id": block_id,
        "cid": cid,
        "sid": sid,
        "fp": fp,
        "name": name,
        "domain": str(item.get("domain", "general")),
        "summary": str(item.get("summary", "")),
        "pseudocode": str(item.get("pseudocode", "")),
        "invariants": list(item.get("invariants", [])),
        "source_code": source_code,
        "param_types": list(item.get("param_types", [])),
        "return_type": str(item.get("return_type", "Any")),
        "dependencies": list(item.get("dependencies", [])),
        "pitfalls": normalized_pitfalls,
        "teleological_intent": str(item.get("teleological_intent", "")),
        "dimensional_units": str(item.get("dimensional_units", "")),
        "trust_tier": str(item.get("trust_tier", fallback_trust_tier)),
        "contributor_agent": str(item.get("contributor_agent", "opencommons_expansion")),
        "wave": int(item.get("wave", 1)),
        "created_at_ms": float(item.get("created_at_ms", time.time() * 1000.0)),
    }


# ---------------------------------------------------------------------------
# CRDT G-Set Store Operations & Atomic Persistence
# ---------------------------------------------------------------------------

def load_store(store_path: Path) -> dict[str, Any]:
    """Read existing store JSON from disk or initialize empty store structure."""
    if not store_path.is_file():
        return {
            "schema_version": "1.0.0",
            "blocks": [],
            "quarantined_ids": [],
            "peer_review_executions": 0,
            "sid_dedup_hits": 0,
        }
    try:
        data = json.loads(store_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "blocks" in data:
            return data
        else:
            raise ValueError("Store JSON root must be dict with 'blocks' key")
    except Exception as exc:
        ts = int(time.time())
        corrupt_backup = store_path.with_name(f"{store_path.name}.corrupt.{ts}")
        try:
            import shutil
            shutil.copy2(store_path, corrupt_backup)
            print(f"[WARN] Existing store at {store_path} corrupted ({exc}). Backed up to {corrupt_backup}. Starting fresh.", file=sys.stderr)
        except Exception as copy_exc:
            print(f"[WARN] Failed to backup corrupted store: {copy_exc}", file=sys.stderr)

    return {
        "schema_version": "1.0.0",
        "blocks": [],
        "quarantined_ids": [],
        "peer_review_executions": 0,
        "sid_dedup_hits": 0,
    }


TRUST_TIER_ORDER: dict[str, int] = {
    "unverified": 0,
    "probe_verified": 1,
    "peer_reviewed": 2,
    "quarantined": 99,
}


def crdt_merge_blocks(
    existing_store: dict[str, Any],
    incoming_blocks: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], int, int]:
    """Merge incoming blocks into existing store via CRDT G-Set join-semilattice.

    Returns:
      (merged_store_dict, added_count, dedup_hit_count)
    """
    blocks_by_id: dict[str, dict[str, Any]] = {}
    block_id_by_sid: dict[str, str] = {}

    for b in existing_store.get("blocks", []):
        bid = b["block_id"]
        blocks_by_id[bid] = dict(b)
        if "sid" in b:
            block_id_by_sid[b["sid"]] = bid

    quarantined = set(existing_store.get("quarantined_ids", []))
    peer_reviews = int(existing_store.get("peer_review_executions", 0))
    dedup_hits = int(existing_store.get("sid_dedup_hits", 0))

    added_count = 0
    new_dedup_hits = 0

    for b_orig in incoming_blocks:
        b = dict(b_orig)
        sid = b["sid"]
        bid = b["block_id"]
        incoming_tier = b.get("trust_tier", "unverified")

        # Deduplication check 1: SID already exists in store
        if sid in block_id_by_sid:
            new_dedup_hits += 1
            existing_bid = block_id_by_sid[sid]
            target_block = blocks_by_id[existing_bid]
            target_tier = target_block.get("trust_tier", "unverified")

            # Quarantine propagation (absorbing defect state)
            if incoming_tier == "quarantined" or bid in quarantined or existing_bid in quarantined:
                quarantined.add(existing_bid)
                target_block["trust_tier"] = "quarantined"
            elif target_tier != "quarantined":
                # Join-semilattice upgrade if incoming tier is higher priority
                if TRUST_TIER_ORDER.get(incoming_tier, 0) > TRUST_TIER_ORDER.get(target_tier, 0):
                    target_block["trust_tier"] = incoming_tier
            continue

        # Deduplication check 2: block_id already exists in store
        if bid in blocks_by_id:
            new_dedup_hits += 1
            target_block = blocks_by_id[bid]
            target_tier = target_block.get("trust_tier", "unverified")

            if incoming_tier == "quarantined" or bid in quarantined:
                quarantined.add(bid)
                target_block["trust_tier"] = "quarantined"
            elif target_tier != "quarantined":
                if TRUST_TIER_ORDER.get(incoming_tier, 0) > TRUST_TIER_ORDER.get(target_tier, 0):
                    target_block["trust_tier"] = incoming_tier
            continue

        # Net-new block
        if incoming_tier == "quarantined" or bid in quarantined:
            quarantined.add(bid)
            b["trust_tier"] = "quarantined"
        blocks_by_id[bid] = b
        block_id_by_sid[sid] = bid
        added_count += 1

    # Reconstruct sorted block list
    sorted_blocks = [blocks_by_id[bid] for bid in sorted(blocks_by_id.keys())]

    merged_store = {
        "schema_version": "1.0.0",
        "blocks": sorted_blocks,
        "quarantined_ids": sorted(list(quarantined)),
        "peer_review_executions": peer_reviews,
        "sid_dedup_hits": dedup_hits + new_dedup_hits,
    }
    return merged_store, added_count, new_dedup_hits


def save_store_atomic(
    store_data: dict[str, Any],
    store_path: Path,
    incoming_blocks: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Atomically save store JSON to disk with POSIX advisory locking and secure permissions."""
    store_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = store_path.with_suffix(store_path.suffix + ".lock")

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Re-read on-disk store inside lock to handle concurrent modifications
            if store_path.is_file():
                try:
                    on_disk = json.loads(store_path.read_text(encoding="utf-8"))
                    if incoming_blocks is not None:
                        merged, _, _ = crdt_merge_blocks(on_disk, incoming_blocks)
                        store_data = merged
                    else:
                        on_disk_sids = {b.get("sid") for b in on_disk.get("blocks", [])}
                        diff_blocks = [b for b in store_data.get("blocks", []) if b.get("sid") not in on_disk_sids]
                        if diff_blocks:
                            merged, _, _ = crdt_merge_blocks(on_disk, diff_blocks)
                            store_data = merged
                except (
                    OSError,
                    json.JSONDecodeError,
                    ValueError,
                    KeyError,
                    TypeError,
                    AttributeError,
                ) as exc:
                    # AttributeError covers malformed on-disk payloads (e.g. a
                    # JSON array or scalar where the merge expects an object
                    # with a "blocks" list) racing in from a concurrent writer.
                    sys.stderr.write(f"Warning: skipping concurrent CRDT merge for {store_path}: {exc}\n")

            serialized = json.dumps(store_data, indent=2, sort_keys=True) + "\n"
            fd, tmp_name = tempfile.mkstemp(
                dir=str(store_path.parent),
                prefix=".oc_tmp_",
                suffix=".json",
            )
            try:
                os.chmod(tmp_name, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                    tmp_f.write(serialized)
                os.replace(tmp_name, store_path)
            finally:
                if os.path.exists(tmp_name):
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# CLI Ingestion Controller
# ---------------------------------------------------------------------------

def run_sync(
    catalog_path: str | None = None,
    store_path: Path | str | None = None,
    dry_run: bool = False,
    verify: bool = False,
    verbose: bool = False,
) -> int:
    """Execute concept catalog ingestion and synchronization."""
    t_start = time.perf_counter()
    target_store = (
        Path(store_path).expanduser().resolve()
        if store_path
        else DEFAULT_STORE_PATH.resolve()
    )

    catalog_files = find_catalog_files(catalog_path)
    if not catalog_files:
        print("[ERROR] No catalog files found. Check --catalog argument or default paths.", file=sys.stderr)
        return 1

    print(f"=== OpenCommons Store Sync ===")
    print(f"Catalog sources: {[str(f) for f in catalog_files]}")
    print(f"Target store:    {target_store}")
    print(f"Mode:            {'DRY RUN (no disk writes)' if dry_run else 'LIVE SYNC'}")
    print(f"Verification:    {'FULL (L1-L7)' if verify else 'FAST (schema & AST)'}")

    all_raw_items: list[dict[str, Any]] = []
    has_parse_error = False
    for f in catalog_files:
        items, ok = load_catalog_items(f)
        if not ok:
            has_parse_error = True
        if verbose:
            print(f"  Loaded {len(items):>4} items from {f.name}")
        all_raw_items.extend(items)

    if has_parse_error:
        print("[ERROR] One or more catalog files failed to parse or could not be read.", file=sys.stderr)
        return 1

    print(f"Total raw concepts loaded: {len(all_raw_items)}")
    if not all_raw_items:
        print("[ERROR] No concepts found to ingest.", file=sys.stderr)
        return 1

    verified_blocks: list[dict[str, Any]] = []
    rejected_count = 0

    for idx, item in enumerate(all_raw_items):
        name = item.get("name", f"item_{idx}")
        # Always run verification: FAST (run_deep_sandbox=False) by default, FULL (True) if verify=True
        ok, err_msg = verify_concept_item(item, run_deep_sandbox=verify)
        if not ok:
            rejected_count += 1
            if verbose:
                print(f"  [REJECT] {name}: {err_msg}")
            continue

        block = normalize_to_store_block(
            item,
            fallback_trust_tier="probe_verified" if verify else "unverified",
        )
        verified_blocks.append(block)

    print(f"Verification results: {len(verified_blocks)} accepted, {rejected_count} rejected.")

    if rejected_count > 0:
        print(f"[ERROR] Ingestion halted: {rejected_count} concept(s) rejected during verification.", file=sys.stderr)
        return 1

    if not verified_blocks:
        print("[ERROR] Zero valid concepts available to ingest.", file=sys.stderr)
        return 1

    # Load existing store and merge
    existing_store = load_store(target_store)
    prior_block_count = len(existing_store.get("blocks", []))

    merged_store, added_count, dedup_hits = crdt_merge_blocks(existing_store, verified_blocks)
    final_block_count = len(merged_store["blocks"])

    print(f"\nMerge Statistics:")
    print(f"  Existing store blocks: {prior_block_count:>5}")
    print(f"  Incoming candidates:   {len(verified_blocks):>5}")
    print(f"  Net-new blocks added:  {added_count:>5}")
    print(f"  Deduplication hits:    {dedup_hits:>5}")
    print(f"  Final store total:     {final_block_count:>5}")

    if dry_run:
        print("\n[DRY RUN] All checks passed. No modifications written to disk.")
    else:
        save_store_atomic(merged_store, target_store, incoming_blocks=verified_blocks)
        print(f"\n[SUCCESS] Successfully synced {added_count} blocks to {target_store}.")

    elapsed = (time.perf_counter() - t_start) * 1000.0
    print(f"Elapsed time: {elapsed:.2f} ms")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize OpenCommons concept catalogs into local persistent store.",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Path or glob pattern to concept catalog file(s) (default: data/opencommons_expansion_v1.json or data/domains/*.json)",
    )
    parser.add_argument(
        "--store-path",
        type=str,
        default=None,
        help="Path to local OpenCommons store file (default: ~/.opencommons/store.json or $OPENCOMMONS_STORE_PATH)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and calculate merge diff without writing to disk",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run full L1-L6 defense and L7 sandbox verification on concepts before syncing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose progress output",
    )

    args = parser.parse_args()
    code = run_sync(
        catalog_path=args.catalog,
        store_path=args.store_path,
        dry_run=args.dry_run,
        verify=args.verify,
        verbose=args.verbose,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
