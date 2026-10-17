"""Distributed Open-Science Concept Commons store, multi-modal concept representation
(raw code, algorithmic pseudocode building blocks, dual concept cards, and refuted pitfalls),
numbered-concept & live I/O-probe retrieval, and G-Set CRDT federation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import re
import time
from typing import Any, Callable, Literal

from frontier_research.commons.fingerprint import (
    FingerprintResult,
    _canonical_serialize,
    fingerprint_function,
)
from frontier_research.commons.peer_review import (
    PeerReviewVerdict,
    TrustTier,
    run_async_peer_review,
)

ServingMode = Literal["raw_code", "pseudocode", "dual_card", "dual_card_with_pitfalls"]

_TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+")
_ALPHANUM_CONCEPT_RE = re.compile(r"([a-zA-Z]+)(\d+)")


def tokenize_concept_query(text: str) -> tuple[set[str], dict[str, set[str]]]:
    """Tokenize identifier/query text while preserving numbered-concept stems (F87 lesson).

    Returns:
      - general_tokens: lowercase word/number tokens
      - numbered_stems: mapping from alpha stem -> set of numeric suffixes
        (e.g. 'base58' -> {'base': {'58'}}, 'crc16' -> {'crc': {'16'}}, 'rk4' -> {'rk': {'4'}})
        so 'base58' and 'base64' are explicitly discriminated rather than conflated!
    """
    lowered = text.lower()
    numbered_stems: dict[str, set[str]] = {}
    for m in _ALPHANUM_CONCEPT_RE.finditer(lowered):
        stem, num = m.group(1), m.group(2)
        if len(stem) >= 2:
            numbered_stems.setdefault(stem, set()).add(num)

    raw_parts = re.split(r"[^a-z0-9]+", lowered)
    tokens: set[str] = set()
    for p in raw_parts:
        if not p:
            continue
        tokens.add(p)
        for sub in _TOKEN_RE.findall(p):
            if len(sub) >= 2 or sub.isdigit():
                tokens.add(sub)
    return tokens, numbered_stems


@dataclass(frozen=True)
class PitfallWarning:
    """A documented dead-end, numerical instability, or edge-case counterexample attached to a concept."""

    pitfall_id: str
    naive_pattern: str
    failure_mechanism: str
    counterexample_input: str
    verified_fix: str


@dataclass(frozen=True)
class ConceptBlock:
    """A unit of scientific or algorithmic knowledge stored in the Open-Science Concept Commons."""

    block_id: str
    cid: str
    sid: str
    fp: str
    name: str
    domain: str
    summary: str
    pseudocode: str
    invariants: tuple[str, ...]
    source_code: str
    param_types: tuple[str, ...]
    return_type: str
    dependencies: tuple[str, ...] = ()
    pitfalls: tuple[PitfallWarning, ...] = ()
    teleological_intent: str = ""
    dimensional_units: str = ""
    trust_tier: TrustTier = "probe_verified"
    contributor_agent: str = "agent_0"
    wave: int = 1
    created_at_ms: float = field(default_factory=lambda: time.time() * 1000.0)

    def render_for_agent(self, mode: ServingMode = "dual_card_with_pitfalls") -> str:
        """Render the concept in the requested representation modality to measure token efficiency & transfer."""
        sig = f"{self.name}({', '.join(self.param_types)}) -> {self.return_type}"
        if mode == "raw_code":
            return f"# Concept: {sig} [{self.cid}]\n{self.source_code.strip()}"

        intent_line = f"\n  Intent    : {self.teleological_intent}" if self.teleological_intent else ""
        units_line = f"\n  Units     : {self.dimensional_units}" if self.dimensional_units else ""

        if mode == "pseudocode":
            inv_str = "; ".join(self.invariants) if self.invariants else "total on domain"
            return (
                f"CONCEPT {sig} [{self.cid}]\n"
                f"  Summary   : {self.summary}{intent_line}{units_line}\n"
                f"  Algorithm : {self.pseudocode}\n"
                f"  Invariants: {inv_str}"
            )

        if mode == "dual_card":
            inv_str = "; ".join(self.invariants) if self.invariants else "total on domain"
            return (
                f"CONCEPT {sig} [{self.cid} | tier={self.trust_tier}]\n"
                f"  Summary   : {self.summary}{intent_line}{units_line}\n"
                f"  Algorithm : {self.pseudocode}\n"
                f"  Invariants: {inv_str}\n"
                f"  Reference :\n{self.source_code.strip()}"
            )

        # dual_card_with_pitfalls (full Open-Science Concept Card)
        inv_str = "; ".join(self.invariants) if self.invariants else "total on domain"
        pitfall_lines = ""
        if self.pitfalls:
            warnings = [
                f"    - AVOID '{p.naive_pattern}': {p.failure_mechanism} "
                f"(fails at {p.counterexample_input}) -> {p.verified_fix}"
                for p in self.pitfalls
            ]
            pitfall_lines = "\n  Refuted Pitfalls / Dead-Ends:\n" + "\n".join(warnings)
        return (
            f"CONCEPT {sig} [{self.cid} | tier={self.trust_tier}]\n"
            f"  Summary   : {self.summary}{intent_line}{units_line}\n"
            f"  Algorithm : {self.pseudocode}\n"
            f"  Invariants: {inv_str}{pitfall_lines}\n"
            f"  Reference :\n{self.source_code.strip()}"
        )

    def estimate_tokens(self, mode: ServingMode = "dual_card_with_pitfalls") -> int:
        """Estimate LLM BPE token count (~3.8 chars/token for code/technical text)."""
        rendered = self.render_for_agent(mode=mode)
        return max(1, int(round(len(rendered) / 3.8)))


@dataclass(frozen=True)
class RetrievalHit:
    """A ranked concept retrieval result with behavioral probe verification status."""

    block: ConceptBlock
    score: float
    io_probe_verified: bool
    bundle_blocks: tuple[ConceptBlock, ...]
    latency_ms: float


class OpenCommonsStore:
    """Fast, meaning-indexed Open-Science Concept Commons with G-Set CRDT merge."""

    def __init__(self) -> None:
        self.blocks_by_id: dict[str, ConceptBlock] = {}
        self.cid_by_fp: dict[str, str] = {}
        self.canonical_block_by_cid: dict[str, str] = {}
        self.quarantined_ids: set[str] = set()
        self.naming_collisions: list[tuple[str, str, str]] = []  # (name, cid_1, cid_2)
        self._token_index: dict[str, set[str]] = {}
        self._name_to_cids: dict[str, set[str]] = {}
        self._compiled_cache: dict[str, Callable[..., Any]] = {}
        self._block_id_by_sid: dict[str, str] = {}
        self.peer_review_executions: int = 0
        self.sid_dedup_hits: int = 0

    def ingest(
        self,
        source_code: str,
        domain: str = "general",
        summary: str = "",
        pseudocode: str = "",
        invariants: tuple[str, ...] = (),
        pitfalls: tuple[PitfallWarning, ...] = (),
        teleological_intent: str = "",
        dimensional_units: str = "",
        contributor_agent: str = "agent_0",
        wave: int = 1,
        run_peer_review: bool = False,
        contract_io: list[tuple[tuple[Any, ...], Any]] | None = None,
        invariant_fn: Callable[[tuple[Any, ...], Any], bool] | None = None,
        reference_fn: Callable[..., Any] | None = None,
        allow_mutation: bool = False,
    ) -> tuple[ConceptBlock, FingerprintResult, PeerReviewVerdict | None]:
        """Ingest a native Python concept block in <5ms, computing sid and behavioral fp.

        Performs O(1) structural sid deduplication so redundant submissions from swarm
        agents bypass duplicate 16-probe and 5-gate peer-review compute.
        """
        from frontier_research.commons.fingerprint import compute_structural_sid

        sid, func_name, param_types, return_type, deps = compute_structural_sid(source_code)
        block_id = f"{func_name}@{sid[4:12]}"

        # Fast-path O(1) structural sid deduplication against carrying tax
        if sid in self._block_id_by_sid and self._block_id_by_sid[sid] in self.blocks_by_id:
            existing_block = self.blocks_by_id[self._block_id_by_sid[sid]]
            if not contract_io and not invariant_fn and not reference_fn:
                self.sid_dedup_hits += 1
                cached_fp = FingerprintResult(
                    sid=existing_block.sid,
                    fp=existing_block.fp,
                    func_name=existing_block.name,
                    param_types=existing_block.param_types,
                    return_type=existing_block.return_type,
                    dependencies=existing_block.dependencies,
                    valid_probe_count=16,
                    total_probe_count=16,
                    mutates_arguments=False,
                    is_constant_output=False,
                    elapsed_ms=0.05,
                    sample_outputs=(),
                )
                cached_verdict = (
                    PeerReviewVerdict(
                        trust_tier=existing_block.trust_tier,
                        passed=(existing_block.trust_tier != "quarantined"),
                        reasons=(),
                    )
                    if run_peer_review
                    else None
                )
                return existing_block, cached_fp, cached_verdict

        extra_globals = self._build_dependency_globals()
        fp_res = fingerprint_function(source_code, extra_globals=extra_globals)

        # Canonical Equivalence Class ID (cid) is keyed by behavioral fingerprint fp
        if fp_res.fp in self.cid_by_fp:
            cid = self.cid_by_fp[fp_res.fp]
        else:
            cid = f"cid:{fp_res.fp.split(':', 1)[1]}"
            self.cid_by_fp[fp_res.fp] = cid

        auto_summary = summary or f"{fp_res.func_name}({', '.join(fp_res.param_types)}) -> {fp_res.return_type}"
        auto_pseudo = pseudocode or f"Compute {fp_res.func_name} over {', '.join(fp_res.param_types)}"

        verdict: PeerReviewVerdict | None = None
        trust_tier: TrustTier = "probe_verified"
        if run_peer_review:
            self.peer_review_executions += 1
            verdict = run_async_peer_review(
                source_code=source_code,
                fp_result=fp_res,
                contract_io=contract_io,
                invariant_fn=invariant_fn,
                reference_fn=reference_fn,
                allow_mutation=allow_mutation,
                extra_globals=extra_globals,
            )
            trust_tier = verdict.trust_tier

        block = ConceptBlock(
            block_id=block_id,
            cid=cid,
            sid=fp_res.sid,
            fp=fp_res.fp,
            name=fp_res.func_name,
            domain=domain,
            summary=auto_summary,
            pseudocode=auto_pseudo,
            invariants=invariants,
            source_code=source_code,
            param_types=fp_res.param_types,
            return_type=fp_res.return_type,
            dependencies=fp_res.dependencies,
            pitfalls=pitfalls,
            teleological_intent=teleological_intent,
            dimensional_units=dimensional_units,
            trust_tier=trust_tier,
            contributor_agent=contributor_agent,
            wave=wave,
        )

        self._block_id_by_sid[sid] = block_id
        self._insert_block(block)
        return block, fp_res, verdict

    def report_counterexample(
        self,
        block_id: str,
        failing_args: tuple[Any, ...],
        expected_output: Any | None = None,
        invariant_fn: Callable[[tuple[Any, ...], Any], bool] | None = None,
        failure_description: str = "unprobed_17th_input_counterexample",
    ) -> bool:
        """Monotonic Counterexample Ratchet against the (K+1)-th Input Probe Blind Spot.

        If a downstream agent encounters an unprobed (K+1)-th input x_{K+1} where a stored
        block raises an exception, violates a domain invariant, or disagrees with expected
        output, this method verifies the failure, permanently quarantines the block in Q,
        promotes the next valid representative for that cid, and attaches a PitfallWarning.
        Returns True if the block was refuted and quarantined.
        """
        if block_id not in self.blocks_by_id:
            return False
        block = self.blocks_by_id[block_id]
        fn = self._compiled_cache.get(block_id)
        refuted = False
        if fn is None:
            refuted = True
        else:
            try:
                out = fn(*copy.deepcopy(failing_args))
                if expected_output is not None and _canonical_serialize(out) != _canonical_serialize(expected_output):
                    refuted = True
                if invariant_fn is not None and not invariant_fn(failing_args, out):
                    refuted = True
            except Exception:  # noqa: BLE001
                refuted = True

        if not refuted:
            return False

        # Monotonically quarantine the refuted block
        quarantined_block = replace(block, trust_tier="quarantined")
        self.blocks_by_id[block_id] = quarantined_block
        self.quarantined_ids.add(block_id)

        # Re-elect canonical block for cid from remaining non-quarantined blocks
        if self.canonical_block_by_cid.get(block.cid) == block_id:
            del self.canonical_block_by_cid[block.cid]
            for cand_id, cand in self.blocks_by_id.items():
                if cand.cid == block.cid and cand_id not in self.quarantined_ids:
                    self.canonical_block_by_cid[block.cid] = cand_id
                    break
        return True

    def _build_dependency_globals(self) -> dict[str, Any]:
        """Provide already-verified commons functions in execution namespace for compositional waves."""
        import math as _math
        ns: dict[str, Any] = {"math": _math}
        for cid, bid in self.canonical_block_by_cid.items():
            if bid in self.quarantined_ids:
                continue
            b = self.blocks_by_id[bid]
            if bid in self._compiled_cache:
                ns[b.name] = self._compiled_cache[bid]
        return ns

    def _insert_block(self, block: ConceptBlock) -> None:
        self.blocks_by_id[block.block_id] = block
        if block.trust_tier == "quarantined":
            self.quarantined_ids.add(block.block_id)
            return

        # Compile callable once for fast <1ms live I/O probe filtering
        try:
            ns = self._build_dependency_globals()
            exec(compile(block.source_code, f"<{block.block_id}>", "exec"), ns)  # noqa: S102
            if block.name in ns and callable(ns[block.name]):
                self._compiled_cache[block.block_id] = ns[block.name]
        except (SyntaxError, TypeError, ValueError, NameError, RuntimeError):
            self._compiled_cache.pop(block.block_id, None)

        self.cid_by_fp.setdefault(block.fp, block.cid)

        # Track naming collisions (same function name, distinct behavioral fp/cid)
        existing_cids = self._name_to_cids.setdefault(block.name, set())
        for existing_cid in existing_cids:
            if existing_cid != block.cid:
                pair = (block.name, min(existing_cid, block.cid), max(existing_cid, block.cid))
                if pair not in self.naming_collisions:
                    self.naming_collisions.append(pair)
        existing_cids.add(block.cid)

        # Keep the highest-tier or most-documented canonical representative per cid
        if block.cid not in self.canonical_block_by_cid:
            self.canonical_block_by_cid[block.cid] = block.block_id
        else:
            curr = self.blocks_by_id[self.canonical_block_by_cid[block.cid]]
            if (block.trust_tier == "peer_reviewed" and curr.trust_tier != "peer_reviewed") or (
                len(block.pitfalls) > len(curr.pitfalls)
            ):
                self.canonical_block_by_cid[block.cid] = block.block_id

        # Index tokens for O(1) candidate lookup
        index_text = f"{block.name} {block.domain} {block.summary} {block.pseudocode}"
        tokens, _ = tokenize_concept_query(index_text)
        for tok in tokens:
            self._token_index.setdefault(tok, set()).add(block.cid)

    def find(
        self,
        query: str,
        param_types: tuple[str, ...] | None = None,
        return_type: str | None = None,
        io_examples: list[tuple[tuple[Any, ...], Any]] | None = None,
        top_k: int = 3,
        include_bundle: bool = True,
        min_score: float = 2.5,
        enforce_numbered_discrimination: bool = True,
    ) -> list[RetrievalHit]:
        """Retrieve verified concepts matching query, signature, and live I/O probe assertions in <5ms."""
        t0 = time.perf_counter()
        q_tokens, q_num_stems = tokenize_concept_query(query)

        # Gather candidate cids via inverted token index (or all active cids if small)
        candidate_cids: set[str] = set()
        for tok in q_tokens:
            if tok in self._token_index:
                candidate_cids.update(self._token_index[tok])
        if not candidate_cids:
            candidate_cids = set(self.canonical_block_by_cid.keys())

        scored: list[tuple[float, bool, ConceptBlock]] = []
        for cid in candidate_cids:
            bid = self.canonical_block_by_cid.get(cid)
            if not bid or bid in self.quarantined_ids:
                continue
            block = self.blocks_by_id[bid]

            name_tokens, b_num_stems = tokenize_concept_query(f"{block.name} {block.summary}")
            doc_tokens, _ = tokenize_concept_query(f"{block.domain} {block.pseudocode}")

            # F87 Numbered-concept conflict penalty (e.g. query asks for base58, block is base64)
            if enforce_numbered_discrimination and q_num_stems:
                numbered_conflict = False
                for stem, q_nums in q_num_stems.items():
                    if stem in b_num_stems and b_num_stems[stem] != q_nums:
                        numbered_conflict = True
                        break
                if numbered_conflict:
                    continue

            # Compute hybrid relevance score
            name_overlap = len(q_tokens & name_tokens)
            doc_overlap = len(q_tokens & doc_tokens)
            score = 3.0 * name_overlap + 1.0 * doc_overlap

            # Exact identifier substring bonus
            if block.name.lower() in query.lower():
                score += 4.0

            # Type signature compatibility check
            if param_types is not None:
                if len(param_types) != len(block.param_types):
                    continue
                type_matches = sum(
                    1
                    for qp, bp in zip(param_types, block.param_types)
                    if qp.lower() == bp.lower() or "any" in (qp.lower(), bp.lower())
                )
                score += 1.5 * type_matches

            if return_type is not None and return_type.lower() != "any":
                if block.return_type.lower() != return_type.lower() and block.return_type.lower() != "any":
                    continue
                score += 2.0

            # Live I/O Probe Verification (<1ms execution against concrete input-output pairs)
            io_verified = False
            if io_examples:
                fn = self._compiled_cache.get(block.block_id)
                if fn is None:
                    continue
                all_passed = True
                for args, expected_out in io_examples:
                    try:
                        actual_out = fn(*copy.deepcopy(args))
                        if _canonical_serialize(actual_out) != _canonical_serialize(expected_out):
                            all_passed = False
                            break
                    except Exception:  # noqa: BLE001
                        all_passed = False
                        break
                if not all_passed:
                    continue
                io_verified = True
                score += 10.0  # Guaranteed behavioral match

            if score >= min_score:
                scored.append((score, io_verified, block))

        scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        hits: list[RetrievalHit] = []
        for score, io_ver, block in scored[:top_k]:
            bundle = self._resolve_bundle(block) if include_bundle else (block,)
            hits.append(
                RetrievalHit(
                    block=block,
                    score=round(score, 3),
                    io_probe_verified=io_ver,
                    bundle_blocks=bundle,
                    latency_ms=round(elapsed_ms, 4),
                )
            )
        return hits

    def _resolve_bundle(self, root_block: ConceptBlock) -> tuple[ConceptBlock, ...]:
        """Resolve transitive dependency closure in topological order (helpers first, root last)."""
        visited: set[str] = set()
        ordered: list[ConceptBlock] = []

        def _dfs(b: ConceptBlock) -> None:
            if b.cid in visited:
                return
            visited.add(b.cid)
            for dep_name in b.dependencies:
                dep_cids = self._name_to_cids.get(dep_name, set())
                for dcid in dep_cids:
                    dbid = self.canonical_block_by_cid.get(dcid)
                    if dbid and dbid not in self.quarantined_ids:
                        _dfs(self.blocks_by_id[dbid])
                        break
            ordered.append(b)

        _dfs(root_block)
        return tuple(ordered)

    def merge(self, other: OpenCommonsStore) -> OpenCommonsStore:
        """State-based G-Set join-semilattice CRDT merge (L_self ⊔ L_other).

        Associative, commutative, and idempotent without coordination locks.
        """
        merged = OpenCommonsStore()
        all_blocks = sorted(
            list(self.blocks_by_id.values()) + list(other.blocks_by_id.values()),
            key=lambda b: (b.block_id, b.cid),
        )
        for b in all_blocks:
            if b.block_id not in merged.blocks_by_id or (
                b.trust_tier == "quarantined" and b.block_id not in merged.quarantined_ids
            ):
                merged._insert_block(b)
        return merged
