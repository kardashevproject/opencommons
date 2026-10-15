"""Unit tests for the Open-Science Concept Commons engine."""

from __future__ import annotations

from frontier_research.commons import (
    OpenCommonsStore,
    PitfallWarning,
    fingerprint_function,
)


def test_behavioral_fingerprint_merges_distinct_algorithms() -> None:
    # Euclidean GCD (iterative while loop)
    src_euclid = """
def gcd_euclid(a: int, b: int) -> int:
    x, y = abs(a), abs(b)
    while y != 0:
        x, y = y, x % y
    return x
"""
    # Recursive GCD with different parameter names
    src_recursive = """
def greatest_common_divisor(u: int, v: int) -> int:
    u, v = abs(u), abs(v)
    if v == 0:
        return u
    return greatest_common_divisor(v, u % v)
"""
    # Mutant GCD (off-by-one bug)
    src_mutant = """
def gcd_buggy(a: int, b: int) -> int:
    x, y = abs(a), abs(b)
    while y > 1:
        x, y = y, x % y
    return x
"""
    fp1 = fingerprint_function(src_euclid)
    fp2 = fingerprint_function(src_recursive)
    fp_mut = fingerprint_function(src_mutant)

    assert fp1.sid != fp2.sid  # Structurally different ASTs
    assert fp1.fp == fp2.fp    # Identical behavioral fingerprint in <5ms!
    assert fp1.fp != fp_mut.fp # Mutant separated cleanly
    assert fp1.elapsed_ms < 25.0


def test_peer_review_quarantines_bugs_and_mutations() -> None:
    store = OpenCommonsStore()

    # 1. Degenerate constant stub
    stub_code = """
def compute_checksum(data: bytes) -> int:
    return 0
"""
    b_stub, _, v_stub = store.ingest(stub_code, run_peer_review=True)
    assert v_stub is not None and not v_stub.passed
    assert b_stub.trust_tier == "quarantined"

    # 2. Silent caller argument mutation (F80)
    mut_code = """
def sort_scores(xs: list[int]) -> list[int]:
    xs.sort()
    return xs
"""
    b_mut, _, v_mut = store.ingest(mut_code, run_peer_review=True, allow_mutation=False)
    assert v_mut is not None and not v_mut.passed
    assert b_mut.trust_tier == "quarantined"

    # 3. Clean pure implementation passes peer review
    clean_code = """
def sort_scores_pure(xs: list[int]) -> list[int]:
    return sorted(xs)
"""
    b_clean, _, v_clean = store.ingest(
        clean_code,
        run_peer_review=True,
        invariant_fn=lambda args, out: len(out) == len(args[0]) and out == sorted(args[0]),
    )
    assert v_clean is not None and v_clean.passed
    assert b_clean.trust_tier == "peer_reviewed"


def test_numbered_concept_discrimination_and_live_io_probes() -> None:
    store = OpenCommonsStore()

    b64_src = """
def base64_index(c: str) -> int:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    return alphabet.find(c) if len(c) == 1 else -1
"""
    b58_src = """
def base58_index(c: str) -> int:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    return alphabet.find(c) if len(c) == 1 else -1
"""
    store.ingest(b64_src, domain="encoding", summary="Lookup character index in standard base64 alphabet")
    store.ingest(b58_src, domain="encoding", summary="Lookup character index in Bitcoin base58 alphabet")

    # Query specifically for base58 must NOT return base64
    hits = store.find("base58 alphabet character index", param_types=("str",), return_type="int")
    assert len(hits) == 1
    assert hits[0].block.name == "base58_index"

    # Query with live I/O probe (in base58, '1' is index 0; in base64, '1' is index 53)
    io_hits = store.find(
        "character index in encoding alphabet",
        param_types=("str",),
        return_type="int",
        io_examples=[(("1",), 0), (("9",), 8)],
    )
    assert len(io_hits) == 1
    assert io_hits[0].block.name == "base58_index"
    assert io_hits[0].io_probe_verified is True


def test_dependency_bundle_and_crdt_semilattice_laws() -> None:
    s1 = OpenCommonsStore()
    s2 = OpenCommonsStore()

    helper_src = """
def kahan_step(total: float, comp: float, x: float) -> tuple[float, float]:
    y = x - comp
    t = total + y
    new_comp = (t - total) - y
    return (t, new_comp)
"""
    main_src = """
def kahan_sum(xs: list[float]) -> float:
    total = 0.0
    comp = 0.0
    for x in xs:
        total, comp = kahan_step(total, comp, x)
    return total
"""
    s1.ingest(helper_src, domain="numerical_science", summary="Single compensated Kahan summation step")
    s2.ingest(
        main_src,
        domain="numerical_science",
        summary="Numerically stable Kahan summation over float list",
        pseudocode="Maintain running compensation c = (t - sum) - y per element",
        invariants=("|kahan_sum(xs) - exact_sum(xs)| <= 2 * eps * sum(|xs|)",),
        pitfalls=(
            PitfallWarning(
                pitfall_id="P-NUM-01",
                naive_pattern="sum(xs) in left-to-right float64 loop",
                failure_mechanism="O(N * eps) precision loss when adding small increments to large accumulator",
                counterexample_input="[1e16, 1.0, -1e16]",
                verified_fix="Track high-order cancellation error in compensation register comp",
            ),
        ),
    )

    # CRDT laws: commutativity (s1 ⊔ s2 == s2 ⊔ s1) and idempotence (m ⊔ s1 == m)
    m12 = s1.merge(s2)
    m21 = s2.merge(s1)
    m_idem = m12.merge(s1)

    assert set(m12.canonical_block_by_cid.keys()) == set(m21.canonical_block_by_cid.keys())
    assert set(m12.canonical_block_by_cid.keys()) == set(m_idem.canonical_block_by_cid.keys())

    # Dependency bundle retrieval returns [kahan_step, kahan_sum] in topological order
    hits = m12.find("kahan_sum numerically stable summation", include_bundle=True)
    assert len(hits) >= 1
    bundle_names = [b.name for b in hits[0].bundle_blocks]
    assert bundle_names == ["kahan_step", "kahan_sum"]

    # Check multi-modal rendering token compression
    root_block = hits[0].block
    assert root_block.estimate_tokens("pseudocode") < root_block.estimate_tokens("dual_card_with_pitfalls")
    assert "P-NUM-01" in root_block.render_for_agent("dual_card_with_pitfalls") or "AVOID" in root_block.render_for_agent(
        "dual_card_with_pitfalls"
    )


def test_counterexample_ratchet_and_sid_deduplication() -> None:
    store = OpenCommonsStore()

    # 1. O(1) structural sid deduplication across variable-renamed duplicates
    clean_v1 = """
def safe_poly(x: int) -> int:
    return x * x + 2 * x + 1
"""
    clean_v2_renamed = """
def poly_renamed(val: int) -> int:
    return val * val + 2 * val + 1
"""
    b1, fp1, _ = store.ingest(
        clean_v1,
        domain="math",
        summary="Quadratic polynomial (x+1)^2",
        teleological_intent="Conserves positive-definite quadratic energy",
        dimensional_units="Joules (J)",
        run_peer_review=True,
    )
    b2, fp2, v2 = store.ingest(
        clean_v2_renamed,
        domain="math",
        summary="Renamed quadratic polynomial",
        run_peer_review=False,
    )
    assert fp1.sid == fp2.sid
    assert v2 is None  # Peer review skipped via O(1) sid deduplication
    assert b1.block_id == b2.block_id
    assert store.sid_dedup_hits == 1
    assert store.peer_review_executions == 1

    # Verify teleological intent and dimensional units are rendered in Dual Card
    rendered = b1.render_for_agent("dual_card")
    assert "Conserves positive-definite quadratic energy" in rendered
    assert "Joules (J)" in rendered

    # 2. Monotonic Counterexample Ratchet against (K+1)-th input blind spot
    trojan_src = """
def trojan_poly(x: int) -> int:
    target = (1 << 10) + 313  # 1337 hidden from AST literal extractor
    if x == target:
        raise ZeroDivisionError("17th input trap")
    return (x + 1) ** 2
"""
    b_trojan, _, _ = store.ingest(
        trojan_src,
        domain="math",
        summary="Trojan quadratic polynomial matching on 16 probes",
    )
    assert b_trojan.cid == b1.cid  # Shares behavioral equivalence class on K=16 probes

    # Force Trojan to be current representative to test promotion on quarantine
    store.canonical_block_by_cid[b1.cid] = b_trojan.block_id

    quarantined = store.report_counterexample(
        block_id=b_trojan.block_id,
        failing_args=(1337,),
        failure_description="Raises ZeroDivisionError on x=1337",
    )
    assert quarantined is True
    assert b_trojan.block_id in store.quarantined_ids
    assert store.canonical_block_by_cid[b1.cid] == b1.block_id


def test_persistent_store_and_mcp_server(tmp_path) -> None:
    from frontier_research.commons.harness_cli import main as cli_main
    from frontier_research.commons.mcp_server import handle_jsonrpc_request
    from frontier_research.commons.persistence import (
        load_persistent_store,
        save_persistent_store,
        seed_default_scientific_primitives,
    )

    store_file = tmp_path / "commons_store.json"

    # 1. Seed default primitives and save atomically
    s1 = OpenCommonsStore()
    added = seed_default_scientific_primitives(s1)
    assert added == 4
    save_persistent_store(s1, store_file)

    # 2. Concurrent store merges via CRDT on save
    s2 = OpenCommonsStore()
    s2.ingest(
        "def cube_int(n: int) -> int:\n    return n * n * n\n",
        domain="math",
        summary="Compute exact integer cube",
    )
    merged = save_persistent_store(s2, store_file)
    assert len(merged.canonical_block_by_cid) == 5

    loaded = load_persistent_store(store_file)
    assert len(loaded.canonical_block_by_cid) == 5

    # 3. Test 2-tool JSON-RPC 2.0 MCP server (initialize, tools/list, find, record)
    init_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, store_path=store_file)
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "opencommons-mcp"

    tools_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, store_path=store_file)
    assert tools_resp is not None
    tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
    assert tool_names == ["find", "record"]

    find_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "find",
                "arguments": {"query": "numerically stable logsumexp", "modality": "dual_card"},
            },
        },
        store_path=store_file,
    )
    assert find_resp is not None
    assert "stable_logsumexp" in find_resp["result"]["content"][0]["text"]

    record_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "record",
                "arguments": {
                    "code": "def square_int(n: int) -> int:\n    return n * n\n",
                    "domain": "math",
                    "summary": "Compute integer square",
                    "teleological_intent": "Area scaling law",
                    "dimensional_units": "m^2",
                },
            },
        },
        store_path=store_file,
    )
    assert record_resp is not None
    rec_text = record_resp["result"]["content"][0]["text"]
    assert '"status": "accepted"' in rec_text

    # 4. Test CLI status and find commands
    rc_status = cli_main(["--store", str(store_file), "status", "--json"])
    assert rc_status == 0
    rc_find = cli_main(["--store", str(store_file), "find", "integer square", "--top-k", "1"])
    assert rc_find == 0

