"""Ultra-Compact 2-Tool JSON-RPC 2.0 MCP Server (`opencommons-mcp`).

Implements the minimal 2-tool Model Context Protocol interface validated in
Experiment R-005 (~681 schema tokens, eliminating the 18-tool 1.399x context tax):
1. `find`: Query the Concept Commons by natural language / numbered concept stem,
   parameter/return type signature, and live <2ms input-output probe examples,
   returning verified Dual Concept Cards (with topological dependency bundles).
2. `record`: Ingest a newly synthesized Python function into the Concept Commons
   (with O(1) structural AST Merkle `sid` deduplication, 16-probe canonical
   behavioral fingerprinting `fp -> cid`, and 5-gate adversarial peer review)
   or report a (K+1)-th input counterexample (`report_counterexample`).
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
from typing import Any

from frontier_research.commons.persistence import (
    DEFAULT_STORE_PATH,
    load_persistent_store,
    save_persistent_store,
)
from frontier_research.commons.store import PitfallWarning, ServingMode

MCP_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "find",
        "description": (
            "Search OpenCommons for verified scientific/algorithmic building blocks by query, "
            "type signature, and live <2ms I/O probe assertions. Returns Dual Concept Cards "
            "and topological dependency bundles."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Algorithm name, domain keywords, or numbered concept (e.g. 'rk4', 'stable_logsumexp', 'crc32').",
                },
                "param_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional parameter type hints, e.g. ['list[float]'] or ['int', 'int'].",
                },
                "return_type": {
                    "type": "string",
                    "description": "Optional return type hint, e.g. 'float' or 'int'.",
                },
                "io_examples": {
                    "type": "string",
                    "description": "Optional Python literal list of ((arg1, ...), expected_output) pairs for live I/O probe filtering.",
                },
                "modality": {
                    "type": "string",
                    "enum": ["dual_card_with_pitfalls", "dual_card", "pseudocode", "raw_code"],
                    "default": "dual_card_with_pitfalls",
                },
                "top_k": {
                    "type": "integer",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "record",
        "description": (
            "Contribute a verified Python function to OpenCommons (computes AST Merkle sid, "
            "16-probe behavioral fp/cid, and runs 5-gate adversarial peer review) or report "
            "a counterexample on an existing block."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_code": {
                    "type": "string",
                    "description": "Complete Python def function source code.",
                },
                "domain": {
                    "type": "string",
                    "default": "general",
                    "description": "Scientific/engineering domain (e.g. 'numerical_analysis', 'cryptography', 'biophysics').",
                },
                "summary": {
                    "type": "string",
                    "description": "Concise 1-line summary of what the function computes.",
                },
                "pseudocode": {
                    "type": "string",
                    "description": "Step-by-step algorithmic skeleton decoupled from surface variable names.",
                },
                "invariants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Mathematical invariants (e.g. ['det J == 1', 'shift invariance LSE(x-c)+c']).",
                },
                "teleological_intent": {
                    "type": "string",
                    "description": "Physical conservation law or scientific why behind the transformation.",
                },
                "dimensional_units": {
                    "type": "string",
                    "description": "Dimensional unit signature (e.g. 'c0: mg/L, ke: 1/hr, t: hr -> mg/L').",
                },
                "counterexample_block_id": {
                    "type": "string",
                    "description": "Optional block_id to quarantine if reporting an unprobed (K+1)-th input failure.",
                },
                "counterexample_args": {
                    "type": "string",
                    "description": "Optional Python tuple literal of failing arguments for counterexample ratcheting.",
                },
            },
        },
    },
]


def execute_find_tool(
    arguments: dict[str, Any],
    store_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute the `find` MCP tool against the persistent OpenCommons store."""
    store = load_persistent_store(store_path)
    query = str(arguments.get("query", ""))
    raw_pt = arguments.get("param_types")
    param_types = tuple(str(x) for x in raw_pt) if isinstance(raw_pt, list) else None
    return_type = str(arguments["return_type"]) if arguments.get("return_type") else None

    io_examples = None
    if arguments.get("io_examples"):
        try:
            parsed_io = ast.literal_eval(str(arguments["io_examples"]))
            if isinstance(parsed_io, list):
                io_examples = parsed_io
        except Exception:
            io_examples = None

    modality: ServingMode = arguments.get("modality", "dual_card_with_pitfalls")
    top_k = int(arguments.get("top_k", 3))

    hits = store.find(
        query=query,
        param_types=param_types,
        return_type=return_type,
        io_examples=io_examples,
        top_k=top_k,
        include_bundle=True,
    )
    rendered_hits = []
    for h in hits:
        bundle_cards = [b.render_for_agent(mode=modality) for b in h.bundle_blocks]
        rendered_hits.append(
            {
                "block_id": h.block.block_id,
                "cid": h.block.cid,
                "sid": h.block.sid,
                "trust_tier": h.block.trust_tier,
                "score": h.score,
                "io_probe_verified": h.io_probe_verified,
                "latency_ms": h.latency_ms,
                "bundle_size": len(h.bundle_blocks),
                "rendered_card": "\n\n".join(bundle_cards),
            }
        )
    return {
        "store_concepts": len(store.canonical_block_by_cid),
        "hit_count": len(rendered_hits),
        "hits": rendered_hits,
    }


def execute_record_tool(
    arguments: dict[str, Any],
    store_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute the `record` MCP tool (ingest + 5-gate peer review or counterexample ratchet)."""
    store = load_persistent_store(store_path)

    # Check if this is a (K+1)-th input counterexample report
    if arguments.get("counterexample_block_id") and arguments.get("counterexample_args"):
        bid = str(arguments["counterexample_block_id"])
        failing_args = ast.literal_eval(str(arguments["counterexample_args"]))
        if not isinstance(failing_args, tuple):
            failing_args = (failing_args,)
        quarantined = store.report_counterexample(block_id=bid, failing_args=failing_args)
        save_persistent_store(store, store_path)
        return {
            "action": "report_counterexample",
            "block_id": bid,
            "quarantined": quarantined,
            "active_quarantined_count": len(store.quarantined_ids),
        }

    source_code = str(arguments.get("source_code") or arguments.get("code") or "").strip()
    if not source_code:
        return {"error": "source_code is required when recording a new concept block."}

    invariants_raw = arguments.get("invariants") or []
    invariants = tuple(str(x) for x in invariants_raw) if isinstance(invariants_raw, list) else ()

    block, fp_res, verdict = store.ingest(
        source_code=source_code,
        domain=str(arguments.get("domain", "general")),
        summary=str(arguments.get("summary", "")),
        pseudocode=str(arguments.get("pseudocode", "")),
        invariants=invariants,
        teleological_intent=str(arguments.get("teleological_intent", "")),
        dimensional_units=str(arguments.get("dimensional_units", "")),
        run_peer_review=True,
    )
    merged_store = save_persistent_store(store, store_path)
    passed = verdict.passed if verdict else True
    return {
        "action": "ingest",
        "status": "accepted" if passed else "quarantined",
        "block_id": block.block_id,
        "cid": block.cid,
        "sid": block.sid,
        "fp": block.fp,
        "trust_tier": block.trust_tier,
        "passed_peer_review": passed,
        "quarantine_reasons": list(verdict.reasons) if verdict else [],
        "elapsed_ms": fp_res.elapsed_ms,
        "total_active_concepts": len(merged_store.canonical_block_by_cid),
    }


def handle_mcp_request(
    req: dict[str, Any],
    store_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Process a single JSON-RPC 2.0 MCP request and return the response dict."""
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "notifications/initialized":
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "opencommons-mcp", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": MCP_TOOLS_SCHEMA},
        }

    if method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {}) or {}
        try:
            if tool_name == "find":
                res_payload = execute_find_tool(arguments, store_path=store_path)
            elif tool_name == "record":
                res_payload = execute_record_tool(arguments, store_path=store_path)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(res_payload, indent=2),
                        }
                    ],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": str(exc)})}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not supported: {method}"},
    }


handle_jsonrpc_request = handle_mcp_request


def run_stdio_mcp_server(store_path: Path | str | None = None) -> int:
    """Run the JSON-RPC 2.0 MCP server over stdin/stdout."""
    resolved_path = store_path or os.environ.get("OPENCOMMONS_STORE_PATH") or DEFAULT_STORE_PATH
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_mcp_request(req, store_path=resolved_path)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0


def main() -> int:
    return run_stdio_mcp_server()


if __name__ == "__main__":
    raise SystemExit(main())
