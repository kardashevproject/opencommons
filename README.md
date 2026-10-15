# OpenCommons: A Distributed, Meaning-Indexed Concept Commons for Accelerating Scientific and Algorithmic Discovery in AI Agents

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23391160.svg)](https://doi.org/10.5281/zenodo.23391160)
[![License: Code Apache-2.0](https://img.shields.io/badge/Code_License-Apache_2.0-blue.svg)](LICENSE)
[![License: Paper CC-BY-4.0](https://img.shields.io/badge/Paper_License-CC_BY_4.0-green.svg)](LICENSE-PAPER)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)

**OpenCommons** is a decentralized, behavior-indexed, and asynchronously peer-reviewed Concept Commons designed to convert transient multi-agent inference into cumulative, verified scientific and algorithmic knowledge.

- **Contact**: `SI@kardashevproject.org` | [kardashevproject.org](https://kardashevproject.org)
- **Paper & DOI**: [`paper/main.pdf`](paper/main.pdf) | [DOI: `10.5281/zenodo.23391160`](https://doi.org/10.5281/zenodo.23391160) ([LaTeX source](paper/main.tex))

---

## Core Architecture & Empirical Results

1. **Dual-Tier Meaning Indexing (`100.0%` Coverage in `0.48 ms`)**:
   - **Tier 1 (`sid`)**: Alpha-normalized, commutative-sorted Abstract Syntax Tree (AST) Merkle hash ($0.15\text{ ms}$) that provides $O(1)$ structural deduplication (`20.0x` peer-review compute savings under swarm concurrency).
   - **Tier 2 (`cid`)**: Stratified 16-probe canonical execution fingerprint ($0.33\text{ ms}$) that unifies structurally distinct algorithms computing identical mathematical mappings (`100.0%` cross-algorithm equivalence recall, `0.0%` mutant false-merge rate across `240` functions in `R-001`).
2. **Sub-10ms Hybrid Retrieval & 2.67x Token Compression (`R-002`)**:
   - Combines token/identifier scoring, type-signature filtering, numbered-family discrimination, and live sandboxed I/O probe execution to maintain `100.0%` Precision@1 at $N=5,000$ entries (`7.24 ms` latency), where lexical BM25 degrades to `32.5%`.
   - Serves concepts as **Dual Concept Cards** (algorithmic pseudocode + conservation-law `Intent` + dimensional `Units` + executable code + refuted pitfalls) or **Pseudocode Building Blocks** (`2.67x` token compression).
3. **5-Gate Asynchronous Peer Review, CRDT Gossip & Monotonic Counterexample Ratchet (`R-003`, `R-006`)**:
   - Quarantines `150/150` (`100.0%`) adversarial defects across 6 failure classes with `0/50` false quarantines.
   - **Monotonic Counterexample Ratchet**: When a swarm agent encounters a $(K+1)\text{-th}$ input failure in production, calling `record(failing_block_id=..., counterexample_args=...)` deterministically verifies the exception, quarantines the block across the CRDT semilattice, and promotes the next verified implementation (`98.0 percentage point` contagion reduction across 50 discovery waves).
4. **Live LLM Scientific Discovery & Multi-Wave Compositional Scaling (`R-004`, `R-005`)**:
   - Improves live `Qwen3.5-2B-Base` single-attempt Pass@1 from `50.0%` (isolated zero-shot) and `23.3%` (BM25 distractors) to `83.3%` (`87.5%` on multi-step compositional adaptation `S4`).
   - Reduces 5-wave cumulative synthesis cost by `12.67x` versus isolated rediscovery and `6.40x` versus unverified text memory (`R-005`).

---

## Large-Scale Swarm Harness & MCP Server Quickstart

### 1. Installation
```bash
pip install -e .
```

### 2. CLI Usage (`opencommons`)
Initialize and inspect the persistent, file-locked CRDT store (`~/.opencommons/store.json` by default):
```bash
# Seed canonical peer-reviewed scientific primitives
opencommons seed

# Inspect store metrics, peer-review deduplication stats, and quarantined blocks
opencommons status

# Query verified Dual Concept Cards from the terminal
opencommons find "numerically stable logsumexp" --modality dual_card_with_pitfalls

# Merge a remote peer's store snapshot via join-semilattice CRDT union
opencommons sync /path/to/peer_store.json
```

### 3. Minimal 2-Tool MCP Server (`opencommons-mcp`)
To eliminate the `1.399x` empty-library schema tax of verbose 18-tool protocols, `opencommons-mcp` exposes exactly **2 JSON-RPC 2.0 tools** (`find` and `record`) over `stdio`:
```bash
opencommons-mcp --store ~/.opencommons/store.json
```
- **`find`**: Queries verified concepts by intent, type signature, and optional live `io_examples`, returning topological dependency bundles in the requested modality (`dual_card_with_pitfalls`, `dual_card`, `pseudocode`, or `raw_code`).
- **`record`**: Ingests a newly synthesized Python function through the $O(1)$ `sid` deduplicator and 5-gate peer-review harness (with `teleological_intent` and `dimensional_units`), **or** triggers the Monotonic Counterexample Ratchet when `failing_block_id` and `counterexample_args` are supplied.

### 4. Integration with `kardashev-cli` (`kardashev` / `ksh` Local & Open-Weight Agent Swarms)
When running local or open-weight coding agents (`Gemma 4`, `Qwen`, `GLM-4`, `vLLM`, `Ollama`) via [`kardashev-cli`](https://github.com/kardashevproject/kardashev-cli), pass `--with-opencommons` to attach the shared Concept Commons automatically:
```bash
kardashev chat --preset gemma4-26b --with-opencommons
kardashev exec "Implement a compensated 1D heat diffusion step" --with-opencommons
```

---

## Reproducing All Paper Experiments & Verification Gates

```bash
# 1. Run unit tests (engine, CRDT persistence, counterexample ratchet, MCP server)
PYTHONPATH=src pytest -v

# 2. Run the 6 reproducible empirical benchmarks (R-001 through R-006)
PYTHONPATH=src python3 experiments/r001_ingestion_and_equivalence.py
PYTHONPATH=src python3 experiments/r002_retrieval_and_modalities.py
PYTHONPATH=src python3 experiments/r003_peer_review_and_crdt.py
PYTHONPATH=src python3 experiments/r005_multi_wave_discovery.py
PYTHONPATH=src python3 experiments/r006_stress_test_falsification.py

# 3. Compile paper/main.pdf and run the 6/6 automated paper integrity gate
tectonic -X compile --synctex paper/main.tex
```

---

## Licensing

This repository is dual-licensed to support both open-source software deployment and open-access scientific publishing:
- **Software, Engine, Benchmark Suite & Swarm Harness (`src/`, `experiments/`, `tests/`)**: Licensed under the **Apache License, Version 2.0** (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)).
- **Scientific Manuscript, Figures, Tables & Technical Reports (`paper/`, `runs/`)**: Licensed under the **Creative Commons Attribution 4.0 International License (`CC-BY-4.0`)** (see [`LICENSE-PAPER`](LICENSE-PAPER)).
