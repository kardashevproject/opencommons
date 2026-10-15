"""R-004: Live LLM Scientific & Algorithmic Discovery Across Concept-Serving Modalities.

Pre-registered experiment evaluating live open-weights language model generation
(Qwen3.5-2B-Base via Transformers 5.14.1 / PyTorch 2.13.0) across a 24-task
Scientific & Algorithmic Discovery Suite spanning 4 domains:
  - S1: Numerical & Physical Simulation (symplectic Verlet, Kahan compensated sum,
        LogSumExp stability, RK4 step, Thomas tridiagonal solver, Richardson extrapolation)
  - S2: Information Theory & Cryptography (GF(2^8) multiplication, CRC-8 polynomial,
        RLE with escape framing, Elias gamma bit-length, FNV-1a 32-bit, SipHash round)
  - S3: Quantitative Biology & Chemometrics (Hill-Langmuir isotherm, Michaelis-Menten
        competitive inhibition, Shannon-Wiener diversity with zero-count guard,
        Needleman-Wunsch linear gap score, Jukes-Cantor distance with saturation guard,
        Hardy-Weinberg chi-square with zero-allele guard)
  - S4: Compositional Multi-Step Adaptation (tasks where the target function adapts
        or composes a base concept—e.g. Weighted Kahan Sum, Damped Symplectic Leapfrog,
        Windowed LogSumExp Softmax, Cooperative Competitive Hill-Michaelis Flux,
        Framed CRC-8 Packet Verifier, Regularized Thomas Solver)

Compares 5 Concept-Serving Modalities:
  1. `isolated_zero_shot`: No commons access (task docstring + signature only)
  2. `distractor_bm25`: Standard lexical BM25 retrieval from a 500-concept pool
     containing near-miss distractors (e.g., Euler vs Verlet, naive sum vs Kahan)
  3. `pseudocode_building_block`: Serves OpenCommons compact algorithmic pseudocode +
     mathematical invariants only (~52 tokens)
  4. `raw_code_block`: Serves OpenCommons raw reference implementation only (~165 tokens)
  5. `opencommons_dual_card`: Serves OpenCommons Dual Concept Card (pseudocode +
     invariants + refuted pitfalls from async peer review + reference code),
     verified by live <5ms I/O probes.

All generated programs are evaluated in a sandboxed execution harness against hidden
scientific unit tests (including numerical stability and boundary-condition probes),
alongside exact log-likelihood scoring of canonical vs. defective implementations.
"""
from __future__ import annotations

import ast
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import transformers

from frontier_research.commons.store import OpenCommonsStore, PitfallWarning
from frontier_research.harness.prereg import ExperimentSpec, HypothesisSpec
from frontier_research.harness.runner import ResearchHarness
from frontier_research.harness.stats import audit_confound, bootstrap_ci, compare_arms, holm_bonferroni


MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"),
)


@dataclass
class ScientificTask:
    task_id: str
    domain: str  # S1, S2, S3, S4
    name: str
    signature: str
    docstring: str
    # The underlying concept stored in OpenCommons
    concept_name: str
    concept_pseudocode: str
    concept_invariants: List[str]
    concept_pitfalls: List[str]
    concept_raw_code: str
    # Distractor concept returned when BM25 matches surface keywords without I/O probes
    distractor_code: str
    # Probe inputs/outputs for OpenCommons retrieval of the underlying concept
    retrieval_probes: List[Tuple[Tuple[Any, ...], Any]]
    # Hidden scientific test suite for the generated target function
    test_fn: Callable[[Callable[..., Any]], bool]
    # Canonical target body (for exact log-likelihood evaluation) and common pitfall body
    canonical_body: str
    pitfall_body: str


def build_scientific_suite() -> List[ScientificTask]:
    tasks: List[ScientificTask] = []

    # =========================================================================
    # S1: Numerical & Physical Simulation (6 tasks)
    # =========================================================================
    tasks.append(ScientificTask(
        task_id="S1_01", domain="S1", name="kahan_sum",
        signature="def kahan_sum(xs: list) -> float:",
        docstring="Compute compensated Kahan summation of float list xs to eliminate O(N*eps) floating-point drift.",
        concept_name="concept_kahan_compensated_sum",
        concept_pseudocode="1. Init total = 0.0, comp = 0.0\n2. For x in xs:\n   y = x - comp\n   t = total + y\n   comp = (t - total) - y\n   total = t\n3. Return total",
        concept_invariants=["Error bound |sum - exact| <= 2*eps independent of N", "Preserves low-order bits in comp = (t - total) - y"],
        concept_pitfalls=["REFUTED PITFALL: Do not simplify `(t - total) - y` to `0.0` or use naive `sum(xs)` which loses precision on [1e16, 1.0, -1e16]."],
        concept_raw_code="def ref_kahan(xs: list) -> float:\n    # Verbose reference implementation of Kahan summation\n    # Maintains running compensation variable c for lost low-order bits\n    total = 0.0\n    comp = 0.0\n    for x in xs:\n        y = x - comp\n        t = total + y\n        comp = (t - total) - y\n        total = t\n    return total",
        distractor_code="def naive_float_sum(xs: list) -> float:\n    # Fast summation of float list xs\n    total = 0.0\n    for x in xs:\n        total = total + x\n    return total",
        retrieval_probes=[(([1.0, 2.0, 3.0],), 6.0), (([1e16, 1.0, 1.0, -1e16],), 2.0)],
        test_fn=lambda fn: abs(fn([1e16, 1.0, 1.0, -1e16]) - 2.0) < 1e-9 and abs(fn([1.0, 2.5, 3.5]) - 7.0) < 1e-9,
        canonical_body="    total = 0.0\n    comp = 0.0\n    for x in xs:\n        y = x - comp\n        t = total + y\n        comp = (t - total) - y\n        total = t\n    return total\n",
        pitfall_body="    total = 0.0\n    for x in xs:\n        total += x\n    return total\n",
    ))

    tasks.append(ScientificTask(
        task_id="S1_02", domain="S1", name="symplectic_euler_step",
        signature="def symplectic_euler_step(q: float, p: float, dt: float, k: float) -> tuple:",
        docstring="Advance 1D harmonic oscillator (H = p^2/2 + k*q^2/2) by one semi-implicit symplectic Euler step (q_next, p_next) preserving phase-space volume.",
        concept_name="concept_symplectic_euler",
        concept_pseudocode="1. Update momentum first: p_next = p - k * q * dt\n2. Update position using NEW momentum: q_next = q + p_next * dt\n3. Return (q_next, p_next)",
        concept_invariants=["Jacobian determinant det(d(q_next, p_next)/d(q, p)) == 1.0", "Uses updated p_next in q_next update"],
        concept_pitfalls=["REFUTED PITFALL: Explicit Euler `q_next = q + p * dt; p_next = p - k * q * dt` has Jacobian det = 1 + k*dt^2 > 1 and spirals outward."],
        concept_raw_code="def ref_symplectic_euler(q: float, p: float, dt: float, k: float) -> tuple:\n    # Semi-implicit Euler integrator for Hamiltonian system\n    p_next = p - k * q * dt\n    q_next = q + p_next * dt\n    return (q_next, p_next)",
        distractor_code="def explicit_euler_step(q: float, p: float, dt: float, k: float) -> tuple:\n    # Standard explicit Euler step for harmonic oscillator\n    q_next = q + p * dt\n    p_next = p - k * q * dt\n    return (q_next, p_next)",
        retrieval_probes=[((1.0, 0.0, 0.1, 2.0), (0.98, -0.2)), ((0.0, 1.0, 0.5, 4.0), (0.5, 1.0))],
        test_fn=lambda fn: abs(fn(1.0, 0.0, 0.1, 2.0)[0] - 0.98) < 1e-9 and abs(fn(1.0, 0.0, 0.1, 2.0)[1] - (-0.2)) < 1e-9,
        canonical_body="    p_next = p - k * q * dt\n    q_next = q + p_next * dt\n    return (q_next, p_next)\n",
        pitfall_body="    q_next = q + p * dt\n    p_next = p - k * q * dt\n    return (q_next, p_next)\n",
    ))

    tasks.append(ScientificTask(
        task_id="S1_03", domain="S1", name="stable_logsumexp",
        signature="def stable_logsumexp(xs: list) -> float:",
        docstring="Compute log(sum(exp(x_i))) for non-empty float list xs without overflow/underflow for |x_i| >= 1000.",
        concept_name="concept_logsumexp_shift",
        concept_pseudocode="1. Find m = max(xs)\n2. Compute s = sum(math.exp(x - m) for x in xs)\n3. Return m + math.log(s)",
        concept_invariants=["Shift invariance: LSE(x) = m + log(sum(exp(x - m)))", "No overflow for max(xs) = 1000.0"],
        concept_pitfalls=["REFUTED PITFALL: Direct `math.log(sum(math.exp(x) for x in xs))` raises OverflowError for x >= 710."],
        concept_raw_code="def ref_logsumexp(xs: list) -> float:\n    # Numerically stable LogSumExp via max-shift identity\n    m = max(xs)\n    total = 0.0\n    for x in xs:\n        total += math.exp(x - m)\n    return m + math.log(total)",
        distractor_code="def direct_logsumexp(xs: list) -> float:\n    # LogSumExp calculation\n    return math.log(sum(math.exp(x) for x in xs))",
        retrieval_probes=[(([0.0, 0.0],), math.log(2.0)), (([1000.0, 1000.0],), 1000.0 + math.log(2.0))],
        test_fn=lambda fn: abs(fn([1000.0, 1000.0]) - (1000.0 + math.log(2.0))) < 1e-9 and abs(fn([-1000.0, -1000.0]) - (-1000.0 + math.log(2.0))) < 1e-9,
        canonical_body="    m = max(xs)\n    return m + math.log(sum(math.exp(x - m) for x in xs))\n",
        pitfall_body="    return math.log(sum(math.exp(x) for x in xs))\n",
    ))

    tasks.append(ScientificTask(
        task_id="S1_04", domain="S1", name="richardson_derivative",
        signature="def richardson_derivative(y_m2: float, y_m1: float, y_p1: float, y_p2: float, h: float) -> float:",
        docstring="Compute O(h^4) Richardson-extrapolated first derivative from central values f(x-2h), f(x-h), f(x+h), f(x+2h).",
        concept_name="concept_richardson_o4_diff",
        concept_pseudocode="1. D1 = (y_p1 - y_m1) / (2 * h)\n2. D2 = (y_p2 - y_m2) / (4 * h)\n3. Return (4 * D1 - D2) / 3 == (y_m2 - 8*y_m1 + 8*y_p1 - y_p2) / (12 * h)",
        concept_invariants=["Cancels O(h^2) Taylor error term via (4*D(h) - D(2h))/3", "Exact for cubic polynomials"],
        concept_pitfalls=["REFUTED PITFALL: Standard central difference `(y_p1 - y_m1) / (2 * h)` is only O(h^2) and fails cubic exactness."],
        concept_raw_code="def ref_richardson_diff(y_m2: float, y_m1: float, y_p1: float, y_p2: float, h: float) -> float:\n    # Fourth-order five-point stencil via Richardson extrapolation\n    return (y_m2 - 8.0 * y_m1 + 8.0 * y_p1 - y_p2) / (12.0 * h)",
        distractor_code="def central_diff(y_m2: float, y_m1: float, y_p1: float, y_p2: float, h: float) -> float:\n    # Central finite difference derivative\n    return (y_p1 - y_m1) / (2.0 * h)",
        retrieval_probes=[((-8.0, -1.0, 1.0, 8.0, 1.0), 0.0), ((-1.0, 0.0, 8.0, 27.0, 1.0), 3.0)],
        test_fn=lambda fn: abs(fn(-8.0, -1.0, 1.0, 8.0, 1.0) - 0.0) < 1e-9 and abs(fn(-1.0, 0.0, 8.0, 27.0, 1.0) - 3.0) < 1e-9,
        canonical_body="    return (y_m2 - 8.0 * y_m1 + 8.0 * y_p1 - y_p2) / (12.0 * h)\n",
        pitfall_body="    return (y_p1 - y_m1) / (2.0 * h)\n",
    ))

    tasks.append(ScientificTask(
        task_id="S1_05", domain="S1", name="welford_sample_variance",
        signature="def welford_sample_variance(xs: list) -> float:",
        docstring="Compute unbiased sample variance (Bessel-corrected N-1 denominator) of float list xs in a single pass using Welford's online algorithm; return 0.0 if len(xs) < 2.",
        concept_name="concept_welford_variance",
        concept_pseudocode="1. If len(xs) < 2: return 0.0\n2. mean = 0.0, m2 = 0.0, n = 0\n3. For x in xs: n += 1; delta = x - mean; mean += delta / n; m2 += delta * (x - mean)\n4. Return m2 / (n - 1)",
        concept_invariants=["Immune to catastrophic cancellation on large-offset data (e.g., 1e9 + x)", "Uses pre-update delta times post-update (x - mean)"],
        concept_pitfalls=["REFUTED PITFALL: Two-pass or `E[X^2] - E[X]^2` fails on shifted inputs `[1e9+1, 1e9+2, 1e9+3]`, and population denominator `n` fails unbiased Bessel correction."],
        concept_raw_code="def ref_welford(xs: list) -> float:\n    if len(xs) < 2:\n        return 0.0\n    mean = 0.0\n    m2 = 0.0\n    for i, x in enumerate(xs, 1):\n        delta = x - mean\n        mean += delta / i\n        m2 += delta * (x - mean)\n    return m2 / (len(xs) - 1)",
        distractor_code="def naive_variance(xs: list) -> float:\n    if len(xs) < 2:\n        return 0.0\n    m = sum(xs) / len(xs)\n    return sum((x - m) ** 2 for x in xs) / len(xs)",
        retrieval_probes=[(([1.0, 2.0, 3.0],), 1.0), (([1e9 + 1.0, 1e9 + 2.0, 1e9 + 3.0],), 1.0)],
        test_fn=lambda fn: abs(fn([1e9 + 1.0, 1e9 + 2.0, 1e9 + 3.0]) - 1.0) < 1e-6 and fn([5.0]) == 0.0,
        canonical_body="    if len(xs) < 2:\n        return 0.0\n    mean = 0.0\n    m2 = 0.0\n    for i, x in enumerate(xs, 1):\n        delta = x - mean\n        mean += delta / i\n        m2 += delta * (x - mean)\n    return m2 / (len(xs) - 1)\n",
        pitfall_body="    if not xs:\n        return 0.0\n    m = sum(xs) / len(xs)\n    return sum((x - m) ** 2 for x in xs) / len(xs)\n",
    ))

    tasks.append(ScientificTask(
        task_id="S1_06", domain="S1", name="hypotenuse_stable",
        signature="def hypotenuse_stable(a: float, b: float) -> float:",
        docstring="Compute sqrt(a^2 + b^2) without intermediate overflow or underflow for |a|, |b| up to 1e300.",
        concept_name="concept_stable_hypot",
        concept_pseudocode="1. x = abs(a), y = abs(b)\n2. If x < y: swap(x, y)\n3. If x == 0.0: return 0.0\n4. r = y / x; return x * math.sqrt(1.0 + r * r)",
        concept_invariants=["Ratio r = min(|a|,|b|)/max(|a|,|b|) in [0, 1]", "No overflow when max(|a|,|b|) = 1e300"],
        concept_pitfalls=["REFUTED PITFALL: Direct `math.sqrt(a*a + b*b)` overflows to inf when a = 3e200, b = 4e200."],
        concept_raw_code="def ref_hypot(a: float, b: float) -> float:\n    x, y = abs(a), abs(b)\n    if x < y:\n        x, y = y, x\n    if x == 0.0:\n        return 0.0\n    r = y / x\n    return x * math.sqrt(1.0 + r * r)",
        distractor_code="def naive_hypot(a: float, b: float) -> float:\n    return math.sqrt(a * a + b * b)",
        retrieval_probes=[((3.0, 4.0), 5.0), ((3e200, 4e200), 5e200)],
        test_fn=lambda fn: abs(fn(3e200, 4e200) - 5e200) / 5e200 < 1e-9 and fn(0.0, 0.0) == 0.0,
        canonical_body="    x, y = abs(a), abs(b)\n    if x < y:\n        x, y = y, x\n    if x == 0.0:\n        return 0.0\n    r = y / x\n    return x * math.sqrt(1.0 + r * r)\n",
        pitfall_body="    return math.sqrt(a * a + b * b)\n",
    ))

    # =========================================================================
    # S2: Information Theory & Cryptography (6 tasks)
    # =========================================================================
    tasks.append(ScientificTask(
        task_id="S2_01", domain="S2", name="gf256_mul",
        signature="def gf256_mul(a: int, b: int) -> int:",
        docstring="Multiply two bytes a, b in Rijndael GF(2^8) with irreducible polynomial x^8 + x^4 + x^3 + x + 1 (0x11B).",
        concept_name="concept_gf256_rijndael_mul",
        concept_pseudocode="1. p = 0\n2. Loop 8 times:\n   If b & 1: p ^= a\n   hi = a & 0x80\n   a = (a << 1) & 0xFF\n   If hi: a ^= 0x1B\n   b >>= 1\n3. Return p",
        concept_invariants=["Carryless XOR addition and reduction modulo 0x11B", "Output strictly in 0..255"],
        concept_pitfalls=["REFUTED PITFALL: Standard integer `(a * b) % 283` uses arithmetic carry instead of polynomial XOR."],
        concept_raw_code="def ref_gf256_mul(a: int, b: int) -> int:\n    p = 0\n    for _ in range(8):\n        if b & 1:\n            p ^= a\n        hi = a & 0x80\n        a = (a << 1) & 0xFF\n        if hi:\n            a ^= 0x1B\n        b >>= 1\n    return p",
        distractor_code="def mod256_mul(a: int, b: int) -> int:\n    return (a * b) % 283",
        retrieval_probes=[((0x57, 0x83), 0xC1), ((0x57, 0x13), 0xFE)],
        test_fn=lambda fn: fn(0x57, 0x83) == 0xC1 and fn(0x57, 0x13) == 0xFE and fn(0x01, 0xAB) == 0xAB,
        canonical_body="    p = 0\n    for _ in range(8):\n        if b & 1:\n            p ^= a\n        hi = a & 0x80\n        a = (a << 1) & 0xFF\n        if hi:\n            a ^= 0x1B\n        b >>= 1\n    return p\n",
        pitfall_body="    return (a * b) % 283\n",
    ))

    tasks.append(ScientificTask(
        task_id="S2_02", domain="S2", name="crc8_ccitt",
        signature="def crc8_ccitt(data: list) -> int:",
        docstring="Compute CRC-8-CCITT checksum (polynomial 0x07, initial value 0x00, MSB-first) over a list of byte integers.",
        concept_name="concept_crc8_ccitt",
        concept_pseudocode="1. crc = 0x00\n2. For byte in data:\n   crc ^= (byte & 0xFF)\n   Repeat 8 times:\n     crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF\n3. Return crc",
        concept_invariants=["MSB-first shift with 0x07 polynomial XOR when bit 7 is set", "crc8([0x01]) == 0x07 and crc8([0x31, 0x32]) == 0x72"],
        concept_pitfalls=["REFUTED PITFALL: LSB-first shifting (`crc >> 1`) or forgetting `& 0xFF` mask corrupts the remainder."],
        concept_raw_code="def ref_crc8(data: list) -> int:\n    crc = 0x00\n    for b in data:\n        crc ^= (b & 0xFF)\n        for _ in range(8):\n            if crc & 0x80:\n                crc = ((crc << 1) ^ 0x07) & 0xFF\n            else:\n                crc = (crc << 1) & 0xFF\n    return crc",
        distractor_code="def xor_checksum(data: list) -> int:\n    crc = 0\n    for b in data:\n        crc = (crc ^ b) & 0xFF\n    return crc",
        retrieval_probes=[(([0x01],), 0x07), (([0x31, 0x32],), 0x72)],
        test_fn=lambda fn: fn([0x01]) == 0x07 and fn([0x31, 0x32]) == 0x72 and fn([]) == 0,
        canonical_body="    crc = 0x00\n    for b in data:\n        crc ^= (b & 0xFF)\n        for _ in range(8):\n            if crc & 0x80:\n                crc = ((crc << 1) ^ 0x07) & 0xFF\n            else:\n                crc = (crc << 1) & 0xFF\n    return crc\n",
        pitfall_body="    crc = 0\n    for b in data:\n        crc = (crc ^ b) & 0xFF\n    return crc\n",
    ))

    tasks.append(ScientificTask(
        task_id="S2_03", domain="S2", name="fnv1a_32",
        signature="def fnv1a_32(data: list) -> int:",
        docstring="Compute 32-bit FNV-1a hash (offset basis 0x811C9DC5, prime 0x01000193) over a list of byte integers.",
        concept_name="concept_fnv1a_32",
        concept_pseudocode="1. h = 0x811C9DC5\n2. For b in data:\n   h ^= (b & 0xFF)\n   h = (h * 0x01000193) & 0xFFFFFFFF\n3. Return h",
        concept_invariants=["FNV-1a XORs before multiplying (unlike FNV-1 which multiplies before XORing)", "Masked to 32-bit unsigned integer"],
        concept_pitfalls=["REFUTED PITFALL: FNV-1 order `h = ((h * prime) ^ b) & 0xFFFFFFFF` produces the wrong hash."],
        concept_raw_code="def ref_fnv1a(data: list) -> int:\n    h = 0x811C9DC5\n    for b in data:\n        h ^= (b & 0xFF)\n        h = (h * 0x01000193) & 0xFFFFFFFF\n    return h",
        distractor_code="def fnv1_32(data: list) -> int:\n    h = 0x811C9DC5\n    for b in data:\n        h = ((h * 0x01000193) ^ (b & 0xFF)) & 0xFFFFFFFF\n    return h",
        retrieval_probes=[(([],), 0x811C9DC5), (([0x61],), 0xE40C292C)],
        test_fn=lambda fn: fn([]) == 0x811C9DC5 and fn([0x61]) == 0xE40C292C and fn([0x66, 0x6F, 0x6F]) == 0xA9F37ED7,
        canonical_body="    h = 0x811C9DC5\n    for b in data:\n        h ^= (b & 0xFF)\n        h = (h * 0x01000193) & 0xFFFFFFFF\n    return h\n",
        pitfall_body="    h = 0x811C9DC5\n    for b in data:\n        h = ((h * 0x01000193) ^ (b & 0xFF)) & 0xFFFFFFFF\n    return h\n",
    ))

    tasks.append(ScientificTask(
        task_id="S2_04", domain="S2", name="elias_gamma_bitlen",
        signature="def elias_gamma_bitlen(n: int) -> int:",
        docstring="Return the exact bit length of the Elias gamma universal code for positive integer n >= 1 (return 0 if n <= 0).",
        concept_name="concept_elias_gamma_len",
        concept_pseudocode="1. If n <= 0: return 0\n2. k = n.bit_length() - 1\n3. Return 2 * k + 1",
        concept_invariants=["Elias gamma prefixes k = floor(log2(n)) zeros before the (k+1)-bit binary representation", "Length is 2*floor(log2(n)) + 1"],
        concept_pitfalls=["REFUTED PITFALL: Returning `n.bit_length()` omits the unary prefix of floor(log2(n)) zeros."],
        concept_raw_code="def ref_elias_gamma_len(n: int) -> int:\n    if n <= 0:\n        return 0\n    return 2 * (n.bit_length() - 1) + 1",
        distractor_code="def binary_bitlen(n: int) -> int:\n    if n <= 0:\n        return 0\n    return n.bit_length()",
        retrieval_probes=[((1,), 1), ((4,), 5), ((9,), 7)],
        test_fn=lambda fn: fn(1) == 1 and fn(4) == 5 and fn(9) == 7 and fn(0) == 0,
        canonical_body="    if n <= 0:\n        return 0\n    return 2 * (n.bit_length() - 1) + 1\n",
        pitfall_body="    if n <= 0:\n        return 0\n    return n.bit_length()\n",
    ))

    tasks.append(ScientificTask(
        task_id="S2_05", domain="S2", name="binary_entropy_bits",
        signature="def binary_entropy_bits(p: float) -> float:",
        docstring="Compute Shannon binary entropy H_2(p) = -p*log2(p) - (1-p)*log2(1-p) in bits, handling boundary limits p=0.0 and p=1.0 via limit 0*log2(0) = 0.0.",
        concept_name="concept_binary_entropy",
        concept_pseudocode="1. If p <= 0.0 or p >= 1.0: return 0.0\n2. Return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))",
        concept_invariants=["H_2(0.0) == 0.0 and H_2(1.0) == 0.0 without ValueError", "H_2(0.5) == 1.0 bit"],
        concept_pitfalls=["REFUTED PITFALL: Calling `math.log2(p)` directly without guarding `p <= 0.0 or p >= 1.0` raises ValueError: math domain error."],
        concept_raw_code="def ref_binary_entropy(p: float) -> float:\n    if p <= 0.0 or p >= 1.0:\n        return 0.0\n    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))",
        distractor_code="def unguarded_entropy(p: float) -> float:\n    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))",
        retrieval_probes=[((0.5,), 1.0), ((0.0,), 0.0), ((1.0,), 0.0)],
        test_fn=lambda fn: abs(fn(0.5) - 1.0) < 1e-9 and fn(0.0) == 0.0 and fn(1.0) == 0.0,
        canonical_body="    if p <= 0.0 or p >= 1.0:\n        return 0.0\n    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))\n",
        pitfall_body="    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))\n",
    ))

    tasks.append(ScientificTask(
        task_id="S2_06", domain="S2", name="zigzag_encode_int",
        signature="def zigzag_encode_int(n: int) -> int:",
        docstring="Encode signed integer n into non-negative integer using Protocol Buffers ZigZag mapping (0->0, -1->1, 1->2, -2->3, 2->4).",
        concept_name="concept_zigzag_encode",
        concept_pseudocode="1. If n >= 0: return 2 * n\n2. Else: return -2 * n - 1",
        concept_invariants=["Bijection Z -> N mapping small-magnitude negative numbers to small odd integers", "zigzag(-1) == 1, zigzag(1) == 2"],
        concept_pitfalls=["REFUTED PITFALL: Using `(n << 1) ^ (n >> 31)` in Python fails for arbitrary-precision negative ints unless arithmetic sign check is used."],
        concept_raw_code="def ref_zigzag(n: int) -> int:\n    return 2 * n if n >= 0 else -2 * n - 1",
        distractor_code="def buggy_c_zigzag(n: int) -> int:\n    return (n << 1) ^ (n >> 31) & 0xFFFFFFFF",
        retrieval_probes=[((0,), 0), ((-1,), 1), ((1,), 2), ((-2,), 3)],
        test_fn=lambda fn: fn(0) == 0 and fn(-1) == 1 and fn(1) == 2 and fn(-2) == 3 and fn(2) == 4,
        canonical_body="    return 2 * n if n >= 0 else -2 * n - 1\n",
        pitfall_body="    return 2 * n if n >= 0 else -2 * n + 1\n",
    ))

    # =========================================================================
    # S3: Quantitative Biology & Chemometrics (6 tasks)
    # =========================================================================
    tasks.append(ScientificTask(
        task_id="S3_01", domain="S3", name="hill_langmuir_occupancy",
        signature="def hill_langmuir_occupancy(ligand: float, kd: float, n: float) -> float:",
        docstring="Compute cooperative receptor fractional occupancy theta in [0, 1] using the Hill-Langmuir equation where kd is the apparent dissociation constant (concentration at half-maximal occupancy): theta = L^n / (kd^n + L^n). Return 0.0 if ligand <= 0.0.",
        concept_name="concept_hill_langmuir_kd",
        concept_pseudocode="1. If ligand <= 0.0: return 0.0\n2. ln = ligand ** n; kn = kd ** n\n3. Return ln / (kn + ln)",
        concept_invariants=["At ligand == kd, occupancy is exactly 0.5 for all Hill coefficients n > 0", "Zero-ligand guard returns 0.0"],
        concept_pitfalls=["REFUTED PITFALL: Writing `ligand**n / (kd + ligand**n)` without raising `kd` to power `n` violates theta(kd) == 0.5."],
        concept_raw_code="def ref_hill_langmuir(ligand: float, kd: float, n: float) -> float:\n    if ligand <= 0.0:\n        return 0.0\n    ln = ligand ** n\n    kn = kd ** n\n    return ln / (kn + ln)",
        distractor_code="def hill_ka_form(ligand: float, kd: float, n: float) -> float:\n    return (ligand ** n) / (kd + (ligand ** n))",
        retrieval_probes=[((2.0, 2.0, 3.0), 0.5), ((0.0, 2.0, 2.0), 0.0)],
        test_fn=lambda fn: abs(fn(2.0, 2.0, 3.0) - 0.5) < 1e-9 and fn(0.0, 2.0, 2.0) == 0.0 and abs(fn(4.0, 2.0, 2.0) - 0.8) < 1e-9,
        canonical_body="    if ligand <= 0.0:\n        return 0.0\n    ln = ligand ** n\n    kn = kd ** n\n    return ln / (kn + ln)\n",
        pitfall_body="    return (ligand ** n) / (kd + ligand ** n)\n",
    ))

    tasks.append(ScientificTask(
        task_id="S3_02", domain="S3", name="competitive_michaelis_menten",
        signature="def competitive_michaelis_menten(vmax: float, s: float, km: float, inhibitor: float, ki: float) -> float:",
        docstring="Compute enzymatic reaction velocity v under reversible competitive inhibition (vmax, substrate s, km, inhibitor concentration, inhibition constant ki).",
        concept_name="concept_competitive_mm",
        concept_pseudocode="1. If s <= 0.0: return 0.0\n2. km_app = km * (1.0 + inhibitor / ki)\n3. Return (vmax * s) / (km_app + s)",
        concept_invariants=["Competitive inhibition scales Km by alpha = (1 + I/Ki) without changing Vmax", "v(s=0) == 0.0"],
        concept_pitfalls=["REFUTED PITFALL: Dividing Vmax by `(1 + inhibitor/ki)` models non-competitive inhibition rather than competitive inhibition."],
        concept_raw_code="def ref_competitive_mm(vmax: float, s: float, km: float, inhibitor: float, ki: float) -> float:\n    if s <= 0.0:\n        return 0.0\n    km_app = km * (1.0 + inhibitor / ki)\n    return (vmax * s) / (km_app + s)",
        distractor_code="def noncompetitive_mm(vmax: float, s: float, km: float, inhibitor: float, ki: float) -> float:\n    return (vmax / (1.0 + inhibitor / ki)) * s / (km + s)",
        retrieval_probes=[((10.0, 4.0, 2.0, 2.0, 2.0), 5.0), ((10.0, 0.0, 2.0, 1.0, 1.0), 0.0)],
        test_fn=lambda fn: abs(fn(10.0, 4.0, 2.0, 2.0, 2.0) - 5.0) < 1e-9 and fn(10.0, 0.0, 2.0, 1.0, 1.0) == 0.0,
        canonical_body="    if s <= 0.0:\n        return 0.0\n    km_app = km * (1.0 + inhibitor / ki)\n    return (vmax * s) / (km_app + s)\n",
        pitfall_body="    return (vmax / (1.0 + inhibitor / ki)) * s / (km + s)\n",
    ))

    tasks.append(ScientificTask(
        task_id="S3_03", domain="S3", name="jukes_cantor_distance",
        signature="def jukes_cantor_distance(p: float) -> float:",
        docstring="Compute Jukes-Cantor evolutionary nucleotide distance d = -0.75 * ln(1 - 4*p/3) from observed mismatch fraction p; return -1.0 if saturated (p >= 0.75) and 0.0 if p <= 0.0.",
        concept_name="concept_jukes_cantor",
        concept_pseudocode="1. If p <= 0.0: return 0.0\n2. If p >= 0.75: return -1.0\n3. Return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)",
        concept_invariants=["Saturation guard at p >= 0.75 prevents math.log(<=0) ValueError", "Zero mismatch returns 0.0"],
        concept_pitfalls=["REFUTED PITFALL: Evaluating `math.log(1 - 4*p/3)` without checking `p >= 0.75` crashes on saturated sequences."],
        concept_raw_code="def ref_jukes_cantor(p: float) -> float:\n    if p <= 0.0:\n        return 0.0\n    if p >= 0.75:\n        return -1.0\n    return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)",
        distractor_code="def unguarded_jc(p: float) -> float:\n    return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)",
        retrieval_probes=[((0.0,), 0.0), ((0.75,), -1.0), ((0.375,), -0.75 * math.log(0.5))],
        test_fn=lambda fn: fn(0.0) == 0.0 and fn(0.75) == -1.0 and fn(0.8) == -1.0 and abs(fn(0.375) - (-0.75 * math.log(0.5))) < 1e-9,
        canonical_body="    if p <= 0.0:\n        return 0.0\n    if p >= 0.75:\n        return -1.0\n    return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)\n",
        pitfall_body="    return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)\n",
    ))

    tasks.append(ScientificTask(
        task_id="S3_04", domain="S3", name="shannon_wiener_diversity",
        signature="def shannon_wiener_diversity(counts: list) -> float:",
        docstring="Compute ecological Shannon-Wiener diversity index H' = -sum(p_i * ln(p_i)) from species abundance counts, skipping zero counts and returning 0.0 if total count <= 0.",
        concept_name="concept_shannon_wiener",
        concept_pseudocode="1. total = sum(c for c in counts if c > 0)\n2. If total <= 0: return 0.0\n3. Return -sum((c / total) * math.log(c / total) for c in counts if c > 0)",
        concept_invariants=["Skips zero abundances c == 0 to avoid math.log(0)", "Uses natural logarithm ln"],
        concept_pitfalls=["REFUTED PITFALL: Iterating over all `c in counts` without `if c > 0` raises ValueError when a species count is 0."],
        concept_raw_code="def ref_shannon_wiener(counts: list) -> float:\n    pos = [c for c in counts if c > 0]\n    total = sum(pos)\n    if total <= 0:\n        return 0.0\n    return -sum((c / total) * math.log(c / total) for c in pos)",
        distractor_code="def buggy_shannon(counts: list) -> float:\n    total = sum(counts)\n    return -sum((c / total) * math.log(c / total) for c in counts)",
        retrieval_probes=[(([10, 0, 10],), math.log(2.0)), (([0, 0],), 0.0)],
        test_fn=lambda fn: abs(fn([10, 0, 10]) - math.log(2.0)) < 1e-9 and fn([0, 0]) == 0.0 and fn([5]) == 0.0,
        canonical_body="    pos = [c for c in counts if c > 0]\n    total = sum(pos)\n    if total <= 0:\n        return 0.0\n    return -sum((c / total) * math.log(c / total) for c in pos)\n",
        pitfall_body="    total = sum(counts)\n    return -sum((c / total) * math.log(c / total) for c in counts)\n",
    ))

    tasks.append(ScientificTask(
        task_id="S3_05", domain="S3", name="hardy_weinberg_chisq",
        signature="def hardy_weinberg_chisq(n_aa: int, n_ab: int, n_bb: int) -> float:",
        docstring="Compute Hardy-Weinberg equilibrium Pearson chi-square statistic from genotype counts (n_aa, n_ab, n_bb); return 0.0 if total N == 0 or either allele frequency p, q == 0.",
        concept_name="concept_hw_chisq",
        concept_pseudocode="1. n = n_aa + n_ab + n_bb; if n <= 0: return 0.0\n2. p = (2 * n_aa + n_ab) / (2 * n); q = 1.0 - p\n3. If p <= 0.0 or q <= 0.0: return 0.0\n4. e_aa = n*p*p; e_ab = 2*n*p*q; e_bb = n*q*q\n5. Return (n_aa-e_aa)**2/e_aa + (n_ab-e_ab)**2/e_ab + (n_bb-e_bb)**2/e_bb",
        concept_invariants=["Guards monomorphic populations (p==0 or q==0) against ZeroDivisionError", "Returns 0.0 at exact HWE proportions (25, 50, 25)"],
        concept_pitfalls=["REFUTED PITFALL: Computing `(n_bb - e_bb)**2 / e_bb` without checking `q == 0` raises ZeroDivisionError on monomorphic `(10, 0, 0)`."],
        concept_raw_code="def ref_hw_chisq(n_aa: int, n_ab: int, n_bb: int) -> float:\n    n = n_aa + n_ab + n_bb\n    if n <= 0:\n        return 0.0\n    p = (2.0 * n_aa + n_ab) / (2.0 * n)\n    q = 1.0 - p\n    if p <= 0.0 or q <= 0.0:\n        return 0.0\n    e_aa, e_ab, e_bb = n * p * p, 2.0 * n * p * q, n * q * q\n    return ((n_aa - e_aa) ** 2) / e_aa + ((n_ab - e_ab) ** 2) / e_ab + ((n_bb - e_bb) ** 2) / e_bb",
        distractor_code="def unguarded_hw(n_aa: int, n_ab: int, n_bb: int) -> float:\n    n = n_aa + n_ab + n_bb\n    p = (2.0 * n_aa + n_ab) / (2.0 * n)\n    q = 1.0 - p\n    return ((n_aa - n*p*p)**2)/(n*p*p) + ((n_ab - 2*n*p*q)**2)/(2*n*p*q) + ((n_bb - n*q*q)**2)/(n*q*q)",
        retrieval_probes=[((25, 50, 25), 0.0), ((10, 0, 0), 0.0)],
        test_fn=lambda fn: abs(fn(25, 50, 25) - 0.0) < 1e-9 and fn(10, 0, 0) == 0.0 and fn(0, 0, 0) == 0.0 and abs(fn(40, 20, 40) - 36.0) < 1e-6,
        canonical_body="    n = n_aa + n_ab + n_bb\n    if n <= 0:\n        return 0.0\n    p = (2.0 * n_aa + n_ab) / (2.0 * n)\n    q = 1.0 - p\n    if p <= 0.0 or q <= 0.0:\n        return 0.0\n    e_aa, e_ab, e_bb = n * p * p, 2.0 * n * p * q, n * q * q\n    return ((n_aa - e_aa) ** 2) / e_aa + ((n_ab - e_ab) ** 2) / e_ab + ((n_bb - e_bb) ** 2) / e_bb\n",
        pitfall_body="    n = n_aa + n_ab + n_bb\n    p = (2.0 * n_aa + n_ab) / (2.0 * n)\n    q = 1.0 - p\n    return ((n_aa - n*p*p)**2)/(n*p*p) + ((n_ab - 2*n*p*q)**2)/(2*n*p*q) + ((n_bb - n*q*q)**2)/(n*q*q)\n",
    ))

    tasks.append(ScientificTask(
        task_id="S3_06", domain="S3", name="beer_lambert_transmittance",
        signature="def beer_lambert_transmittance(epsilon: float, conc: float, path_cm: float) -> float:",
        docstring="Compute optical transmittance fraction T = 10^(-epsilon * conc * path_cm) in (0, 1] from decadic molar attenuation coefficient epsilon, concentration conc, and path length.",
        concept_name="concept_beer_lambert_decadic",
        concept_pseudocode="1. absorbance = epsilon * conc * path_cm\n2. Return 10.0 ** (-absorbance)",
        concept_invariants=["Chemical Beer-Lambert absorbance A = -log10(T) uses base-10, not natural base e", "T(A=1.0) == 0.1"],
        concept_pitfalls=["REFUTED PITFALL: Using `math.exp(-epsilon * conc * path_cm)` computes Naperian transmittance instead of decadic chemical transmittance."],
        concept_raw_code="def ref_beer_lambert(epsilon: float, conc: float, path_cm: float) -> float:\n    return 10.0 ** (-(epsilon * conc * path_cm))",
        distractor_code="def naperian_transmittance(epsilon: float, conc: float, path_cm: float) -> float:\n    return math.exp(-(epsilon * conc * path_cm))",
        retrieval_probes=[((1.0, 1.0, 1.0), 0.1), ((2.0, 0.5, 2.0), 0.01)],
        test_fn=lambda fn: abs(fn(1.0, 1.0, 1.0) - 0.1) < 1e-9 and abs(fn(2.0, 0.5, 2.0) - 0.01) < 1e-9,
        canonical_body="    return 10.0 ** (-(epsilon * conc * path_cm))\n",
        pitfall_body="    return math.exp(-(epsilon * conc * path_cm))\n",
    ))

    # =========================================================================
    # S4: Compositional Multi-Step Adaptation (12 tasks)
    # Here the target function ADAPTS or COMPOSES the stored concept!
    # Raw code copy-pasting fails if the agent/model blindly copies the retrieved body,
    # whereas pseudocode + invariants / Dual Cards guide structural adaptation.
    # =========================================================================
    tasks.append(ScientificTask(
        task_id="S4_01", domain="S4", name="weighted_kahan_dot",
        signature="def weighted_kahan_dot(xs: list, ws: list) -> float:",
        docstring="Adapt Kahan compensated summation to compute the inner product sum(x_i * w_i) of two equal-length float lists xs, ws without floating-point drift.",
        concept_name="concept_kahan_compensated_sum",
        concept_pseudocode="1. Init total = 0.0, comp = 0.0\n2. For term in stream (e.g. x * w):\n   y = term - comp\n   t = total + y\n   comp = (t - total) - y\n   total = t\n3. Return total",
        concept_invariants=["Apply Kahan step to each product term = x * w", "Error bound O(eps) independent of N"],
        concept_pitfalls=["REFUTED PITFALL: Do not copy single-list `for x in xs` without multiplying by `w`, and do not use uncompensated `sum(x*w)`."],
        concept_raw_code="def ref_kahan(xs: list) -> float:\n    total = 0.0\n    comp = 0.0\n    for x in xs:\n        y = x - comp\n        t = total + y\n        comp = (t - total) - y\n        total = t\n    return total",
        distractor_code="def naive_dot(xs: list, ws: list) -> float:\n    return sum(x * w for x, w in zip(xs, ws))",
        retrieval_probes=[(([1.0, 2.0, 3.0],), 6.0), (([1.0] + [1e-16]*10000,), 1.0 + 1e-12)],
        test_fn=lambda fn: abs((fn([1.0] + [1e-16] * 10000, [1.0] + [1.0] * 10000) - 1.0) - 1e-12) < 1e-15 and abs(fn([2.0, 4.0], [0.5, 0.25]) - 2.0) < 1e-9,
        canonical_body="    total = 0.0\n    comp = 0.0\n    for x, w in zip(xs, ws):\n        y = (x * w) - comp\n        t = total + y\n        comp = (t - total) - y\n        total = t\n    return total\n",
        pitfall_body="    return sum(x * w for x, w in zip(xs, ws))\n",
    ))

    tasks.append(ScientificTask(
        task_id="S4_02", domain="S4", name="damped_symplectic_step",
        signature="def damped_symplectic_step(q: float, p: float, dt: float, k: float, gamma: float) -> tuple:",
        docstring="Adapt semi-implicit symplectic Euler to include linear viscous damping gamma: p_next = p - (k*q + gamma*p)*dt, then q_next = q + p_next*dt.",
        concept_name="concept_symplectic_euler",
        concept_pseudocode="1. Update momentum first with total force F(q, p): p_next = p + F(q, p) * dt\n2. Update position using NEW momentum: q_next = q + p_next * dt\n3. Return (q_next, p_next)",
        concept_invariants=["Position update MUST use updated p_next, not old p", "Damping force is -gamma * p added to restoring force -k * q"],
        concept_pitfalls=["REFUTED PITFALL: Do not omit `- gamma * p * dt` from the momentum update or use old `p` in `q_next`."],
        concept_raw_code="def ref_symplectic_euler(q: float, p: float, dt: float, k: float) -> tuple:\n    p_next = p - k * q * dt\n    q_next = q + p_next * dt\n    return (q_next, p_next)",
        distractor_code="def explicit_damped(q: float, p: float, dt: float, k: float, gamma: float) -> tuple:\n    return (q + p * dt, p - (k * q + gamma * p) * dt)",
        retrieval_probes=[((1.0, 0.0, 0.1, 2.0), (0.98, -0.2)), ((0.0, 1.0, 0.5, 4.0), (0.5, 1.0))],
        test_fn=lambda fn: abs(fn(1.0, 2.0, 0.1, 2.0, 0.5)[1] - 1.7) < 1e-9 and abs(fn(1.0, 2.0, 0.1, 2.0, 0.5)[0] - 1.17) < 1e-9,
        canonical_body="    p_next = p - (k * q + gamma * p) * dt\n    q_next = q + p_next * dt\n    return (q_next, p_next)\n",
        pitfall_body="    q_next = q + p * dt\n    p_next = p - (k * q + gamma * p) * dt\n    return (q_next, p_next)\n",
    ))

    tasks.append(ScientificTask(
        task_id="S4_03", domain="S4", name="stable_softmax_prob",
        signature="def stable_softmax_prob(xs: list, idx: int) -> float:",
        docstring="Adapt the max-shifted LogSumExp trick to return the exact softmax probability exp(xs[idx]) / sum_j exp(xs[j]) without overflow for |xs[j]| >= 1000.",
        concept_name="concept_logsumexp_shift",
        concept_pseudocode="1. m = max(xs)\n2. lse = m + math.log(sum(math.exp(x - m) for x in xs))\n3. Softmax at idx: math.exp(xs[idx] - lse)",
        concept_invariants=["Subtract max(xs) before exponentiation so max exponent is 0.0", "Outputs probability in (0, 1]"],
        concept_pitfalls=["REFUTED PITFALL: Returning `lse` instead of `math.exp(xs[idx] - lse)` or computing unshifted `math.exp(xs[idx])` overflows."],
        concept_raw_code="def ref_logsumexp(xs: list) -> float:\n    m = max(xs)\n    return m + math.log(sum(math.exp(x - m) for x in xs))",
        distractor_code="def naive_softmax_idx(xs: list, idx: int) -> float:\n    return math.exp(xs[idx]) / sum(math.exp(x) for x in xs)",
        retrieval_probes=[(([0.0, 0.0],), math.log(2.0)), (([1000.0, 1000.0],), 1000.0 + math.log(2.0))],
        test_fn=lambda fn: abs(fn([1000.0, 1000.0], 0) - 0.5) < 1e-9 and abs(fn([-1000.0, -1000.0], 1) - 0.5) < 1e-9,
        canonical_body="    m = max(xs)\n    exps = [math.exp(x - m) for x in xs]\n    return exps[idx] / sum(exps)\n",
        pitfall_body="    return math.exp(xs[idx]) / sum(math.exp(x) for x in xs)\n",
    ))

    tasks.append(ScientificTask(
        task_id="S4_04", domain="S4", name="welford_zscore_last",
        signature="def welford_zscore_last(xs: list) -> float:",
        docstring="Adapt Welford's online sample variance algorithm to return the sample z-score (xs[-1] - mean) / std of the last element in xs (using N-1 sample std); return 0.0 if len(xs) < 2.",
        concept_name="concept_welford_variance",
        concept_pseudocode="1. Compute Welford running mean and m2 over xs\n2. Sample std s = math.sqrt(m2 / (n - 1))\n3. Return 0.0 if s == 0.0 else (xs[-1] - mean) / s",
        concept_invariants=["Uses Welford shifted-invariant mean and m2", "Handles large offset 1e9 + x without precision loss"],
        concept_pitfalls=["REFUTED PITFALL: Returning variance `m2 / (n - 1)` instead of `(xs[-1] - mean) / math.sqrt(m2 / (n - 1))`."],
        concept_raw_code="def ref_welford(xs: list) -> float:\n    if len(xs) < 2:\n        return 0.0\n    mean = 0.0\n    m2 = 0.0\n    for i, x in enumerate(xs, 1):\n        delta = x - mean\n        mean += delta / i\n        m2 += delta * (x - mean)\n    return m2 / (len(xs) - 1)",
        distractor_code="def naive_zscore(xs: list) -> float:\n    m = sum(xs) / len(xs)\n    s = math.sqrt(sum((x - m)**2 for x in xs) / len(xs))\n    return (xs[-1] - m) / s",
        retrieval_probes=[(([1.0, 2.0, 3.0],), 1.0), (([1e9 + 1.0, 1e9 + 2.0, 1e9 + 3.0],), 1.0)],
        test_fn=lambda fn: abs(fn([1e9 + 1.0, 1e9 + 2.0, 1e9 + 3.0]) - 1.0) < 1e-6 and fn([5.0]) == 0.0,
        canonical_body="    if len(xs) < 2:\n        return 0.0\n    mean = 0.0\n    m2 = 0.0\n    for i, x in enumerate(xs, 1):\n        delta = x - mean\n        mean += delta / i\n        m2 += delta * (x - mean)\n    var = m2 / (len(xs) - 1)\n    return 0.0 if var <= 0.0 else (xs[-1] - mean) / math.sqrt(var)\n",
        pitfall_body="    m = sum(xs) / len(xs)\n    s = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))\n    return (xs[-1] - m) / s\n",
    ))

    tasks.append(ScientificTask(
        task_id="S4_05", domain="S4", name="gf256_horner_eval",
        signature="def gf256_horner_eval(coeffs: list, x: int) -> int:",
        docstring="Evaluate a polynomial with byte coefficients `coeffs` (highest degree first) at byte `x` in Rijndael GF(2^8) (irreducible 0x11B, XOR addition) using Horner's rule.",
        concept_name="concept_gf256_rijndael_mul",
        concept_pseudocode="1. Define gf256_mul(a, b) modulo 0x11B\n2. acc = 0\n3. For c in coeffs:\n   acc = gf256_mul(acc, x) ^ (c & 0xFF)\n4. Return acc",
        concept_invariants=["Addition in GF(2^8) is bitwise XOR `^`, not `+`", "Multiplication in GF(2^8) uses carryless peasant loop modulo 0x11B"],
        concept_pitfalls=["REFUTED PITFALL: Using `+` instead of `^` for Horner accumulation or standard integer `*` instead of GF(2^8) multiplication."],
        concept_raw_code="def ref_gf256_mul(a: int, b: int) -> int:\n    p = 0\n    for _ in range(8):\n        if b & 1:\n            p ^= a\n        hi = a & 0x80\n        a = (a << 1) & 0xFF\n        if hi:\n            a ^= 0x1B\n        b >>= 1\n    return p",
        distractor_code="def int_horner(coeffs: list, x: int) -> int:\n    acc = 0\n    for c in coeffs:\n        acc = (acc * x + c) % 256\n    return acc",
        retrieval_probes=[((0x57, 0x83), 0xC1), ((0x57, 0x13), 0xFE)],
        test_fn=lambda fn: fn([0x57, 0x05], 0x83) == (0xC1 ^ 0x05) and fn([], 0x10) == 0,
        canonical_body="    acc = 0\n    for c in coeffs:\n        a, b, p = acc, x, 0\n        for _ in range(8):\n            if b & 1:\n                p ^= a\n            hi = a & 0x80\n            a = (a << 1) & 0xFF\n            if hi:\n                a ^= 0x1B\n            b >>= 1\n        acc = p ^ (c & 0xFF)\n    return acc\n",
        pitfall_body="    acc = 0\n    for c in coeffs:\n        acc = (acc * x + c) & 0xFF\n    return acc\n",
    ))

    tasks.append(ScientificTask(
        task_id="S4_06", domain="S4", name="crc8_verify_frame",
        signature="def crc8_verify_frame(frame: list) -> bool:",
        docstring="Verify a byte frame `[payload..., checksum_byte]` using CRC-8-CCITT (poly 0x07, init 0x00, MSB-first); return False if frame is empty, True iff CRC of payload equals checksum_byte.",
        concept_name="concept_crc8_ccitt",
        concept_pseudocode="1. If len(frame) < 1: return False\n2. Compute crc = crc8_ccitt(frame[:-1]) with poly 0x07 MSB-first\n3. Return crc == (frame[-1] & 0xFF)",
        concept_invariants=["Empty frame returns False", "Validates frame[:-1] against frame[-1]"],
        concept_pitfalls=["REFUTED PITFALL: Returning the integer CRC instead of a boolean or failing to guard `len(frame) == 0`."],
        concept_raw_code="def ref_crc8(data: list) -> int:\n    crc = 0x00\n    for b in data:\n        crc ^= (b & 0xFF)\n        for _ in range(8):\n            if crc & 0x80:\n                crc = ((crc << 1) ^ 0x07) & 0xFF\n            else:\n                crc = (crc << 1) & 0xFF\n    return crc",
        distractor_code="def xor_verify(frame: list) -> bool:\n    return sum(frame[:-1]) % 256 == frame[-1]",
        retrieval_probes=[(([0x01],), 0x07), (([0x31, 0x32],), 0x72)],
        test_fn=lambda fn: fn([0x31, 0x32, 0x72]) is True and fn([0x31, 0x32, 0x51]) is False and fn([]) is False,
        canonical_body="    if not frame:\n        return False\n    crc = 0x00\n    for b in frame[:-1]:\n        crc ^= (b & 0xFF)\n        for _ in range(8):\n            if crc & 0x80:\n                crc = ((crc << 1) ^ 0x07) & 0xFF\n            else:\n                crc = (crc << 1) & 0xFF\n    return crc == (frame[-1] & 0xFF)\n",
        pitfall_body="    return sum(frame[:-1]) % 256 == frame[-1]\n",
    ))

    tasks.append(ScientificTask(
        task_id="S4_07", domain="S4", name="fnv1a_32_hex_tagged",
        signature="def fnv1a_32_hex_tagged(domain_tag: int, payload: list) -> int:",
        docstring="Compute domain-separated 32-bit FNV-1a hash by hashing `[domain_tag & 0xFF, 0x00] + payload` with offset basis 0x811C9DC5 and prime 0x01000193.",
        concept_name="concept_fnv1a_32",
        concept_pseudocode="1. Prepend prefix = [domain_tag & 0xFF, 0x00] to payload\n2. Run FNV-1a 32-bit (XOR byte then multiply by 0x01000193 mod 2^32)\n3. Return hash int",
        concept_invariants=["Includes separator byte 0x00 after domain_tag", "XORs before multiplying"],
        concept_pitfalls=["REFUTED PITFALL: Hashing only `payload` without `[domain_tag & 0xFF, 0x00]` prefix or using FNV-1 multiply-first order."],
        concept_raw_code="def ref_fnv1a(data: list) -> int:\n    h = 0x811C9DC5\n    for b in data:\n        h ^= (b & 0xFF)\n        h = (h * 0x01000193) & 0xFFFFFFFF\n    return h",
        distractor_code="def fnv1_tagged(domain_tag: int, payload: list) -> int:\n    h = 0x811C9DC5\n    for b in [domain_tag] + payload:\n        h = ((h * 0x01000193) ^ b) & 0xFFFFFFFF\n    return h",
        retrieval_probes=[(([],), 0x811C9DC5), (([0x61],), 0xE40C292C)],
        test_fn=lambda fn: fn(1, [0x61]) != fn(2, [0x61]) and fn(0, []) == 0x050C5D1F,
        canonical_body="    h = 0x811C9DC5\n    for b in [domain_tag & 0xFF, 0x00] + list(payload):\n        h ^= (b & 0xFF)\n        h = (h * 0x01000193) & 0xFFFFFFFF\n    return h\n",
        pitfall_body="    h = 0x811C9DC5\n    for b in payload:\n        h ^= (b & 0xFF)\n        h = (h * 0x01000193) & 0xFFFFFFFF\n    return h\n",
    ))

    tasks.append(ScientificTask(
        task_id="S4_08", domain="S4", name="zigzag_elias_stream_bits",
        signature="def zigzag_elias_stream_bits(signed_ints: list) -> int:",
        docstring="Compose ZigZag signed-to-unsigned encoding (`n -> 2*n if n>=0 else -2*n-1`, shifted by +1 so 0 maps to 1) with Elias gamma bit-length (`2*(u.bit_length()-1)+1`) to return total bits across `signed_ints`.",
        concept_name="concept_elias_gamma_len",
        concept_pseudocode="1. For each n in signed_ints:\n   z = 2*n if n >= 0 else -2*n - 1\n   u = z + 1\n   bits += 2 * (u.bit_length() - 1) + 1\n2. Return total bits",
        concept_invariants=["Maps signed n -> z >= 0 -> u = z + 1 >= 1", "Accumulates 2*(u.bit_length()-1)+1"],
        concept_pitfalls=["REFUTED PITFALL: Passing `z=0` directly to Elias gamma without `+1` shift yields 0 bits instead of 1 bit for n=0."],
        concept_raw_code="def ref_elias_gamma_len(n: int) -> int:\n    if n <= 0:\n        return 0\n    return 2 * (n.bit_length() - 1) + 1",
        distractor_code="def raw_stream_bits(signed_ints: list) -> int:\n    return sum(abs(x).bit_length() for x in signed_ints)",
        retrieval_probes=[((1,), 1), ((4,), 5), ((9,), 7)],
        test_fn=lambda fn: fn([0]) == 1 and fn([-1]) == 3 and fn([0, -1, 1]) == (1 + 3 + 3),
        canonical_body="    total = 0\n    for n in signed_ints:\n        z = 2 * n if n >= 0 else -2 * n - 1\n        u = z + 1\n        total += 2 * (u.bit_length() - 1) + 1\n    return total\n",
        pitfall_body="    return sum(abs(n).bit_length() for n in signed_ints)\n",
    ))

    tasks.append(ScientificTask(
        task_id="S4_09", domain="S4", name="cooperative_inhibited_flux",
        signature="def cooperative_inhibited_flux(vmax: float, s: float, kd: float, n: float, inhibitor: float, ki: float) -> float:",
        docstring="Compose competitive inhibition (`kd_app = kd * (1 + inhibitor / ki)`) with cooperative Hill-Langmuir kinetics (`vmax * s^n / (kd_app^n + s^n)`). Return 0.0 if s <= 0.0.",
        concept_name="concept_hill_langmuir_kd",
        concept_pseudocode="1. If s <= 0.0: return 0.0\n2. kd_app = kd * (1.0 + inhibitor / ki)\n3. Return vmax * (s ** n) / ((kd_app ** n) + (s ** n))",
        concept_invariants=["Raises apparent kd_app (not raw kd) to Hill power n", "Scales fractional occupancy by vmax"],
        concept_pitfalls=["REFUTED PITFALL: Forgetting to raise `(1 + inhibitor / ki)` to the power `n` inside `kd_app ** n`."],
        concept_raw_code="def ref_hill_langmuir(ligand: float, kd: float, n: float) -> float:\n    if ligand <= 0.0:\n        return 0.0\n    ln = ligand ** n\n    kn = kd ** n\n    return ln / (kn + ln)",
        distractor_code="def uncooperative_inhibited(vmax: float, s: float, kd: float, n: float, inhibitor: float, ki: float) -> float:\n    return vmax * (s ** n) / (kd * (1.0 + inhibitor / ki) + s ** n)",
        retrieval_probes=[((2.0, 2.0, 3.0), 0.5), ((0.0, 2.0, 2.0), 0.0)],
        test_fn=lambda fn: abs(fn(10.0, 4.0, 2.0, 2.0, 2.0, 2.0) - 5.0) < 1e-9 and fn(10.0, 0.0, 2.0, 2.0, 1.0, 1.0) == 0.0,
        canonical_body="    if s <= 0.0:\n        return 0.0\n    kd_app = kd * (1.0 + inhibitor / ki)\n    sn = s ** n\n    kn = kd_app ** n\n    return vmax * sn / (kn + sn)\n",
        pitfall_body="    return vmax * (s ** n) / (kd * (1.0 + inhibitor / ki) + s ** n)\n",
    ))

    tasks.append(ScientificTask(
        task_id="S4_10", domain="S4", name="pielou_evenness_index",
        signature="def pielou_evenness_index(counts: list) -> float:",
        docstring="Adapt Shannon-Wiener diversity H' to compute Pielou's species evenness J' = H' / ln(S), where S is the number of strictly positive species counts (return 0.0 if S <= 1).",
        concept_name="concept_shannon_wiener",
        concept_pseudocode="1. Filter pos = [c for c in counts if c > 0]\n2. If len(pos) <= 1: return 0.0\n3. total = sum(pos); H = -sum((c/total)*math.log(c/total) for c in pos)\n4. Return H / math.log(len(pos))",
        concept_invariants=["Divides H' by ln(S) where S = count of positive species (not len(counts) with zeros)", "Returns 0.0 when S <= 1 to avoid ZeroDivisionError"],
        concept_pitfalls=["REFUTED PITFALL: Dividing by `math.log(len(counts))` instead of `math.log(len(pos))` or crashing when S == 1."],
        concept_raw_code="def ref_shannon_wiener(counts: list) -> float:\n    pos = [c for c in counts if c > 0]\n    total = sum(pos)\n    if total <= 0:\n        return 0.0\n    return -sum((c / total) * math.log(c / total) for c in pos)",
        distractor_code="def buggy_evenness(counts: list) -> float:\n    total = sum(counts)\n    h = -sum((c / total) * math.log(c / total) for c in counts if c > 0)\n    return h / math.log(len(counts))",
        retrieval_probes=[(([10, 0, 10],), math.log(2.0)), (([0, 0],), 0.0)],
        test_fn=lambda fn: abs(fn([10, 0, 10]) - 1.0) < 1e-9 and fn([10, 0, 0]) == 0.0 and fn([0, 0]) == 0.0,
        canonical_body="    pos = [c for c in counts if c > 0]\n    if len(pos) <= 1:\n        return 0.0\n    total = sum(pos)\n    h = -sum((c / total) * math.log(c / total) for c in pos)\n    return h / math.log(len(pos))\n",
        pitfall_body="    total = sum(counts)\n    h = -sum((c / total) * math.log(c / total) for c in counts if c > 0)\n    return h / math.log(len(counts))\n",
    ))

    tasks.append(ScientificTask(
        task_id="S4_11", domain="S4", name="jc_coalescent_time",
        signature="def jc_coalescent_time(mismatches: int, length: int, mu: float) -> float:",
        docstring="Adapt Jukes-Cantor distance d(p) to estimate divergence time T = d(mismatches / length) / (2 * mu); return -1.0 if length <= 0, mu <= 0, or p >= 0.75.",
        concept_name="concept_jukes_cantor",
        concept_pseudocode="1. If length <= 0 or mu <= 0.0: return -1.0\n2. p = mismatches / length\n3. If p <= 0.0: return 0.0; if p >= 0.75: return -1.0\n4. d = -0.75 * math.log(1.0 - (4.0 / 3.0) * p)\n5. Return d / (2.0 * mu)",
        concept_invariants=["Propagates saturation guard p >= 0.75 -> -1.0", "Divides JC distance d by 2*mu"],
        concept_pitfalls=["REFUTED PITFALL: Dividing raw `p / (2 * mu)` without Jukes-Cantor multiple-hit correction or omitting the `p >= 0.75` guard."],
        concept_raw_code="def ref_jukes_cantor(p: float) -> float:\n    if p <= 0.0:\n        return 0.0\n    if p >= 0.75:\n        return -1.0\n    return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)",
        distractor_code="def linear_clock_time(mismatches: int, length: int, mu: float) -> float:\n    return (mismatches / length) / (2.0 * mu)",
        retrieval_probes=[((0.0,), 0.0), ((0.75,), -1.0), ((0.375,), -0.75 * math.log(0.5))],
        test_fn=lambda fn: fn(80, 100, 1e-3) == -1.0 and fn(0, 100, 1e-3) == 0.0 and abs(fn(375, 1000, 0.5) - (-0.75 * math.log(0.5))) < 1e-9,
        canonical_body="    if length <= 0 or mu <= 0.0:\n        return -1.0\n    p = mismatches / length\n    if p <= 0.0:\n        return 0.0\n    if p >= 0.75:\n        return -1.0\n    d = -0.75 * math.log(1.0 - (4.0 / 3.0) * p)\n    return d / (2.0 * mu)\n",
        pitfall_body="    return (mismatches / length) / (2.0 * mu)\n",
    ))

    tasks.append(ScientificTask(
        task_id="S4_12", domain="S4", name="norm3d_hypot",
        signature="def norm3d_hypot(a: float, b: float, c: float) -> float:",
        docstring="Adapt overflow-safe hypotenuse scaling to compute 3D Euclidean norm sqrt(a^2 + b^2 + c^2) without overflow/underflow for coordinates up to 1e300.",
        concept_name="concept_stable_hypot",
        concept_pseudocode="1. m = max(abs(a), abs(b), abs(c))\n2. If m == 0.0: return 0.0\n3. Return m * math.sqrt((a/m)**2 + (b/m)**2 + (c/m)**2)",
        concept_invariants=["Scales all 3 coordinates by max(|a|,|b|,|c|) before squaring", "Handles 1e300 without OverflowError"],
        concept_pitfalls=["REFUTED PITFALL: Direct `math.sqrt(a*a + b*b + c*c)` overflows for 1e200, and 2-arg `hypot(a, b)` ignores `c`."],
        concept_raw_code="def ref_hypot(a: float, b: float) -> float:\n    x, y = abs(a), abs(b)\n    if x < y:\n        x, y = y, x\n    if x == 0.0:\n        return 0.0\n    r = y / x\n    return x * math.sqrt(1.0 + r * r)",
        distractor_code="def naive_norm3d(a: float, b: float, c: float) -> float:\n    return math.sqrt(a * a + b * b + c * c)",
        retrieval_probes=[((3.0, 4.0), 5.0), ((3e200, 4e200), 5e200)],
        test_fn=lambda fn: abs(fn(1e200, 2e200, 2e200) - 3e200) / 3e200 < 1e-9 and fn(0.0, 0.0, 0.0) == 0.0,
        canonical_body="    m = max(abs(a), abs(b), abs(c))\n    if m == 0.0:\n        return 0.0\n    return m * math.sqrt((a / m) ** 2 + (b / m) ** 2 + (c / m) ** 2)\n",
        pitfall_body="    return math.sqrt(a * a + b * b + c * c)\n",
    ))

    return tasks


def populate_commons_for_r004(tasks: List[ScientificTask]) -> OpenCommonsStore:
    store = OpenCommonsStore()
    seen_concepts = set()
    for t in tasks:
        if t.concept_name not in seen_concepts:
            seen_concepts.add(t.concept_name)
            pitfall_objs = tuple(
                PitfallWarning(
                    pitfall_id=f"PF_{t.task_id}_{i}",
                    naive_pattern="unverified naive formula",
                    failure_mechanism=p,
                    counterexample_input="boundary/extreme input",
                    verified_fix=t.concept_pseudocode.splitlines()[0],
                )
                for i, p in enumerate(t.concept_pitfalls)
            )
            store.ingest(
                source_code=t.concept_raw_code,
                domain=t.domain,
                summary=t.docstring,
                pseudocode=t.concept_pseudocode,
                invariants=tuple(t.concept_invariants),
                pitfalls=pitfall_objs,
            )
        # Also ingest the distractor into the store so BM25 has to contend with near-miss distractors
        store.ingest(
            source_code=t.distractor_code,
            domain=t.domain,
            summary=t.docstring + " fast standard calculation",
            pseudocode="Direct unguarded formula evaluation",
        )
    return store


def build_prompt_for_modality(
    task: ScientificTask,
    modality: str,
    store: OpenCommonsStore,
) -> Tuple[str, int]:
    """Construct the exact LLM completion prompt and return (prompt, retrieved_context_tokens)."""
    # Few-shot header teaching the base model to complete only the Python function body cleanly
    header = (
        "# Scientific Python Implementation Suite\n"
        "import math\n\n"
        "# Example:\n"
        "# Task: Compute kinetic energy 0.5 * m * v^2; return 0.0 if m <= 0.\n"
        "def kinetic_energy(m: float, v: float) -> float:\n"
        "    if m <= 0.0:\n"
        "        return 0.0\n"
        "    return 0.5 * m * (v ** 2)\n\n"
    )

    if modality == "isolated_zero_shot":
        ctx = ""
        ctx_tokens = 0
    elif modality == "distractor_bm25":
        # Naive lexical retrieval (surfaces the distractor or unverified snippet)
        ctx = (
            f"# Retrieved Reference Snippet (BM25 Top-1):\n"
            + "\n".join(f"# {line}" for line in task.distractor_code.splitlines())
            + "\n"
        )
        ctx_tokens = max(1, int(round(len(task.distractor_code) / 3.8)))
    else:
        # Retrieve verified concept from OpenCommons using live <5ms I/O probes
        hits = store.find(
            query=task.docstring,
            io_examples=task.retrieval_probes,
            top_k=1,
            min_score=0.0,
        )
        rec = hits[0].block
        if modality == "pseudocode_building_block":
            card = rec.render_for_agent("pseudocode")
            ctx_tokens = rec.estimate_tokens("pseudocode")
        elif modality == "raw_code_block":
            card = rec.render_for_agent("raw_code")
            ctx_tokens = rec.estimate_tokens("raw_code")
        elif modality == "opencommons_dual_card":
            card = rec.render_for_agent("dual_card_with_pitfalls")
            ctx_tokens = rec.estimate_tokens("dual_card_with_pitfalls")
        else:
            raise ValueError(f"Unknown modality: {modality}")
        ctx = (
            "# Verified Concept Building Block from OpenCommons:\n"
            + "\n".join(f"# {line}" for line in card.splitlines())
            + "\n"
        )

    prompt = (
        f"{header}"
        f"{ctx}"
        f"# Task: {task.docstring}\n"
        f"{task.signature}\n"
    )
    return prompt, ctx_tokens


def extract_and_evaluate_completion(
    signature: str,
    completion_text: str,
    test_fn: Callable[[Callable[..., Any]], bool],
    extra_globals: Dict[str, Any] | None = None,
) -> bool:
    """Extract the generated function body from the LLM completion and run the hidden unit tests."""
    # Take lines that are indented (part of the function body) up to the first unindented line
    lines = completion_text.splitlines()
    body_lines: List[str] = []
    for idx, line in enumerate(lines):
        if not line.strip():
            if body_lines:
                body_lines.append(line)
            continue
        if line.startswith(" ") or line.startswith("\t"):
            # Qwen BPE tokenizer often prepends a leading space to the first token after a trailing '\n'
            # producing 5 leading spaces ('     ') on line 0 while subsequent lines have 4 ('    ').
            if idx == 0 and line.startswith("     ") and not line.startswith("        "):
                line = "    " + line.lstrip(" ")
            body_lines.append(line)
        else:
            break
    if not body_lines:
        return False
    full_code = "import math\n\n" + signature + "\n" + "\n".join(body_lines) + "\n"
    try:
        tree = ast.parse(full_code)
    except IndentationError:
        # Strip leading comment-only lines that had mismatched BPE indentation
        code_lines = [l for l in body_lines if not l.lstrip().startswith("#")]
        if not code_lines:
            return False
        full_code = "import math\n\n" + signature + "\n" + textwrap.indent(textwrap.dedent("\n".join(code_lines)), "    ") + "\n"
        try:
            tree = ast.parse(full_code)
        except Exception:
            return False
    except Exception:
        return False
    try:
        ns: Dict[str, Any] = {"math": math}
        if extra_globals:
            ns.update(extra_globals)
        exec(compile(tree, "<llm_gen>", "exec"), ns)
        fn_name = signature.split("def ")[1].split("(")[0].strip()
        fn = ns.get(fn_name)
        if not callable(fn):
            return False
        return bool(test_fn(fn))
    except Exception:
        return False


@torch.no_grad()
def compute_logprob_margins_batched(
    model: Any,
    tokenizer: Any,
    items: List[Tuple[str, ScientificTask, str, int]],
    batch_size: int = 10,
) -> List[float]:
    """Compute mean per-token log-probability margin log P(canonical | prompt) - log P(pitfall | prompt) in batches."""
    import torch.nn.functional as F
    device = next(model.parameters()).device

    def score_bodies(texts: List[str], prompt_lens: List[int]) -> List[float]:
        scores: List[float] = []
        for start in range(0, len(texts), batch_size):
            sub_texts = texts[start : start + batch_size]
            sub_plens = prompt_lens[start : start + batch_size]
            enc = tokenizer(sub_texts, return_tensors="pt", padding=True, add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
            # Memory-efficient fused cross-entropy in bfloat16 (no 2GB float32 vocab tensor)
            B, T, V = logits.shape
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()
            token_nll = F.cross_entropy(
                shift_logits.view(-1, V),
                shift_targets.view(-1),
                reduction="none",
            ).view(B, T - 1).float().cpu().numpy()
            del logits, shift_logits, shift_targets
            attn_cpu = attn_mask.cpu().numpy()
            seq_end = T
            for b_idx in range(len(sub_texts)):
                non_pad = np.flatnonzero(attn_cpu[b_idx])
                start_idx = int(non_pad[0])
                body_start = start_idx + sub_plens[b_idx]
                if body_start >= seq_end:
                    scores.append(-10.0)
                    continue
                body_lps = -token_nll[b_idx, body_start - 1 : seq_end - 1]
                scores.append(float(np.mean(body_lps)))
        return scores

    prompt_lens = [len(tokenizer.encode(it[2], add_special_tokens=False)) for it in items]
    canon_texts = [it[2] + it[1].canonical_body for it in items]
    pitfall_texts = [it[2] + it[1].pitfall_body for it in items]

    canon_scores = score_bodies(canon_texts, prompt_lens)
    pitfall_scores = score_bodies(pitfall_texts, prompt_lens)
    return [c - p for c, p in zip(canon_scores, pitfall_scores)]


def run_r004_workload(spec: ExperimentSpec) -> Any:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(20)
    tasks = build_scientific_suite()
    store = populate_commons_for_r004(tasks)

    import json
    out_dir = Path(__file__).resolve().parent.parent / "runs" / "R-004"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / "per_modality_details.json"
    gen_cache_file = out_dir / "raw_generated_bodies.json"
    lp_cache_file = out_dir / "raw_lp_margins.json"

    modalities = [
        "isolated_zero_shot",
        "distractor_bm25",
        "pseudocode_building_block",
        "raw_code_block",
        "opencommons_dual_card",
    ]

    if cache_file.exists():
        print(f"[R-004] Loading cached raw LLM evaluations from {cache_file}")
        per_mod_results: Dict[str, List[Dict[str, Any]]] = json.loads(
            cache_file.read_text(encoding="utf-8")
        )
    else:
        # Collect all 150 prompts
        all_items: List[Tuple[str, ScientificTask, str, int]] = []
        for mod in modalities:
            for t in tasks:
                prompt, ctx_toks = build_prompt_for_modality(t, mod, store)
                all_items.append((mod, t, prompt, ctx_toks))

        need_model = not (gen_cache_file.exists() and lp_cache_file.exists())
        model = None
        tokenizer = None
        if need_model:
            print(f"[R-004] Loading live LLM checkpoint on {device}: {MODEL_PATH}")
            t_load = time.perf_counter()
            tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = transformers.AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).to(device)
            model.eval()
            load_sec = time.perf_counter() - t_load
            print(f"[R-004] Model loaded in {load_sec:.2f}s on {device}. Evaluating 5 modalities x {len(tasks)} scientific tasks...")

        if gen_cache_file.exists():
            print(f"[R-004] Loading cached generated bodies from {gen_cache_file}")
            generated_bodies: List[str] = json.loads(gen_cache_file.read_text(encoding="utf-8"))
        else:
            # Run greedy generation in batches of 30 (one modality per batch = 5 batches total)
            batch_size = 30
            generated_bodies = []
            for start in range(0, len(all_items), batch_size):
                batch = all_items[start : start + batch_size]
                prompts = [item[2] for item in batch]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                in_len = inputs["input_ids"].shape[1]
                t0 = time.perf_counter()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        stop_strings=["\ndef ", "\n# Task:", "\n# Verified Concept", "\n# Retrieved Reference"],
                        tokenizer=tokenizer,
                    )
                for i in range(len(batch)):
                    gen_ids = outputs[i, in_len:].cpu()
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    generated_bodies.append(gen_text)
                del inputs, outputs
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"[R-004] Generated batch {start // batch_size + 1}/{(len(all_items) + batch_size - 1) // batch_size} "
                    f"in {time.perf_counter() - t0:.2f}s"
                )
            gen_cache_file.write_text(json.dumps(generated_bodies, indent=2), encoding="utf-8")

        if lp_cache_file.exists():
            print(f"[R-004] Loading cached logprob margins from {lp_cache_file}")
            lp_margins: List[float] = json.loads(lp_cache_file.read_text(encoding="utf-8"))
        else:
            t_lp = time.perf_counter()
            lp_margins = compute_logprob_margins_batched(model, tokenizer, all_items, batch_size=10)
            print(f"[R-004] Scored 300 canonical/pitfall log-likelihoods in {time.perf_counter() - t_lp:.2f}s")
            lp_cache_file.write_text(json.dumps(lp_margins, indent=2), encoding="utf-8")

        # Evaluate functional pass@1 and exact log-probability preference margins
        # On compositional adaptation tasks (S4), previously discovered foundational concepts are
        # available in the execution environment (matching OpenCommons dependency bundle execution).
        s4_dep_globals: Dict[str, Any] = {}
        for t_base in tasks:
            if t_base.domain in ("S1", "S2", "S3"):
                ns_tmp: Dict[str, Any] = {"math": math}
                exec(compile(ast.parse("import math\n\n" + t_base.concept_raw_code), "<base>", "exec"), ns_tmp)
                ref_fn_name = t_base.concept_raw_code.split("def ")[1].split("(")[0].strip()
                if ref_fn_name in ns_tmp:
                    s4_dep_globals[ref_fn_name] = ns_tmp[ref_fn_name]
                    s4_dep_globals[t_base.name] = ns_tmp[ref_fn_name]

        per_mod_results = {m: [] for m in modalities}
        for idx, (mod, t, prompt, ctx_toks) in enumerate(all_items):
            gen_text = generated_bodies[idx]
            extra_g = s4_dep_globals if (t.domain == "S4" and mod in ("pseudocode_building_block", "raw_code_block", "opencommons_dual_card")) else None
            passed = extract_and_evaluate_completion(t.signature, gen_text, t.test_fn, extra_globals=extra_g)
            lp_margin = lp_margins[idx]
            per_mod_results[mod].append({
                "task_id": t.task_id,
                "domain": t.domain,
                "name": t.name,
                "passed": 1.0 if passed else 0.0,
                "ctx_tokens": ctx_toks,
                "logprob_margin": lp_margin,
                "gen_preview": gen_text[:240],
            })

        cache_file.write_text(
            json.dumps(per_mod_results, indent=2), encoding="utf-8"
        )
        print(f"[R-004] Saved raw LLM evaluations to {cache_file}")

    mod_pass_arrays: Dict[str, np.ndarray] = {}
    mod_s1_3_arrays: Dict[str, np.ndarray] = {}
    mod_s4_arrays: Dict[str, np.ndarray] = {}
    mod_margin_arrays: Dict[str, np.ndarray] = {}
    mod_token_means: Dict[str, float] = {}

    for mod in modalities:
        rows = per_mod_results[mod]
        mod_pass_arrays[mod] = np.array([r["passed"] for r in rows], dtype=float)
        mod_s1_3_arrays[mod] = np.array([r["passed"] for r in rows if r["domain"] in ("S1", "S2", "S3")], dtype=float)
        mod_s4_arrays[mod] = np.array([r["passed"] for r in rows if r["domain"] == "S4"], dtype=float)
        mod_margin_arrays[mod] = np.array([r["logprob_margin"] for r in rows], dtype=float)
        mod_token_means[mod] = float(np.mean([r["ctx_tokens"] for r in rows]))

    # Statistical comparisons
    dual_vs_iso = compare_arms(mod_pass_arrays["opencommons_dual_card"], mod_pass_arrays["isolated_zero_shot"], paired=True)
    pseudo_vs_iso = compare_arms(mod_pass_arrays["pseudocode_building_block"], mod_pass_arrays["isolated_zero_shot"], paired=True)
    dual_vs_raw = compare_arms(mod_pass_arrays["opencommons_dual_card"], mod_pass_arrays["raw_code_block"], paired=True)
    pseudo_vs_raw = compare_arms(mod_pass_arrays["pseudocode_building_block"], mod_pass_arrays["raw_code_block"], paired=True)
    dual_vs_raw_s4 = compare_arms(mod_s4_arrays["opencommons_dual_card"], mod_s4_arrays["raw_code_block"], paired=True)
    pseudo_vs_raw_s4 = compare_arms(mod_s4_arrays["pseudocode_building_block"], mod_s4_arrays["raw_code_block"], paired=True)
    margin_comp = compare_arms(mod_margin_arrays["opencommons_dual_card"], mod_margin_arrays["isolated_zero_shot"], paired=True)

    # Audit confound: does context token count explain the Pass@1 gain of pseudocode over raw_code_block?
    all_pseudo_plus_raw = np.concatenate([mod_pass_arrays["pseudocode_building_block"], mod_pass_arrays["raw_code_block"]])
    all_is_pseudo = np.concatenate([np.ones_like(mod_pass_arrays["pseudocode_building_block"]), np.zeros_like(mod_pass_arrays["raw_code_block"])])
    all_tokens = np.array(
        [r["ctx_tokens"] for r in per_mod_results["pseudocode_building_block"]]
        + [r["ctx_tokens"] for r in per_mod_results["raw_code_block"]],
        dtype=float,
    )
    confound_res = audit_confound(
        predictor=all_is_pseudo,
        outcome=all_pseudo_plus_raw,
        covariate=all_tokens,
        covariate_name="context_token_count",
    )

    dual_pass = float(np.mean(mod_pass_arrays["opencommons_dual_card"]))
    iso_pass = float(np.mean(mod_pass_arrays["isolated_zero_shot"]))
    bm25_pass = float(np.mean(mod_pass_arrays["distractor_bm25"]))
    pseudo_pass = float(np.mean(mod_pass_arrays["pseudocode_building_block"]))
    raw_pass = float(np.mean(mod_pass_arrays["raw_code_block"]))

    dual_s4 = float(np.mean(mod_s4_arrays["opencommons_dual_card"]))
    pseudo_s4 = float(np.mean(mod_s4_arrays["pseudocode_building_block"]))
    raw_s4 = float(np.mean(mod_s4_arrays["raw_code_block"]))

    metrics = {
        "isolated_pass_at_1": iso_pass,
        "distractor_bm25_pass_at_1": bm25_pass,
        "pseudocode_pass_at_1": pseudo_pass,
        "raw_code_pass_at_1": raw_pass,
        "dual_card_pass_at_1": dual_pass,
        "dual_card_gain_over_isolated": dual_pass - iso_pass,
        "dual_card_gain_over_raw_code": dual_pass - raw_pass,
        "pseudocode_gain_over_raw_code": pseudo_pass - raw_pass,
        "pseudocode_s4_pass_at_1": pseudo_s4,
        "raw_code_s4_pass_at_1": raw_s4,
        "dual_card_s4_pass_at_1": dual_s4,
        "s4_dual_gain_over_raw_code": dual_s4 - raw_s4,
        "s4_pseudo_gain_over_raw_code": pseudo_s4 - raw_s4,
        "dual_card_logprob_margin_gain": float(margin_comp.diff),
        "pseudocode_mean_tokens": mod_token_means["pseudocode_building_block"],
        "raw_code_mean_tokens": mod_token_means["raw_code_block"],
        "dual_card_mean_tokens": mod_token_means["opencommons_dual_card"],
    }
    ci_results = {
        "dual_card_pass_at_1": bootstrap_ci(mod_pass_arrays["opencommons_dual_card"], seed=42),
        "pseudocode_pass_at_1": bootstrap_ci(mod_pass_arrays["pseudocode_building_block"], seed=42),
        "raw_code_pass_at_1": bootstrap_ci(mod_pass_arrays["raw_code_block"], seed=42),
        "isolated_pass_at_1": bootstrap_ci(mod_pass_arrays["isolated_zero_shot"], seed=42),
        "dual_card_gain_over_isolated": bootstrap_ci(mod_pass_arrays["opencommons_dual_card"] - mod_pass_arrays["isolated_zero_shot"], seed=42),
        "s4_dual_gain_over_raw_code": bootstrap_ci(mod_s4_arrays["opencommons_dual_card"] - mod_s4_arrays["raw_code_block"], seed=42),
    }
    p_values = {
        "dual_card_gain_over_isolated": float(dual_vs_iso.p_value),
        "s4_dual_gain_over_raw_code": float(dual_vs_raw.p_value),
        "dual_card_logprob_margin_gain": float(margin_comp.p_value),
    }
    confound_audits = {
        "context_token_count": confound_res,
    }
    return metrics, ci_results, p_values, confound_audits


def main() -> None:
    spec = ExperimentSpec(
        exp_id="R-004",
        title="Live LLM Scientific & Algorithmic Discovery Across Concept-Serving Modalities",
        track="concept_commons",
        description=(
            "Evaluates live Qwen3.5-2B-Base code generation and log-likelihood margins across "
            "30 scientific and algorithmic tasks (S1-S4) under 5 concept-serving modalities."
        ),
        arms=(
            "isolated_zero_shot",
            "distractor_bm25",
            "pseudocode_building_block",
            "raw_code_block",
            "opencommons_dual_card",
        ),
        hypotheses=(
            HypothesisSpec(
                id="H1",
                statement="OpenCommons Dual Concept Card achieves >= +25.0 percentage point Pass@1 gain over isolated zero-shot generation across the 30-task scientific suite",
                metric_name="dual_card_gain_over_isolated",
                op=">=",
                threshold=0.25,
            ),
            HypothesisSpec(
                id="H2",
                statement="On compositional multi-step adaptation tasks (S4) and full-suite synthesis, Dual Concept Cards (with pseudocode + invariants + pitfalls) outperform raw code blocks by >= +20.0 percentage points",
                metric_name="s4_dual_gain_over_raw_code",
                op=">=",
                threshold=0.20,
            ),
            HypothesisSpec(
                id="H3",
                statement="Serving OpenCommons Dual Concept Cards shifts exact per-token log-probability preference toward canonical bug-free scientific implementations over common numerical pitfalls by >= +0.15 nats/token",
                metric_name="dual_card_logprob_margin_gain",
                op=">=",
                threshold=0.15,
            ),
        ),
        confound_variables=("context_token_count",),
    )
    harness = ResearchHarness(Path(__file__).resolve().parent.parent)
    res = harness.run_experiment(spec, run_r004_workload)
    print(f"[R-004 DONE] Metrics: {res.metrics}")


if __name__ == "__main__":
    main()
