"""Unit tests for Cryptanalytic Factorization & Lattice Benchmark."""

from __future__ import annotations

import math
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from experiments.domain_discovery.number_theory_and_algebra.benchmark import (
    FUNCS,
    run_cryptanalytic_factorization_benchmark,
)


def test_default_cryptanalytic_benchmark_passes():
    res = run_cryptanalytic_factorization_benchmark(
        p=257,
        q=263,
        private_d=13,
        message=42,
        dlog_secret=19,
    )
    assert res.passed is True
    assert res.p_is_prime is True
    assert res.q_is_prime is True
    assert res.factorization_success is True
    assert res.crt_decryption_match is True
    assert res.dlog_match is True
    assert res.convergents_count > 0


def test_fermat_factorization_and_totient_consistency():
    p, q = 1013, 1019
    res = run_cryptanalytic_factorization_benchmark(
        p=p,
        q=q,
        private_d=7,
        message=99,
        dlog_secret=23,
    )
    assert res.factorization_success is True
    assert res.recovered_factors == (min(p, q), max(p, q))
    assert res.euler_totient == (p - 1) * (q - 1)
    assert res.carmichael_lambda == math.lcm(p - 1, q - 1)


def test_crt_decryption_reconstruction():
    crt_pair = FUNCS["chinese_remainder_theorem_pair"]
    p, q = 137, 139
    m = 1234
    c = pow(m, 5, p * q)
    d = pow(5, -1, (p - 1) * (q - 1))
    mp = pow(c, d % (p - 1), p)
    mq = pow(c, d % (q - 1), q)
    decrypted, lcm_mod = crt_pair(mp, p, mq, q)
    assert decrypted == m
    assert lcm_mod == p * q


def test_discrete_log_baby_step_giant_step():
    dlog = FUNCS["discrete_log_baby_step_giant_step"]
    prime = 503
    base = 5
    for secret in [7, 42, 101, 250]:
        target = pow(base, secret, prime)
        rec = dlog(base, target, prime)
        assert pow(base, rec, prime) == target


def test_continued_fraction_determinant_identity():
    cf_exp = FUNCS["continued_fraction_expansion"]
    cf_conv = FUNCS["continued_fraction_convergents"]
    terms = cf_exp(355, 113, 20)
    convergents = cf_conv(terms)
    assert len(convergents) >= 2
    for i in range(1, len(convergents)):
        h_curr, k_curr = convergents[i]
        h_prev, k_prev = convergents[i - 1]
        det = h_curr * k_prev - h_prev * k_curr
        assert abs(det) == 1
