"""Cryptanalytic Factorization & Lattice Benchmark for Number Theory & Algebra.

Composes >= 5 verified OpenCommons concepts:
1. `miller_rabin_deterministic_round`: Primality certification of key parameters.
2. `fermat_factorization_step`: Difference of squares algebraic factorization of composite modulus.
3. `euler_totient_phi_factorized`: Direct totient evaluation from recovered prime factors.
4. `carmichael_lambda_function`: Minimal universal exponent computation for modular group.
5. `continued_fraction_expansion`: Rational partial quotient expansion of public exponent ratio.
6. `continued_fraction_convergents`: Best rational approximations for Wiener small private key attack.
7. `chinese_remainder_theorem_pair`: Modular CRT secret reconstruction.
8. `discrete_log_baby_step_giant_step`: Subgroup discrete logarithm extraction.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _load_concept_map() -> dict[str, Any]:
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "number_theory_and_algebra.json"
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    concepts = data if isinstance(data, list) else data.get("concepts", data.get("blocks", []))
    func_map = {}
    for c in concepts:
        exec_env: dict[str, Any] = {"math": math}
        exec(c["source_code"], exec_env)
        func_map[c["name"]] = exec_env[c["name"]]
    return func_map


FUNCS = _load_concept_map()
miller_rabin_deterministic_round = FUNCS["miller_rabin_deterministic_round"]
fermat_factorization_step = FUNCS["fermat_factorization_step"]
euler_totient_phi_factorized = FUNCS["euler_totient_phi_factorized"]
carmichael_lambda_function = FUNCS["carmichael_lambda_function"]
continued_fraction_expansion = FUNCS["continued_fraction_expansion"]
continued_fraction_convergents = FUNCS["continued_fraction_convergents"]
chinese_remainder_theorem_pair = FUNCS["chinese_remainder_theorem_pair"]
discrete_log_baby_step_giant_step = FUNCS["discrete_log_baby_step_giant_step"]


@dataclass
class CryptanalyticBenchmarkResult:
    prime_p: int
    prime_q: int
    modulus_n: int
    p_is_prime: bool
    q_is_prime: bool
    recovered_factors: tuple[int, int]
    factorization_success: bool
    euler_totient: int
    carmichael_lambda: int
    continued_fraction_terms: int
    convergents_count: int
    wiener_recovered_d: int | None
    original_message: int
    ciphertext: int
    crt_decrypted_message: int
    crt_decryption_match: bool
    dlog_secret: int
    dlog_recovered_exponent: int
    dlog_match: bool
    passed: bool


def run_cryptanalytic_factorization_benchmark(
    p: int = 257,
    q: int = 263,
    private_d: int = 13,
    message: int = 42,
    dlog_secret: int = 19,
) -> CryptanalyticBenchmarkResult:
    """Execute end-to-end cryptanalytic factorization and lattice benchmark."""
    # 1. Certify primality of p and q with Miller-Rabin deterministic rounds
    p_prime = miller_rabin_deterministic_round(p, 2)
    q_prime = miller_rabin_deterministic_round(q, 2)
    modulus_n = p * q

    # 2. Factor composite modulus n via Fermat's difference of squares
    f1, f2 = fermat_factorization_step(modulus_n, 5000)
    recovered_factors = (min(f1, f2), max(f1, f2))
    factorization_ok = (recovered_factors == (min(p, q), max(p, q)))

    # 3. Compute Euler totient phi(n) and Carmichael lambda(n)
    phi = euler_totient_phi_factorized([recovered_factors[0], recovered_factors[1]])
    lam = carmichael_lambda_function([[recovered_factors[0], 1], [recovered_factors[1], 1]])

    # 4. Form public exponent e matching Wiener small private exponent condition
    # e * private_d = 1 mod lam
    e = 3
    found_e = False
    for cand in range(3, lam, 2):
        if (cand * private_d) % lam == 1 and math.gcd(cand, phi) == 1:
            e = cand
            found_e = True
            break
    if not found_e:
        e = pow(private_d, -1, phi)

    # 5. Continued fraction expansion of e / n
    cf_terms = continued_fraction_expansion(e, modulus_n, 40)
    convergents = continued_fraction_convergents(cf_terms)

    # Wiener attack: search convergents k / d
    recovered_d: int | None = None
    for k_val, d_val in convergents:
        if k_val == 0 or d_val == 0:
            continue
        if (e * d_val - 1) % k_val == 0:
            phi_cand = (e * d_val - 1) // k_val
            # Roots of x^2 - (n - phi_cand + 1)x + n == 0
            s_val = modulus_n - phi_cand + 1
            disc = s_val * s_val - 4 * modulus_n
            if disc >= 0:
                sq_d = int(math.isqrt(disc))
                if sq_d * sq_d == disc:
                    root1 = (s_val - sq_d) // 2
                    root2 = (s_val + sq_d) // 2
                    if root1 * root2 == modulus_n and root1 > 1:
                        recovered_d = d_val
                        break

    # 6. RSA encryption and Chinese Remainder Theorem decryption
    ciphertext = pow(message, e, modulus_n)
    dp = private_d % (p - 1)
    dq = private_d % (q - 1)
    mp = pow(ciphertext, dp, p)
    mq = pow(ciphertext, dq, q)
    decrypted_m, _ = chinese_remainder_theorem_pair(mp, p, mq, q)
    crt_ok = (decrypted_m == message)

    # 7. Discrete logarithm attack in prime subgroup modulo p
    g = 3
    h = pow(g, dlog_secret, p)
    recovered_exponent = discrete_log_baby_step_giant_step(g, h, p)
    dlog_ok = (pow(g, recovered_exponent, p) == h)

    all_passed = (
        p_prime
        and q_prime
        and factorization_ok
        and (phi == (p - 1) * (q - 1))
        and (lam == math.lcm(p - 1, q - 1))
        and (len(convergents) > 0)
        and crt_ok
        and dlog_ok
    )

    return CryptanalyticBenchmarkResult(
        prime_p=p,
        prime_q=q,
        modulus_n=modulus_n,
        p_is_prime=p_prime,
        q_is_prime=q_prime,
        recovered_factors=recovered_factors,
        factorization_success=factorization_ok,
        euler_totient=phi,
        carmichael_lambda=lam,
        continued_fraction_terms=len(cf_terms),
        convergents_count=len(convergents),
        wiener_recovered_d=recovered_d,
        original_message=message,
        ciphertext=ciphertext,
        crt_decrypted_message=decrypted_m,
        crt_decryption_match=crt_ok,
        dlog_secret=dlog_secret,
        dlog_recovered_exponent=recovered_exponent,
        dlog_match=dlog_ok,
        passed=all_passed,
    )
