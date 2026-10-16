"""Pytest test suite for Machine Learning Scaled Transformer Self-Attention Benchmark."""

import json
from pathlib import Path
import sys
import pytest

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from experiments.domain_discovery.machine_learning.benchmark import (
    transformer_causal_self_attention_layer,
    simulate_autoregressive_transformer_training,
    attention_dot_product_scaled,
    causal_attention_mask_apply,
    alibi_relative_bias_scalar,
    layer_norm_1d,
    cross_entropy_with_logits,
    warmup_cosine_decay_lr,
)


def test_causal_mask_and_softmax_sum():
    """Verify that causal attention weights for future positions (j > i) are strictly 0 and row sums are 1.0."""
    seq_len = 5
    head_dim = 4
    q = [[1.0, 0.5, -0.2, 0.1] for _ in range(seq_len)]
    k = [[0.8, -0.4, 0.3, 0.2] for _ in range(seq_len)]
    v = [[0.5, 0.1, 0.9, -0.3] for _ in range(seq_len)]

    res = transformer_causal_self_attention_layer(q, k, v, head_dim=head_dim, alibi_slope=0.5)
    weights = res["attention_weights"]

    assert len(weights) == seq_len
    for i in range(seq_len):
        row = weights[i]
        # Softmax sum should equal 1.0
        assert abs(sum(row) - 1.0) < 1e-5, f"Row {i} softmax sum != 1.0: {sum(row)}"

        # Strict causality check: positions j > i must have exactly 0 probability weight
        for j in range(i + 1, seq_len):
            assert row[j] < 1e-12, f"Future leakage at position ({i}, {j}): {row[j]}"


def test_alibi_bias_decay():
    """Verify ALiBi linear penalty grows with token distance."""
    slope = 0.5
    query_pos = 10
    bias_0 = alibi_relative_bias_scalar(query_pos, query_pos, slope)
    bias_1 = alibi_relative_bias_scalar(query_pos, query_pos - 1, slope)
    bias_2 = alibi_relative_bias_scalar(query_pos, query_pos - 2, slope)
    bias_5 = alibi_relative_bias_scalar(query_pos, query_pos - 5, slope)

    assert bias_0 == 0.0, "Zero distance bias must be 0.0"
    assert bias_1 == -0.5, f"Distance 1 bias expected -0.5, got {bias_1}"
    assert bias_2 == -1.0, f"Distance 2 bias expected -1.0, got {bias_2}"
    assert bias_5 == -2.5, f"Distance 5 bias expected -2.5, got {bias_5}"
    assert bias_5 < bias_2 < bias_1 < bias_0


def test_layer_norm_standardization():
    """Verify layer_norm_1d normalizes arbitrary input to zero mean and unit variance."""
    x = [2.0, 4.0, 6.0, 8.0, 10.0]
    gamma = [1.0] * 5
    beta = [0.0] * 5
    normed = layer_norm_1d(x, gamma, beta, eps=1e-5)

    mean_val = sum(normed) / len(normed)
    var_val = sum((val - mean_val) ** 2 for val in normed) / len(normed)

    assert abs(mean_val) < 1e-5, f"Mean not centered: {mean_val}"
    assert abs(var_val - 1.0) < 1e-3, f"Variance not scaled to 1.0: {var_val}"


def test_cross_entropy_loss_properties():
    """Verify cross_entropy_with_logits is positive and sensitive to target logits."""
    correct_logits = [0.0, 0.0, 10.0, 0.0]
    wrong_logits = [10.0, 0.0, 0.0, 0.0]

    loss_low = cross_entropy_with_logits(correct_logits, target_idx=2)
    loss_high = cross_entropy_with_logits(wrong_logits, target_idx=2)

    assert loss_low > 0.0
    assert loss_low < 0.01, f"Confidently correct prediction should have near-zero loss: {loss_low}"
    assert loss_high > 9.0, f"Confidently wrong prediction should have high loss: {loss_high}"


def test_warmup_cosine_decay_lr_progression():
    """Verify learning rate linearly ascends during warmup and decays monotonically after."""
    warmup_steps = 5
    total_steps = 20
    base_lr = 0.1
    min_lr = 0.01

    lrs = [warmup_cosine_decay_lr(s, warmup_steps, total_steps, base_lr, min_lr) for s in range(total_steps)]

    # 1. Warmup checks
    assert lrs[0] < lrs[1] < lrs[2] < lrs[3] < lrs[4]
    assert abs(lrs[warmup_steps] - base_lr) < 1e-5

    # 2. Decay checks
    for s in range(warmup_steps, total_steps - 1):
        assert lrs[s] >= lrs[s + 1], f"LR not decaying at step {s}: {lrs[s]} < {lrs[s+1]}"

    # 3. Final step matches min_lr
    assert abs(lrs[-1] - min_lr) < 1e-3


def test_autoregressive_transformer_training_convergence():
    """Verify synthetic training loop reduces cross-entropy loss monotonically."""
    res = simulate_autoregressive_transformer_training(
        seq_len=4, dim=4, steps=12, warmup_steps=3, base_lr=0.08, min_lr=0.005
    )

    assert res["loss_decreased"] is True
    assert res["final_loss"] < res["initial_loss"] * 0.5, (
        f"Loss should decrease by at least 50%: {res['initial_loss']} -> {res['final_loss']}"
    )


def test_composed_concepts_registered_in_catalog():
    """Verify that all 6 composed building blocks exist in machine_learning.json."""
    cat_path = repo_root / "data" / "domains" / "machine_learning.json"
    assert cat_path.is_file()
    with open(cat_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    catalog_names = {c["name"] for c in concepts}
    required = {
        "attention_dot_product_scaled",
        "causal_attention_mask_apply",
        "alibi_relative_bias_scalar",
        "layer_norm_1d",
        "cross_entropy_with_logits",
        "warmup_cosine_decay_lr",
    }
    assert required.issubset(catalog_names), f"Missing concepts: {required - catalog_names}"
