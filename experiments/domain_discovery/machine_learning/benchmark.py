"""Scaled Transformer Multi-Head Self-Attention Benchmark.

Composes 6 verified OpenCommons concepts from the machine_learning domain:
1. `attention_dot_product_scaled`: Scaled dot-product query-key attention projection with variance normalization.
2. `causal_attention_mask_apply`: Upper-triangular autoregressive masking enforcing strict temporal causality.
3. `alibi_relative_bias_scalar`: Attention with Linear Biases (ALiBi) geometric positional penalty.
4. `layer_norm_1d`: Numerically stable channel-wise standardization with affine scale/shift.
5. `cross_entropy_with_logits`: Log-sum-exp stabilized negative log likelihood loss.
6. `warmup_cosine_decay_lr`: Learning rate scheduling with linear warmup and half-cycle cosine annealing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable


def _load_domain_functions() -> dict[str, Callable[..., Any]]:
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "machine_learning.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    env: dict[str, Any] = {"math": math}
    for c in concepts:
        code_obj = compile(c["source_code"], f"<{c['name']}>", "exec")
        exec(code_obj, env)
    return {c["name"]: env[c["name"]] for c in concepts}


_FUNCS = _load_domain_functions()
attention_dot_product_scaled = _FUNCS["attention_dot_product_scaled"]
causal_attention_mask_apply = _FUNCS["causal_attention_mask_apply"]
alibi_relative_bias_scalar = _FUNCS["alibi_relative_bias_scalar"]
layer_norm_1d = _FUNCS["layer_norm_1d"]
cross_entropy_with_logits = _FUNCS["cross_entropy_with_logits"]
warmup_cosine_decay_lr = _FUNCS["warmup_cosine_decay_lr"]


def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax over a 1D vector."""
    if not logits:
        return []
    max_l = max(logits)
    exps = [math.exp(max(-80.0, l - max_l)) for l in logits]
    sum_exps = sum(exps)
    if sum_exps == 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [e / sum_exps for e in exps]


def transformer_causal_self_attention_layer(
    q: list[list[float]],
    k: list[list[float]],
    v: list[list[float]],
    head_dim: int,
    alibi_slope: float = 0.5,
    gamma: list[float] | None = None,
    beta: list[float] | None = None,
    mask_value: float = -1e9,
) -> dict[str, Any]:
    """Executes a single causal self-attention transformer head with ALiBi and LayerNorm.

    Args:
        q: Query matrix [seq_len x head_dim]
        k: Key matrix [seq_len x head_dim]
        v: Value matrix [seq_len x head_dim]
        head_dim: Dimensionality of attention keys/queries
        alibi_slope: Head-specific geometric ALiBi positional slope
        gamma: Affine gain for LayerNorm (defaults to 1.0)
        beta: Affine bias for LayerNorm (defaults to 0.0)
        mask_value: High negative scalar for masked future positions

    Returns:
        Dictionary containing raw logits, causal masked logits, attention weights,
        attention context output, and normalized output.
    """
    seq_len = len(q)
    assert len(k) == seq_len and len(v) == seq_len

    # 1. Scaled dot-product attention logits for each query across all keys
    raw_logits_matrix: list[list[float]] = []
    for i in range(seq_len):
        query_i = q[i]
        logits_row = attention_dot_product_scaled(query_i, k, head_dim)
        # 2. Add ALiBi relative position bias
        for j in range(seq_len):
            bias = alibi_relative_bias_scalar(i, j, alibi_slope)
            logits_row[j] += bias
        raw_logits_matrix.append(logits_row)

    # 3. Apply causal upper-triangular mask
    masked_logits = causal_attention_mask_apply(raw_logits_matrix, mask_value)

    # 4. Softmax over key positions to produce attention probability distribution
    attention_weights: list[list[float]] = []
    for row in masked_logits:
        attention_weights.append(_softmax(row))

    # 5. Weighted combination of value vectors
    context_vectors: list[list[float]] = []
    for i in range(seq_len):
        weights_i = attention_weights[i]
        ctx = [0.0] * head_dim
        for j in range(seq_len):
            w = weights_i[j]
            v_j = v[j]
            for d in range(head_dim):
                ctx[d] += w * v_j[d]
        context_vectors.append(ctx)

    # 6. Residual connection + LayerNorm
    if gamma is None:
        gamma = [1.0] * head_dim
    if beta is None:
        beta = [0.0] * head_dim

    normed_output: list[list[float]] = []
    for i in range(seq_len):
        residual = [q[i][d] + context_vectors[i][d] for d in range(head_dim)]
        normed = layer_norm_1d(residual, gamma, beta, eps=1e-5)
        normed_output.append(normed)

    return {
        "raw_logits": raw_logits_matrix,
        "masked_logits": masked_logits,
        "attention_weights": attention_weights,
        "context_vectors": context_vectors,
        "normed_output": normed_output,
    }


def simulate_autoregressive_transformer_training(
    seq_len: int = 4,
    dim: int = 4,
    steps: int = 10,
    warmup_steps: int = 3,
    base_lr: float = 0.05,
    min_lr: float = 0.001,
) -> dict[str, Any]:
    """Simulates autoregressive language model training with cross-entropy loss descent.

    Integrates self-attention, layer normalization, cross-entropy loss, and cosine LR scheduling.
    """
    # Deterministic synthetic embeddings for prompt
    q = [[0.2 * (i + 1) + 0.1 * d for d in range(dim)] for i in range(seq_len)]
    k = [[0.15 * (i + 1) + 0.08 * d for d in range(dim)] for i in range(seq_len)]
    v = [[0.3 * (i + 1) - 0.05 * d for d in range(dim)] for i in range(seq_len)]

    # Linear classifier weights [vocab_size x dim] (vocab_size = 4)
    vocab_size = 4
    w_head = [[0.25 * (c + 1) * (d + 1) for d in range(dim)] for c in range(vocab_size)]

    # Target token sequence to predict at final step
    target_token = 2

    losses = []
    lrs = []

    for step in range(steps):
        # 1. Compute current learning rate via warmup cosine scheduler
        lr = warmup_cosine_decay_lr(step, warmup_steps, steps, base_lr, min_lr)
        lrs.append(lr)

        # 2. Forward pass through causal self-attention layer
        attn_res = transformer_causal_self_attention_layer(q, k, v, head_dim=dim, alibi_slope=0.25)
        last_hidden = attn_res["normed_output"][-1]

        # 3. Compute output logits for last token position
        logits = [sum(w_head[c][d] * last_hidden[d] for d in range(dim)) for c in range(vocab_size)]

        # 4. Compute cross-entropy loss against target token
        loss = cross_entropy_with_logits(logits, target_token)
        losses.append(loss)

        # 5. Gradient descent step on classifier weights
        probs = _softmax(logits)
        grad = [probs[c] - (1.0 if c == target_token else 0.0) for c in range(vocab_size)]
        for c in range(vocab_size):
            for d in range(dim):
                w_head[c][d] -= lr * grad[c] * last_hidden[d]

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
        "lrs": lrs,
        "loss_decreased": losses[-1] < losses[0],
    }
