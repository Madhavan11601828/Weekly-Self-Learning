"""Scaled dot-product and multi-head self-attention, from scratch.

The goal is understanding, not speed: every step is annotated with tensor shapes
so you can see exactly what attention computes. At the end we check the
from-scratch scaled dot-product attention against PyTorch's own
`F.scaled_dot_product_attention` to prove the implementation is correct.

Run:
    python src/attention.py

CPU-only and tiny; finishes in well under a second.

Shape notation used throughout:
    B = batch size
    H = number of heads
    L = sequence length (number of tokens)
    D = model dimension (d_model)
    Dh = per-head dimension = D // H
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def scaled_dot_product_attention(
    q: Tensor,            # (B, H, L, Dh)
    k: Tensor,            # (B, H, L, Dh)
    v: Tensor,            # (B, H, L, Dh)
    causal: bool = False,
) -> tuple[Tensor, Tensor]:
    """The core operation. Returns (output, attention_weights).

    Intuition: each query row asks "how much should I attend to every key?".
    The dot product q·k measures alignment; we scale by sqrt(Dh) to keep the
    logits from blowing up as dimension grows (which would saturate softmax and
    kill gradients); softmax turns logits into a probability distribution over
    keys; the weighted sum over v is the answer.
    """
    Dh = q.shape[-1]

    # Alignment scores between every query and every key.
    # (B, H, L, Dh) @ (B, H, Dh, L) -> (B, H, L, L)
    scores = q @ k.transpose(-2, -1)

    # Scale: without this, variance of the dot product grows with Dh and softmax
    # collapses toward a one-hot, starving gradients. sqrt(Dh) normalizes it.
    scores = scores / math.sqrt(Dh)

    if causal:
        # Autoregressive mask: token i may not see tokens j > i.
        # Upper triangle (above diagonal) set to -inf so softmax zeroes it.
        L = scores.shape[-1]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    # Probability distribution over keys, per query. (B, H, L, L), rows sum to 1.
    weights = F.softmax(scores, dim=-1)

    # Weighted sum of values. (B, H, L, L) @ (B, H, L, Dh) -> (B, H, L, Dh)
    out = weights @ v
    return out, weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention.

    Instead of one D-dimensional attention, run H attentions in parallel on
    Dh-dimensional projections, then concatenate. Each head can specialize in a
    different relationship (e.g. syntactic vs positional), and concatenation lets
    the model use all of them at once.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # One linear each for Q, K, V, plus an output projection.
        # Bias-free to keep the equivalence check with F.sdpa clean.
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        # (B, L, D) -> (B, H, L, Dh)
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        # (B, H, L, Dh) -> (B, L, D)
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(self, x: Tensor, causal: bool = False) -> tuple[Tensor, Tensor]:
        # x: (B, L, D). Self-attention: Q, K, V all come from the same input.
        q = self._split_heads(self.w_q(x))   # (B, H, L, Dh)
        k = self._split_heads(self.w_k(x))   # (B, H, L, Dh)
        v = self._split_heads(self.w_v(x))   # (B, H, L, Dh)

        attn_out, weights = scaled_dot_product_attention(q, k, v, causal=causal)  # (B,H,L,Dh)

        merged = self._merge_heads(attn_out)  # (B, L, D)
        out = self.w_o(merged)                # (B, L, D)
        return out, weights


def _demo() -> None:
    torch.manual_seed(0)
    B, L, D, H = 1, 5, 8, 2
    x = torch.randn(B, L, D)

    mha = MultiHeadSelfAttention(d_model=D, num_heads=H)
    out, weights = mha(x, causal=False)

    print("Multi-head self-attention")
    print(f"  input  x      : {tuple(x.shape)}  (B, L, D)")
    print(f"  output        : {tuple(out.shape)}  (B, L, D)")
    print(f"  attn weights  : {tuple(weights.shape)}  (B, H, L, L)")
    print(f"  weights row 0 sums to ~1: {weights[0, 0, 0].sum().item():.4f}")

    # Causal mask: future positions must be exactly zero-weighted.
    _, cweights = mha(x, causal=True)
    upper = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    max_future = cweights[0, 0][upper].max().item()
    print(f"\nCausal mask check")
    print(f"  max weight on any future position: {max_future:.2e}  (expect ~0)")


def _equivalence_check() -> None:
    """Prove our scaled_dot_product_attention matches PyTorch's own primitive."""
    torch.manual_seed(1)
    B, H, L, Dh = 2, 3, 6, 4
    q, k, v = (torch.randn(B, H, L, Dh) for _ in range(3))

    for causal in (False, True):
        ours, _ = scaled_dot_product_attention(q, k, v, causal=causal)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        ok = torch.allclose(ours, ref, atol=1e-6)
        max_diff = (ours - ref).abs().max().item()
        tag = "causal" if causal else "full  "
        print(f"  [{ 'OK' if ok else 'FAIL' }] {tag}  max abs diff vs F.sdpa: {max_diff:.2e}")
        assert ok, "from-scratch attention diverged from F.scaled_dot_product_attention"


def main() -> None:
    print("=" * 60)
    _demo()
    print("\n" + "=" * 60)
    print("Equivalence check vs torch.nn.functional.scaled_dot_product_attention")
    _equivalence_check()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
