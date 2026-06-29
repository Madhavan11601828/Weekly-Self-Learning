# Week 01 — Plan

> Source: Module 01 · Status: planning
> Prereqs from prior weeks: none (this is the foundation everything builds on)

## Objectives
By the end of this week I can:
- Explain why attention replaced recurrence, and what Q/K/V actually represent.
- Implement scaled dot-product and multi-head self-attention from scratch and
  prove it correct against PyTorch's primitive.
- Explain the causal mask and why decoders need it.
- Distinguish encoder-only / decoder-only / encoder-decoder and what each is for.
- Explain how BPE, WordPiece, and SentencePiece differ, and why token counts matter.

## Layers (basics -> intermediate -> advanced)
- **Basics** — embeddings (discrete -> continuous); the Q/K/V intuition; attention
  as a soft, content-based lookup.
- **Intermediate** — scaled dot-product attention with explicit shapes; why the
  sqrt(d) scaling; multi-head as parallel subspaces; sinusoidal positional encoding
  and why order must be injected; masked vs cross-attention.
- **Advanced** — encoder vs decoder vs encoder-decoder and their use cases; the
  causal mask in autoregressive generation; tokenization's downstream impact
  (cost, rare words, multilingual, code).

## Deliverable
`src/attention.py` — from-scratch SDPA + multi-head self-attention, shape-annotated,
with a causal-mask check and an equivalence check vs `F.scaled_dot_product_attention`.
`src/tokenizers_compare.py` — BPE (GPT-2) vs WordPiece (BERT) vs SentencePiece (T5)
on the same text, showing splits and token counts.

## Scope guardrails
- In scope: self-attention, multi-head, causal masking, tokenizer comparison.
- Out of scope (deferred): KV cache, FlashAttention, MQA/GQA/MLA, RoPE -> all Week 03.
  Full encoder-decoder training -> Week 02. Embedding *models* for retrieval -> Week 14.

## References
- "Attention Is All You Need" (Vaswani et al., 2017)
- The Illustrated Transformer (Jay Alammar)
- HF tokenizers docs; SentencePiece paper (Kudo & Richardson, 2018)
