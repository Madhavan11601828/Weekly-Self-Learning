# Week 01 — Transformer Architecture & Tokenization

> Source: Module 01 · Status: 🟦

How transformers turn tokens into meaning: embeddings, attention (self /
multi-head / masked / cross), encoder vs decoder architectures, and the
tokenization step (BPE / WordPiece / SentencePiece) that feeds them.

## The four parts
1. **[Plan](plan.md)** — objectives, layers, deliverable
2. **[Progress](progress.md)** — day-by-day learning + project progress
3. **[Verify](verify.md)** — concept checklist + runnable proof
4. **[Notes](notes.md)** — concept notes & gotchas

Concept reading (Tue/Thu): **[concept.md](concept.md)** — practitioner-depth
walkthrough of attention, positional encoding, the architecture trichotomy, and
tokenization.

## How to run
```bash
# from the repo root, with the env active (see SETUP.md):
uv pip install -r week-01-transformer-foundations/requirements.txt

python week-01-transformer-foundations/src/attention.py          # from-scratch attention + correctness check
python week-01-transformer-foundations/src/tokenizers_compare.py # BPE vs WordPiece vs SentencePiece
```

Expected: `attention.py` prints shape-annotated steps, a causal-mask check, and
`All checks passed.` (from-scratch attention matches `F.scaled_dot_product_attention`).
`tokenizers_compare.py` prints how each algorithm splits the same text and a
token-count summary. (First tokenizer run needs internet to fetch files.)

## Real-World Implementation & Challenges
- **Where this shows up in production:** every LLM call is tokenize -> attention
  stack -> detokenize. Token counts set your context budget and per-call cost;
  attention cost is what makes long context expensive (it's O(L^2)).
- **Common failure modes:** mismatched tokenizer vs model (garbage output);
  token explosion on code/multilingual text inflating cost; forgetting the causal
  mask in a decoder (the model "cheats" by seeing future tokens and trains wrong).
- **How it's addressed:** always pair a model with its own tokenizer; measure
  token counts on representative inputs before estimating cost; the O(L^2) wall is
  exactly what KV-cache and FlashAttention (Week 03) attack.
