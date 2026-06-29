# Week 01 — Verify

Week is "done" (flip master index row to 🟩) only when these pass.

## Concept self-check (answer each in a sentence or two)
- [ ] **Basics:** What do Q, K, and V represent, and why is attention called a
      "soft lookup"?
- [ ] **Basics:** Why embed tokens into continuous space instead of using IDs?
- [ ] **Intermediate:** Why divide attention scores by sqrt(d_head)? What breaks
      without it?
- [ ] **Intermediate:** What does multi-head buy you over a single large head?
- [ ] **Intermediate:** Why do transformers need positional encoding at all?
- [ ] **Advanced:** What exactly does the causal mask do, and why would a decoder
      train incorrectly without it?
- [ ] **Advanced:** When would you reach for encoder-only vs decoder-only vs
      encoder-decoder?
- [ ] **Advanced:** Give one concrete reason token count matters in production.

## Runnable verification
```bash
python src/attention.py
```
- [ ] Prints correct shapes (B,L,D in/out; B,H,L,L weights)
- [ ] Causal-mask check shows ~0 weight on future positions
- [ ] "All checks passed." — from-scratch attention matches F.scaled_dot_product_attention

```bash
python src/tokenizers_compare.py
```
- [ ] Shows three tokenizers splitting the same text differently
- [ ] You can explain why the rare/compound word fragments

## Honest gaps
- Solid on: <…>
- Shaky on (revisit): <…>

## Done criteria
- [ ] All concept checks answered confidently
- [ ] Both scripts run; attention check passes
- [ ] notes.md + progress.md filled in
- [ ] Master index row 01 flipped to 🟩 with a one-line summary
