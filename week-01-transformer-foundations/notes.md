# Week 01 — Concept Notes

## Mental model
A transformer layer is: "every token looks at every other token, decides how much
each one matters (attention), and updates itself with a weighted blend of them."
Stack that, add positional info and feed-forward mixing, and you get a model that
builds context-aware representations in parallel (no recurrence, no left-to-right
bottleneck).

## The math that matters
- Attention(Q,K,V) = softmax(QK^T / sqrt(d_head)) V
  - QK^T: alignment of each query with each key -> (L, L) score matrix
  - /sqrt(d_head): keeps logit variance ~constant as dimension grows; without it
    softmax saturates toward one-hot and gradients vanish
  - softmax over keys: each query gets a probability distribution (rows sum to 1)
  - @ V: the answer is a weighted blend of value vectors
- Multi-head: split D into H heads of Dh = D/H, attend independently, concat,
  project. H different "views" of relationships, used simultaneously.

## Gotchas
- **Self vs cross attention:** self = Q,K,V from the same sequence; cross = Q from
  the decoder, K,V from the encoder (how seq2seq conditions on the source).
- **Causal mask is non-negotiable in decoders:** set scores for future positions
  to -inf *before* softmax, so position i never attends to j>i. Forget it and the
  model trains by peeking at the answer -> looks great in training, fails at
  generation.
- **Tokenizer markers differ:** GPT-2 byte-level BPE uses 'Ġ' for a leading space;
  BERT WordPiece uses '##' for continuations; T5 SentencePiece uses '▁'. Same idea
  (where does a word start), different notation.
- **Tokenizer must match the model.** A model is trained against one vocab; feeding
  it another tokenizer's IDs is gibberish in -> gibberish out.

## Connections
- Builds on: nothing (foundation).
- Feeds into: Week 02 (attention from code -> fine-tuning the three architectures),
  Week 03 (making this attention fast: KV cache, FlashAttention, MQA/GQA, RoPE).
- The O(L^2) cost of QK^T is *the* reason long context is expensive — Week 03
  exists to attack exactly this.

## References
- Vaswani et al. 2017, "Attention Is All You Need"
- Kudo & Richardson 2018, SentencePiece
