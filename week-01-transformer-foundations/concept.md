# Week 01 — Concept Walkthrough

Practitioner-depth reading for the Tue/Thu concept slots. Read this, then the
code in `src/` is just the same ideas made executable. Each section ends with the
one thing worth keeping.

---

## 1. Embeddings: discrete tokens to continuous space

A token ID (`4592`) carries no usable structure — ID 4592 isn't "closer" to 4593
than to 17. An embedding is a lookup table `E` of shape `(vocab_size, d_model)`;
token `i` becomes the row `E[i]`, a learned vector in continuous space. Now
similarity is geometric: "king" and "queen" can sit near each other, and the
model can do arithmetic on meaning.

The table is learned, not designed. Gradients pull the vectors into an
arrangement that makes the downstream task easier. The same matrix (transposed)
often projects the final hidden state back to vocab logits at the output — weight
tying, which saves parameters and tends to help.

**Keep:** embeddings turn arbitrary IDs into a geometry the model can reason
over. Everything after operates on vectors, never on IDs.

---

## 2. Attention as a soft, content-based lookup

The core question every token asks: "given what I am, which other tokens should I
pull information from, and how much?"

Frame it as a dictionary lookup. A hard dictionary: you have a **query**, you
match it against **keys**, and you retrieve the **value** of the exact match.
Attention softens this — instead of one exact match, every key gets a relevance
score, those scores become weights, and you retrieve a *weighted blend* of all
values.

- **Query (Q):** what this token is looking for.
- **Key (K):** what each token offers as an index.
- **Value (V):** the actual content each token contributes if attended to.

Q, K, V are three different learned linear projections of the same input (in
self-attention). The model learns to project so that "looking-for" and
"offering" align when two tokens should interact.

**Keep:** attention is a differentiable soft lookup. Q matches against K to decide
weights; the output is a weighted sum of V.

---

## 3. Scaled dot-product attention, and why the sqrt(d)

The formula:

    Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_head) ) V

Step by step, with shapes (L = sequence length, d = d_head):

1. `Q Kᵀ` → `(L, L)`. Entry (i, j) is the dot product of query i with key j: how
   aligned they are. This is the score matrix.
2. Divide by `sqrt(d)`. Here's the why: if the components of q and k are roughly
   independent with variance ~1, then their dot product `q·k = Σ qₘ kₘ` (over d
   terms) has variance ~d. So as d_head grows, the raw scores grow in magnitude
   like sqrt(d). Feed large-magnitude logits into softmax and it saturates —
   nearly all mass on one entry, a near one-hot. A saturated softmax has
   vanishing gradient (its Jacobian collapses), so the model can't learn. Dividing
   by sqrt(d) rescales the logits back to ~unit variance, keeping softmax in its
   responsive regime.
3. `softmax` over the last dim → each query row becomes a probability distribution
   over keys (rows sum to 1).
4. Multiply by V → `(L, d)`. Each output row is the blend of value vectors,
   weighted by how much that query attended to each key.

You verified exactly this in `attention.py`: rows summed to 1.0, and your
implementation matched `F.scaled_dot_product_attention` to ~1e-7.

**Keep:** the sqrt(d) is not cosmetic. Without it, deeper/wider attention
saturates softmax and kills gradients. It keeps logit variance ~constant across
model sizes.

---

## 4. Multi-head: parallel subspaces

One attention over the full d_model can only express one notion of "relevance" at
a time — a single weighting of tokens. Real language needs several at once: one
relationship might be syntactic (verb→subject), another positional (adjacent
tokens), another semantic (coreference).

Multi-head splits d_model into H heads of size d_head = d_model/H. Each head gets
its own Q/K/V projections, attends independently in its smaller subspace, and the
H outputs are concatenated and passed through an output projection. The cost is
roughly the same as one full-width attention (the dimensions divide), but you get
H specialized views computed in parallel and combined.

Why not just one big head? Because a single softmax forces a single weighting.
Multiple heads let the model attend to different things simultaneously and merge
them — strictly more expressive for the same compute.

**Keep:** heads are parallel relationship detectors in lower-dim subspaces.
Splitting then concatenating buys expressiveness at ~no extra cost.

---

## 5. Positional encoding: why order must be injected

Attention is permutation-equivariant: shuffle the input tokens and the outputs
shuffle the same way, but no token "knows" it moved. The dot products don't
encode position at all. Yet "dog bites man" ≠ "man bites dog" — order is meaning.

So position must be added explicitly. The original transformer uses sinusoidal
positional encodings: for position `pos` and dimension `i`, a fixed pattern of
sines and cosines at geometrically-spaced frequencies, added to the token
embedding. Properties that matter:

- Each position gets a unique, deterministic signature.
- The encoding for `pos + k` is a linear function of the encoding for `pos`,
  which gives the model a consistent way to reason about *relative* offsets.
- It extrapolates (in principle) to lengths not seen in training, since it's a
  formula, not a learned table.

Modern models often replace this with learned positional embeddings or, more
commonly now, RoPE (rotary) — which you'll meet in **Week 03**. The principle is
unchanged: attention is order-blind, so order is injected.

**Keep:** attention has no built-in sense of order; positional encoding supplies
it. Sinusoidal is the classic; RoPE is the modern default (Week 03).

---

## 6. Masked and cross-attention

Three flavors, same machinery, different Q/K/V wiring and masking:

- **Self-attention:** Q, K, V all from the same sequence. Tokens attend to each
  other within one input.
- **Masked (causal) self-attention:** self-attention plus a mask that sets scores
  for future positions to −∞ *before* softmax, so token i can attend only to
  positions ≤ i. This is what makes a decoder autoregressive.
- **Cross-attention:** Q from one sequence (e.g. the decoder), K and V from
  another (e.g. the encoder output). This is how a translation model conditions
  its output on the source sentence.

The causal mask deserves emphasis. During training we feed the whole target
sequence at once for efficiency (teacher forcing). Without the mask, position i
could attend to positions i+1, i+2, … — i.e. peek at the answer it's supposed to
predict. The model would learn to "predict" the next token by copying it, score
beautifully in training, and collapse at generation time when the future doesn't
exist yet. The mask forces every position to depend only on the past, so training
matches inference. You saw the mask working in `attention.py`: max weight on any
future position was exactly 0.

**Keep:** the causal mask makes training causally honest. Omit it in a decoder and
the model cheats during training and fails at generation.

---

## 7. The architecture trichotomy: encoder / decoder / encoder-decoder

All three are stacks of attention + feed-forward blocks. The difference is
masking and what they're built to do.

- **Encoder-only** (BERT, DistilBERT): bidirectional self-attention — every token
  sees every other token (no causal mask). Great for *understanding* the whole
  input at once: classification, named-entity recognition, retrieval embeddings.
  Trained with masked-language-modeling (predict held-out tokens). Not built to
  generate text left-to-right.

- **Decoder-only** (GPT, Llama, Mistral): causal self-attention — each token sees
  only the past. Built for *generation*: predict the next token, autoregressively.
  This is the architecture of essentially every modern LLM. Trained with
  causal-language-modeling.

- **Encoder-decoder** (T5, original Transformer, Whisper): an encoder reads the
  input bidirectionally; a decoder generates output causally while
  cross-attending to the encoder's representation. Built for *sequence
  transduction* where input and output differ: translation, summarization,
  speech-to-text.

Reach for which:
- Need a fixed-size understanding of text (classify, embed, tag) → **encoder-only**.
- Need open-ended generation / chat / completion → **decoder-only**.
- Need to transform one sequence into a different sequence with a distinct input
  to attend back to (translate, summarize, transcribe) → **encoder-decoder**.

You'll fine-tune one of each in **Week 02** (DistilBERT, DistilGPT, T5), which is
the concrete payoff of this distinction.

**Keep:** masking + training objective define the family. Encoder = understand,
decoder = generate, encoder-decoder = transduce.

---

## 8. Tokenization: BPE vs WordPiece vs SentencePiece

Models don't see characters or words — they see token IDs from a fixed vocab.
Subword tokenization is the compromise between two bad extremes: word-level (huge
vocab, can't handle unseen words) and character-level (tiny vocab, sequences far
too long). Subwords keep common words whole and break rare words into reusable
pieces, so there's never a true "unknown word."

The three you compared:

- **BPE (Byte-Pair Encoding)** — GPT-2, Llama. Start from characters/bytes;
  repeatedly merge the most frequent adjacent pair into a new token; stop at the
  target vocab size. *Byte-level* BPE (GPT-2) operates on raw bytes, so it can
  encode literally any input (emoji, code, any language) with no unknowns.

- **WordPiece** — BERT. Similar merging idea, but picks the merge that most
  increases training-corpus likelihood rather than raw frequency. Marks
  word-continuations with `##` (you saw `transform` + `##er`).

- **SentencePiece (Unigram)** — T5, many multilingual models. Treats the input as
  a raw stream (spaces become a real symbol, `▁`), so it's language-agnostic and
  needs no pre-tokenization. The Unigram variant starts from a large candidate
  vocab and *prunes* down, keeping pieces that best explain the corpus
  probabilistically.

Tie this to the exact numbers from your run:

- Ordinary sentence (#1): 8 / 9 / 8. Roughly one token per word — the easy case.
- Rare/compound words (#2): 14 / 17 / 19. `antidisestablishmentarianism`
  shattered into many subwords. No tokenizer has a single token for it; they
  fragment instead of failing. This is subword tokenization doing its job.
- Code (#3): 27 / 29 / 33. Symbols (`@`, `**`, `.`, `(`) each tend to become their
  own token. Code is token-expensive, which is why pasting code burns context
  fast.

Notice the ranking *flips* by text type — no tokenizer is universally "fewest."
Token count is workload-dependent.

**Keep:** subwords avoid unknown words by fragmenting rare ones. Algorithm and
vocab determine token counts, which drive context budget and cost — so measure on
*your* workload, and always pair a model with its own tokenizer (mismatched vocab
= gibberish).

---

## Where this leads

- **Week 02** turns these ideas into practice: code attention, then fine-tune one
  model of each architecture family (DistilBERT / DistilGPT / T5).
- **Week 03** makes attention *fast and cheap*: KV cache, FlashAttention, the
  MQA/GQA/MLA variants, RoPE — all of which exist because of the O(L²) cost of the
  `Q Kᵀ` step you implemented here.

The O(L²) score matrix is the single most important cost in the stack. Hold onto
that — most of the "make LLMs efficient" content later is some way of not paying
the full price of it.
