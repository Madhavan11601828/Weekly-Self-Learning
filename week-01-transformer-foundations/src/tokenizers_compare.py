"""Compare subword tokenizers: BPE vs WordPiece vs SentencePiece (Unigram).

Same text, three algorithms, side by side. The point is to *see* how each splits
words -- especially rare/compound words and code -- and how that drives token
counts, which in turn drive context usage and API cost.

We use well-known pretrained tokenizers as concrete representatives:
    BPE        -> GPT-2          (byte-level BPE)
    WordPiece  -> bert-base-uncased
    Unigram/SP -> google-t5/t5-small  (SentencePiece, unigram model)

Run:
    python src/tokenizers_compare.py

First run downloads small tokenizer files (~a few MB total) to your HF cache.
CPU-only, no model weights loaded -- just the tokenizers.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer

# A handful of sentences chosen to expose differences:
#  - ordinary words           -> mostly whole-word tokens everywhere
#  - a rare/compound word      -> where subword splitting shows up
#  - whitespace / code         -> byte-level BPE vs the others diverge here
SAMPLES = [
    "The transformer architecture changed natural language processing.",
    "Tokenization handles antidisestablishmentarianism and hyperparameterization.",
    "def attention(q, k, v): return softmax(q @ k.T / d**0.5) @ v",
]


@dataclass
class Spec:
    label: str
    model: str


SPECS = [
    Spec("BPE (GPT-2)", "gpt2"),
    Spec("WordPiece (BERT)", "bert-base-uncased"),
    Spec("Unigram/SP (T5)", "google-t5/t5-small"),
]


def load_tokenizers() -> list[tuple[Spec, object]]:
    loaded = []
    for spec in SPECS:
        try:
            tok = AutoTokenizer.from_pretrained(spec.model)
        except Exception as e:  # network/cache miss -- don't crash the whole run
            print(f"  [skip] could not load {spec.label} ({spec.model}): {type(e).__name__}")
            print("         (need internet on first run to fetch the tokenizer files)")
            continue
        loaded.append((spec, tok))
    if not loaded:
        raise SystemExit(
            "No tokenizers could be loaded. Connect to the internet once so the "
            "tokenizer files download into your HF cache, then re-run."
        )
    return loaded


def show_split(text: str, tokenizers: list[tuple[Spec, object]]) -> None:
    print(f"\nTEXT: {text!r}")
    for spec, tok in tokenizers:
        pieces = tok.tokenize(text)
        # Normalize the visual marker each family uses for word boundaries:
        #   GPT-2 byte-level uses 'Ġ' for a leading space
        #   BERT WordPiece uses '##' for word-continuation
        #   T5 SentencePiece uses '▁' for a leading space
        pretty = [p.replace("Ġ", "·").replace("▁", "·") for p in pieces]
        print(f"  {spec.label:<18} {len(pieces):>3} tokens: {pretty}")


def summary_table(tokenizers: list[tuple[Spec, object]]) -> None:
    print("\n" + "=" * 70)
    print("Token-count summary (lower = fewer tokens to encode the same text)")
    header = "  " + "text".ljust(8) + "".join(f"{spec.label:<20}" for spec, _ in tokenizers)
    print(header)
    for i, text in enumerate(SAMPLES):
        row = "  " + f"#{i+1}".ljust(8)
        for _, tok in tokenizers:
            row += f"{len(tok.tokenize(text)):<20}"
        print(row)
    print("\nNotes:")
    print("  '·' marks a word-start (leading space). '##' marks a WordPiece continuation.")
    print("  Watch the rare/compound word (#2) fragment into many subwords,")
    print("  and the code line (#3) where byte-level BPE handles symbols differently.")


def main() -> None:
    print("Loading tokenizers (first run downloads small files to HF cache)...")
    tokenizers = load_tokenizers()
    for text in SAMPLES:
        show_split(text, tokenizers)
    summary_table(tokenizers)


if __name__ == "__main__":
    main()
