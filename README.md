# ml-learning

Production LLM Engineering — a structured, basics → advanced journey through
Transformers, fine-tuning, RAG, agents, and LLMOps, built one week at a time.

Each week is self-contained and runnable. Each topic is taught in three layers
(**basics → intermediate → advanced**) and closes with a **Real-World
Implementation & Challenges** section: the production failure modes for that
topic and how they're addressed.

This file is the **master index**. It is the source of truth for "what have we
covered already" and is updated at the end of every week.

---

## How to use this repo

> First time here? Follow [`SETUP.md`](SETUP.md) for the full Windows-first
> environment bootstrap. Quick version below.

```bash
# one-time: install uv (https://docs.astral.sh/uv/) then:
uv python install 3.11
uv venv --python 3.11 && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

# per week: activate the env (if not already), then install that week's pinned deps
source .venv/bin/activate                            # Windows: .venv\Scripts\Activate.ps1
uv pip install -r week-01-<slug>/requirements.txt

# scaffold a new week folder from the templates
./scripts/new_week.sh 01 transformer-foundations
```

Conventions: Python 3.11, type hints where they add clarity, pinned deps per
week, small reproducible examples. Commits follow Conventional Commits
(`feat:`, `docs:`, `fix:`, `chore:`).

---

## Layer legend

| Mark | Meaning |
|------|---------|
| ⬜ | not started |
| 🟦 | basics reached |
| 🟨 | intermediate reached |
| 🟩 | advanced reached (week complete) |

---

## Plan (interleaved projects · ~40 weeks · ~10 months @ ~8h/week)

Cadence: concept on Tue/Thu mornings, hands-on build + deliverable on the
weekend. Project weeks (P1–P5) run weekend-dominant at ~12–15h and may span
the listed range loosely.

| Week | Topic | Source | Status | One-line summary |
|------|-------|--------|--------|------------------|
| 00 | Repo & reproducible-env scaffold | — | 🟩 | Master index, per-week template, uv env, env-verify script |
| 01 | Transformer Architecture & Tokenization | M1 | ⬜ | Embeddings, attention, encoder/decoder, BPE/WordPiece/SentencePiece |
| 02 | Fine-Tuning Transformer Architectures in Practice | M2 | ⬜ | Attention from scratch; fine-tune DistilBERT / DistilGPT / T5 |
| 03 | Inference Optimization, Attention Variants & Scaling Laws | M3 | ⬜ | KV cache, FlashAttention, MQA/GQA/MLA, RoPE, Chinchilla |
| 04 | LLM Fine-Tuning Lifecycle & Pre-Training | M4 | ⬜ | Pre vs post-training, CPT, CLM/MLM/Prefix-LM, MTP |
| 05 | Data Prep & Synthetic Dataset Generation | M5 | ⬜ | Chat templates, loss masking, Self-Instruct, LLM-as-Judge, collapse |
| 06–07 | SFT, PEFT & Preference Alignment | M6 | ⬜ | LoRA/QLoRA/DoRA, SFT, RLHF-PPO, DPO |
| 08 | Evaluation, Quantization, Deployment & Tooling | M7 | ⬜ | Benchmarks, GPTQ/AWQ/GGUF, vLLM, TRL/Unsloth/Axolotl |
| 09 | MoE + Reasoning Models & RL-Only Training | M8+M9 | ⬜ | Sparse experts, load balancing; CoT, DeepSeek-R1-Zero, distillation |
| 10 | Small Language Models & Knowledge Distillation | M10 | ⬜ | Soft labels, temperature, KL loss, attention transfer, pruning |
| 11–13 | **P1 · MedScript** — Domain LLM Fine-Tuning | Project 01 | ⬜ | QLoRA SFT + DPO + multi-adapter vLLM on AWS |
| 14–15 | **P2 · EdgeReason** — Distillation for CPU | Project 02 | ⬜ | KL + attention transfer from scratch → GGUF → llama.cpp |
| 16 | Vision Foundations: CNNs → ViTs | M11 | ⬜ | Patch embedding, CLS token, CLIP/SigLIP/DINOv2 |
| 17 | VLMs + Speech-to-Text (Whisper) | M12+M13 | ⬜ | Encoder+projector+LLM; Whisper architecture & fine-tuning |
| 18 | Embedding Models: Taxonomy, Matryoshka & Fine-Tuning | M14 | ⬜ | Dense→binary, MRL, embedding fine-tuning |
| 19–20 | LangChain for RAG: Ecosystem, Patterns & Production | M15 | ⬜ | LCEL, memory, tools, LangSmith, deploy (full depth) |
| 21 | RAG Foundations: Vanilla, Embeddings, Chunking, Retrieval | M16 | ⬜ | Baseline RAG, chunking, BM25/SPLADE/ColBERT |
| 22 | Advanced RAG: Query Transforms, Rerankers, Adaptive | M17 | ⬜ | Hybrid, Self/Corrective/Adaptive RAG, Agentic RAG |
| 23–24 | Vector Quantization, Multimodal & Graph RAG | M18 | ⬜ | SQ/BQ/PQ, ColPali, Neo4j GraphRAG, semantic cache, guardrails |
| 25–27 | **P3 · LexisGraph** — Enterprise Legal RAG | Project 03 | ⬜ | ColPali + Neo4j + hybrid + RRF + RAGAS-gated |
| 28 | Agent Foundations: Structured Outputs, Function Calling & MCP | M19 | ⬜ | Pydantic, cross-provider tool calls, MCP servers/clients (full) |
| 29 | LangGraph: Stateful, Multi-Agent & HITL | M20 | ⬜ | Graph state, ReAct, HITL, persistence, multi-agent |
| 30–31 | Production Agents: Observability, A2A & Bedrock AgentCore | M21 | ⬜ | A2A protocol, AgentCore runtime/memory/gateway/guardrails |
| 32–34 | **P4 · AutoOps** — DevOps Multi-Agent w/ HITL & A2A | Project 04 | ⬜ | LangGraph supervisor + MCP + A2A + policy access control |
| 35 | Prompt Engineering: Structure, Techniques & Refinement | M22 | ⬜ | Anatomy, CoT/step-back, structured output, self-refine (full) |
| 36 | Context Engineering: Memory, Pruning & RAG-Context | M23 | ⬜ | Context window anatomy, memory architectures, LLMLingua/RECOMP |
| 37 | Harness Engineering: Evaluation, Benchmarking & Agent CI/CD | M24 | ⬜ | lm-eval-harness, Inspect AI, PromptFoo, eval-gated CI |
| 38–40 | **P5 · ShipLLM** — LLMOps CI/CD for all projects | Project 05 | ⬜ | Eval-gated pipeline, blue/green, auto-rollback wrapping P1–P4 |

> Notes: M8+M9 and M12+M13 are paired because each is a single topic group.
> M6, M15, M18, M21 are split across two weeks due to volume. Projects are
> placed immediately after their prerequisite track completes.

---

## Cross-reference map (what builds on what)

- M3 attention variants → extends M1/M2 attention mechanics
- M4 compute budget → calls back to M3 scaling laws
- M6 QLoRA → uses M7 4-bit quant concepts; P1 executes both
- M10 distillation → P2 implements the loss from scratch
- M14 embeddings → underpins M16–M18 retrieval
- M19 MCP + M20 LangGraph + M21 A2A → composed in P4
- M24 harness → wraps everything in P5

When a new topic overlaps earlier work, the week's notes link back here rather
than re-teaching.
