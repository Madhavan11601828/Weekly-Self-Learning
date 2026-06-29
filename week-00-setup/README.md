# Week 00 — Repo & Reproducible-Env Scaffold

> Source: — · Status: 🟩
> Prereqs from prior weeks: none

## Objectives
- Stand up the `ml-learning` repo so every later week is self-contained and runnable.
- Lock a reproducible Python toolchain (3.11 + uv) with per-week pinned deps.
- Establish conventions (folder layout, commit style, the three-layer + challenges format).
- Verify the environment can import the core ML stack and detect GPU/CPU correctly.

## Layers covered
- **Basics** — repo layout, env manager choice, why pinning matters.
- **Intermediate** — per-week dependency isolation, scaffolding automation.
- **Advanced** — keeping the repo portfolio-clean (artifacts gitignored, deterministic setup).

## Deliverable
A working repo skeleton:
- `README.md` master index (the cross-chat source of truth)
- `templates/` (week README + notes + requirements)
- `scripts/new_week.sh` to scaffold future weeks
- `week-00-setup/src/verify_env.py` — environment sanity check

## How to run
Full Windows-first walkthrough is in [`../SETUP.md`](../SETUP.md). Short version:
```bash
# install uv first: https://docs.astral.sh/uv/
uv python install 3.11
uv venv --python 3.11 && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
uv pip install -r week-00-setup/requirements.txt
python week-00-setup/src/verify_env.py
```

## Key takeaways
- One repo, one folder per week, one pinned `requirements.txt` per week.
- `new_week.sh NN slug` scaffolds a consistent week in one command.
- The master `README.md` table is updated at the **end** of every week — it's what
  future chats read to answer "have we covered this already?".

## Real-World Implementation & Challenges
- **Where this shows up in production:** reproducible envs are the difference between
  "works on my machine" and a CI job that passes deterministically. The same uv/pinned-deps
  discipline scales straight into the P5 LLMOps pipeline.
- **Common failure modes:** unpinned transitive deps silently upgrading (breaking tokenizers
  or torch/CUDA compatibility); committing multi-GB model weights into git history.
- **How it's addressed:** pin exact versions per week; gitignore all model/artifact formats;
  keep Python at 3.11 (several ML libs lag on 3.12).
- **Open questions / things to revisit:** revisit whether to add Docker at the repo level
  once P1 needs GPU serving (vLLM) — likely yes for parity with deployment.

## References
- uv docs: https://docs.astral.sh/uv/
- Conventional Commits: https://www.conventionalcommits.org/
