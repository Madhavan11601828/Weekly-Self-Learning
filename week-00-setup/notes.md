# Week 00 — Concept Notes

## Mental model
The repo is a lab notebook that also runs. Every week is an isolated experiment
with its own pinned deps; the master README is the lab index. If a future-me (or
a separate chat) can't tell what's been covered from the README table, the index
has failed its job.

## Decisions and why
- **uv over conda/poetry:** fast, lockable, one tool for Python + venv + pip.
- **Python 3.11, not 3.12:** transformers/TRL/vLLM and CUDA wheels are most
  reliable on 3.11 as of this curriculum.
- **Per-week requirements.txt:** week 2's transformers pin shouldn't silently
  break week 18's retrieval stack. Isolation > convenience.
- **Artifacts gitignored:** weights, checkpoints, wandb runs never enter git —
  keeps the repo reviewable as a portfolio.

## Gotchas
- `verify_env.py` treats missing packages as informational, not fatal — only the
  Python version is a hard gate, so it can double as a CI check later.
- The MPS/CUDA branch means the same script is honest on a laptop and a GPU box.

## Connections
- Builds on: nothing.
- Feeds into: every week; the env discipline is reused wholesale in P5 (ShipLLM).
