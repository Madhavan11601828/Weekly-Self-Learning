# SETUP — Environment Bootstrap (Windows)

This is the Week 00 deliverable: get a reproducible Python 3.11 + PyTorch
environment running so every later week is runnable. There are no concepts to
study this week -- it's pure setup. Learning starts in Week 01.

Commands below are for PowerShell on Windows. macOS/Linux notes are called out
where they differ.

---

## 0. What you need first

- **Git** -- check with `git --version`. (You already have it.)
- **A terminal** -- PowerShell is fine.
- Your system Python version does NOT matter. You saw `verify_env.py` fail on
  Python 3.12 earlier -- that's expected. uv installs an isolated 3.11 for this
  repo, so the system interpreter is irrelevant.

---

## 1. Install uv

uv is a fast Python package + environment manager (one tool for Python, venvs,
and pip).

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell so `uv` is on PATH, then verify:

```powershell
uv --version
```

(macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

---

## 2. Install Python 3.11 (isolated)

```powershell
uv python install 3.11
```

This does not touch your system Python 3.12. uv keeps 3.11 in its own managed
location and uses it only when asked.

---

## 3. Create and activate the venv

From the repo root (`...\self-learning\ml-learning`):

```powershell
cd "C:\Users\Venu\OneDrive\Documents\self-learning\ml-learning"
uv venv --python 3.11
```

The `--python 3.11` is important -- it forces the venv to 3.11 so the env check
passes (this is exactly what failed before).

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

**If you get a "running scripts is disabled" error**, Windows blocks script
activation by default. Allow it for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then re-run the Activate line. Your prompt should now show `(ml-learning)` or
`(.venv)` at the front.

(macOS/Linux: `source .venv/bin/activate`)

---

## 4. Install PyTorch (pick CPU or CUDA)

First, do you have an NVIDIA GPU? Check:

```powershell
nvidia-smi
```

- **Command works and shows a GPU** -> install the CUDA build.
- **Command not found / no GPU** -> install the CPU build. This is fine for the
  early weeks (attention, tokenization, small fine-tunes run on CPU, just slower).

CPU build:

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CUDA build (example for CUDA 12.4 -- match your driver; see
https://pytorch.org/get-started/locally/ for the exact selector):

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

> Heavy GPU training (QLoRA in P1, vLLM serving) realistically needs a cloud GPU
> later. A local CPU setup is enough to learn and run every week's core code; we
> rent GPU only when a project demands it.

---

## 5. Install the base stack and verify

```powershell
uv pip install transformers datasets numpy
python week-00-setup\src\verify_env.py
```

Expected: Python shows `[OK] 3.11.x`, the packages show `[OK]`, and the
accelerator line reports CUDA or "CPU-only". Exit code 0 means you're good.

If Python still shows 3.12, the venv wasn't activated or wasn't created with
`--python 3.11`. Re-do Step 3.

---

## 6. Configure git (once)

```powershell
git config --global user.name "Madhavan"
git config --global user.email "you@example.com"
git config --global core.autocrlf true
```

`core.autocrlf true` normalizes Windows CRLF line endings on commit so the repo
stays clean and cross-platform.

---

## 7. OneDrive caveat (read this)

This repo lives under OneDrive, so OneDrive will sync it. `.gitignore` keeps
model weights and checkpoints out of *git*, but it does NOT stop OneDrive from
uploading multi-GB files. Before P1/P2 (which download large models), either:

- right-click the repo folder -> "Free up space" / exclude it from sync, or
- move the repo outside OneDrive (e.g. `C:\code\ml-learning`).

For Week 00-05 (small models only), OneDrive sync is harmless.

---

## 7b. Low-disk / CPU-only setup (~13 GB free) -- the path actually used

This repo was set up on a machine with ~13 GB free and no GPU. If that's you,
the standard steps above still apply with three deliberate changes. This is the
configuration that is known-good here.

**1. Put the venv OUTSIDE OneDrive.** `.gitignore` keeps `.venv` out of git, but
OneDrive ignores that and would sync thousands of env files. Create it on a
non-synced path instead of in the repo:

```powershell
New-Item -ItemType Directory -Force -Path "C:\Users\Venu\Downloads\venvs"
uv venv "C:\Users\Venu\Downloads\venvs\ml-learning" --python 3.11
& "C:\Users\Venu\Downloads\venvs\ml-learning\Scripts\Activate.ps1"
```

Because the venv is outside the repo, activate it by full path each session.
Optional shortcut -- add to your PowerShell profile (`notepad $PROFILE`):

```powershell
function actml { & "C:\Users\Venu\Downloads\venvs\ml-learning\Scripts\Activate.ps1" }
```

**2. Pin the HuggingFace cache off OneDrive too**, so model downloads are
watchable and prunable:

```powershell
New-Item -ItemType Directory -Force -Path "C:\Users\Venu\Downloads\hf-cache"
setx HF_HOME "C:\Users\Venu\Downloads\hf-cache"
$env:HF_HOME = "C:\Users\Venu\Downloads\hf-cache"   # setx only affects NEW shells; this sets the current one
# inspect later with:  huggingface-cli scan-cache
```

**3. Install the CPU PyTorch build** (Step 4 CPU line). The CUDA build is
~2.5-3.5 GB; CPU is ~0.4 GB and runs every early-week exercise, just slower.

Rough footprint with this config: env ~1-1.5 GB, plus ~250-900 MB per cached
model in Weeks 01-03. Comfortable inside 13 GB through ~Week 10.

> Heavy GPU work (P1 QLoRA on Llama-3.1-8B, P2 teacher model, vLLM serving) was
> always scoped for cloud GPU and never touches local disk. Your laptop runs the
> learning and small-model code; projects rent GPU when needed.

**Note on the uv hardlink warning:** if you see "Failed to hardlink files;
falling back to full copy," it's because the cache and target are on different
filesystems. Harmless (speed only). Silence it with `setx UV_LINK_MODE copy`.

---

## Weekly workflow cheat-sheet

Start of a session:

```powershell
cd "C:\Users\Venu\OneDrive\Documents\self-learning\ml-learning"
.venv\Scripts\Activate.ps1
```

Scaffold a new week (run from Git Bash, or via `bash scripts/new_week.sh ...`):

```bash
./scripts/new_week.sh 01 transformer-foundations
```

Install that week's deps and run:

```powershell
uv pip install -r week-01-transformer-foundations\requirements.txt
python week-01-transformer-foundations\src\<entry>.py
```

Commit and push at the end of the week:

```powershell
git add -A
git commit -m "feat: week-01 transformer foundations"
git push
```

---

## Troubleshooting quick table

| Symptom | Fix |
|---------|-----|
| `verify_env.py` says Python 3.12 | venv not activated, or made without `--python 3.11`; redo Step 3 |
| `Activate.ps1 ... disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate |
| `uv: command not found` | reopen PowerShell after install; check PATH |
| `nvidia-smi` not recognized | no NVIDIA GPU/driver -> use the CPU torch build |
| torch install pulls wrong CUDA | match `--index-url` to your driver via pytorch.org selector |
| `new_week.sh` won't run in PowerShell | run it from Git Bash, or `bash scripts/new_week.sh ...` |
