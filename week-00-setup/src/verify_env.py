"""Environment sanity check for the ml-learning repo.

Verifies the Python version and reports the core ML stack + accelerator status.
Run after installing week-00-setup/requirements.txt:

    python week-00-setup/src/verify_env.py

Exits non-zero if the Python version is outside the supported range, so it can
double as a cheap CI guard later.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

MIN_PY = (3, 11)
MAX_PY_EXCLUSIVE = (3, 12)

# Packages we expect across early weeks. Absence is reported, not fatal —
# each week installs only what it needs.
CORE_PACKAGES = ["torch", "transformers", "datasets", "numpy"]


@dataclass
class PkgStatus:
    name: str
    version: str | None
    found: bool


def check_python() -> bool:
    v = sys.version_info
    ok = MIN_PY <= (v.major, v.minor) < MAX_PY_EXCLUSIVE
    flag = "OK" if ok else "FAIL"
    print(f"[{flag}] Python {v.major}.{v.minor}.{v.micro} "
          f"(require >={MIN_PY[0]}.{MIN_PY[1]}, <{MAX_PY_EXCLUSIVE[0]}.{MAX_PY_EXCLUSIVE[1]})")
    return ok


def check_package(name: str) -> PkgStatus:
    try:
        mod = importlib.import_module(name)
    except ImportError:
        return PkgStatus(name, None, False)
    return PkgStatus(name, getattr(mod, "__version__", "unknown"), True)


def report_accelerator() -> None:
    """Report GPU availability without making it a hard requirement."""
    try:
        import torch
    except ImportError:
        print("[--] torch not installed; skipping accelerator check")
        return

    if torch.cuda.is_available():
        print(f"[OK] CUDA available — {torch.cuda.device_count()} device(s), "
              f"first: {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("[OK] Apple MPS backend available")
    else:
        print("[--] No GPU detected — CPU-only (fine for early weeks)")


def main() -> int:
    print("ml-learning environment check\n" + "-" * 32)
    py_ok = check_python()

    print("\nPackages:")
    for name in CORE_PACKAGES:
        s = check_package(name)
        flag = "OK" if s.found else "--"
        version = s.version if s.found else "not installed"
        print(f"[{flag}] {s.name:<14} {version}")

    print()
    report_accelerator()

    print("\n" + "-" * 32)
    if not py_ok:
        print("Python version is unsupported. Fix before proceeding.")
        return 1
    print("Environment looks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
