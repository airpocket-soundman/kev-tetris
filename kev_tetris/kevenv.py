"""Where the deployed Kev lives (see docs/kev-local-deployment.md).

Kev runs from its own checkout and venv (D:\\GitHub\\kev, Python 3.12, torch 2.8.0+cu128); kev-tetris only starts its
processes. Override with KEV_HOME (the checkout) and KEV_PYTHON (its interpreter).
"""
from __future__ import annotations

import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def kev_home() -> Path:
    if os.environ.get("KEV_HOME"): return Path(os.environ["KEV_HOME"])
    for cand in (ROOT.parent / "kev", ROOT / "kev"):
        if (cand / "kev" / "serve.py").exists(): return cand
    raise FileNotFoundError("Kev checkout not found: set KEV_HOME (see docs/kev-local-deployment.md)")


def kev_python() -> str:
    if os.environ.get("KEV_PYTHON"): return os.environ["KEV_PYTHON"]
    home = kev_home()
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        if (home / rel).exists(): return str(home / rel)
    return sys.executable


def kev_env() -> dict:
    # WinError 1314: the Hugging Face cache cannot create symlinks without developer mode, so it copies instead.
    # KEV_CUDA_GRAPHS=0 as in the deployment guide: Kev-4B on 16 GB is stabler without captured graphs (overridable).
    return {"KEV_CUDA_GRAPHS": "0", **os.environ, "PYTHONUNBUFFERED": "1",
            "HF_HUB_DISABLE_SYMLINKS": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}


def resolve_run(run: str) -> str:
    """A run as kev.serve / kev.train should see it from Kev's working directory: hub ids unchanged, local run
    directories (relative to kev-tetris) made absolute."""
    p = Path(run)
    if p.is_absolute(): return run
    if (ROOT / p).exists(): return str(ROOT / p)
    return run
