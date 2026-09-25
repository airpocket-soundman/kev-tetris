"""runs/generations.json: one entry per trained generation, read by the stream screen.

    {"gen": 3, "run": "runs/gen-003", "parent": 2, "model": "Kev-0.8B", "created": "...",
     "train": {"episodes": 16, "decisions": 2710, "records": 950, "mean_lines": 11.2, "cumulative_records": 2800},
     "eval": {"games": 5, "mean_lines": 18.4, "mean_score": 2330, "mean_pieces": 71.0, "best_lines": 31}}
"""
from __future__ import annotations

import json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "runs" / "generations.json"


def load(path: Path = REGISTRY) -> list[dict]:
    if not path.exists(): return []
    return json.loads(path.read_text(encoding="utf-8"))


def save(gens: list[dict], path: Path = REGISTRY):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(gens, key=lambda g: g["gen"]), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def upsert(entry: dict, path: Path = REGISTRY):
    gens = [g for g in load(path) if g["gen"] != entry["gen"]]
    entry.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    save(gens + [entry], path)
