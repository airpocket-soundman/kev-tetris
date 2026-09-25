"""Recorded games: every generation's test games are saved with Kev's probabilities, so the stream screen can show a
generation playing without loading it on the GPU (Kev-4B is ~9 GB in bf16: on a 16 GB card it cannot be served next to
a training run).

runs/replays/gen-003.json = [{"seed": 10000, "lines": 12, "pieces": 80, "moves": [{"key": "r1x4", "probs": {...}, "latency_ms": 310.2}, ...]}, ...]

A replay is exact: the piece sequence comes from the seed and the engine is deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path

from .policy import Decision
from .tetris import Game

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "runs" / "replays"
TOP_PROBS = 5   # candidates kept per move (the screen shows 3)


def path(gen: int) -> Path:
    return DIR / f"gen-{gen:03d}.json"


def move_record(d: Decision) -> dict:
    top = sorted(d.probs.items(), key=lambda kv: kv[1], reverse=True)[:TOP_PROBS]
    return {"key": d.placement.key, "probs": {k: round(p, 4) for k, p in top}, "latency_ms": round(d.latency_ms, 1)}


def save(gen: int, games: list[dict]):
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = path(gen).with_suffix(".tmp")
    tmp.write_text(json.dumps(games, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path(gen))


def load(gen: int) -> list[dict]:
    try: return json.loads(path(gen).read_text(encoding="utf-8"))
    except (OSError, ValueError): return []


class ReplayPolicy:
    """Plays one recorded game back. Use with Game(seed=game["seed"])."""

    def __init__(self, game: dict):
        self.moves, self.i = game["moves"], 0

    def decide(self, game: Game) -> Decision:
        if self.i >= len(self.moves): raise StopIteration("the recording ends here")
        m = self.moves[self.i]; self.i += 1
        by_key = {p.key: p for p in game.placements()}
        return Decision(by_key[m["key"]], m["probs"], m.get("latency_ms", 0.0))
