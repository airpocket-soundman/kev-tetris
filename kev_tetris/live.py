"""Moves as the stream screen draws them, and the feed that carries the training loop's games to it.

The training loop already has Kev loaded while it plays its practice and test games, so the stream screen shows those
games instead of loading a model of its own (Kev-4B and training do not both fit next to a second copy on 16 GB).
runs/rl_live.json holds the latest move; the stream polls it.
"""
from __future__ import annotations

import json, threading, os, time
from pathlib import Path

from .tetris import PIECES, Game

ROOT = Path(__file__).resolve().parent.parent
FEED = ROOT / "runs" / "rl_live.json"
FRESH_SECONDS = 8.0   # a feed older than this means no game is being played right now (loading, kev.train)


def move_event(game: Game, pre: list, placements: list, d, piece: str) -> dict:
    """The "move" event for a move just played: `game` after the step, `pre` the board before it."""
    # the page draws the 20 visible rows: boards are cut and cells moved up by the hidden rows (rules 3), and cells in
    # the hidden rows are left out
    hidden = len(pre) - len(game.visible(pre))
    vis = lambda cells: [(x, y - hidden) for x, y in cells if y >= hidden]
    top = sorted(d.probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    by_key = {p.key: p for p in placements}
    thinking = [{"key": k, "p": round(p, 3), "cells": vis(by_key[k].cells) if k in by_key else [], "chosen": k == d.placement.key}
                for k, p in top]
    full = [y - hidden for y in range(hidden, len(pre)) if all(pre[y][x] or (x, y) in d.placement.cells for x in range(len(pre[0])))]
    return {**game.snapshot(), "board": game.visible(pre), "post": game.visible(), "cells": vis(d.placement.cells), "piece": piece,
            "color": PIECES.index(piece) + 1, "cleared": full, "thinking": thinking, "latency_ms": round(d.latency_ms, 1)}


class Feed:
    """Writer side (the training loop)."""

    def __init__(self, path: Path = FEED, keep: int = 30, test_interval: float = 0.4):
        # keep: the latest moves stay in the file, so a reader polling slower than the moves come loses none of them.
        # test_interval: a test move is published no sooner than this after the previous one (the test game waits):
        # the stream draws every test move instead of jumping several at once (the 0.8B answers in ~0.2 s)
        self.path, self.seq, self.recent, self.keep = path, 0, [], keep
        self.test_interval, self.last_test = test_interval, 0.0
        self.lock = threading.Lock()

    def publish(self, info: dict, move: dict):
        if info.get("phase") == "test" and self.test_interval:
            wait = self.last_test + self.test_interval - time.time()
            if wait > 0: time.sleep(wait)
            self.last_test = time.time()
        with self.lock:
            self.seq += 1
            self.recent = (self.recent + [{"seq": self.seq, "info": info, "move": move}])[-self.keep:]
            body = {"seq": self.seq, "pid": os.getpid(), "at": time.time(), "info": info, "move": move, "recent": self.recent}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        for _ in range(20):   # Windows: the reader may hold the file for a moment
            try: os.replace(tmp, self.path); return
            except PermissionError: time.sleep(0.01)
        try: tmp.unlink()
        except OSError: pass   # display only: a dropped move is fine


def read(path: Path = FEED) -> dict | None:
    """Reader side (the stream screen): the latest move, or None when there is none or it is stale."""
    try: data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError): return None
    return data if time.time() - data.get("at", 0) < FRESH_SECONDS else None
