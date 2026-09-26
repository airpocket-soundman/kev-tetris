"""Who picks the move: Kev over its HTTP API, or (for demos and tests without a GPU) a hand-written heuristic.

KevServer starts and stops `python -m kev.serve` for one checkpoint, so a caller can swap generations: the GPU holds one
model at a time.
"""
from __future__ import annotations

import json, math, os, random, subprocess, sys, time, urllib.error, urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import kevenv, proc
from .interface import read_answer, to_request
from .tetris import Game, Placement

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Decision:
    placement: Placement
    probs: dict[str, float]     # probability per placement key (what the model "thinks")
    latency_ms: float


class KevPolicy:
    """Asks a running Kev server. temperature 0 = the argmax; > 0 samples from p^(1/T) (exploration for RL)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8009", temperature: float = 0.0, seed: int | None = None, timeout: float = 300,
                 allow=None, explore: float = 1.0, top_k: int = 0):
        """allow(game, placements) -> the placements sampling may pick from (None = all); only used when temperature > 0.
        The first request after a server start compiles kernels for a while, hence the long timeout."""
        # explore: share of moves that sample (the rest play the best allowed move); top_k: sample among the k likeliest
        self.base_url, self.temperature, self.timeout, self.allow = base_url.rstrip("/"), temperature, timeout, allow
        self.explore, self.top_k = explore, top_k
        self.rng = random.Random(seed)

    def decide(self, game: Game) -> Decision:
        placements = game.placements()
        t0 = time.perf_counter()
        body = json.dumps(to_request(game, placements)).encode()
        req = urllib.request.Request(f"{self.base_url}/v1/systemone", body, {"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.load(r)
        chosen, probs = read_answer(resp["answers"], placements)
        if self.temperature > 0:
            pool = (self.allow(game, placements) if self.allow else None) or placements
            pool = sorted(pool, key=lambda p: probs.get(p.key, 0.0), reverse=True)
            if self.rng.random() >= self.explore: pool = pool[:1]          # exploit: the likeliest allowed move
            elif self.top_k: pool = pool[:self.top_k]
            keys = [p.key for p in pool]
            w = [max(probs.get(k, 0.0), 1e-9) ** (1 / self.temperature) for k in keys]
            chosen = pool[self.rng.choices(range(len(keys)), weights=w)[0]]
        return Decision(chosen, probs, resp.get("latency_ms", (time.perf_counter() - t0) * 1000))


class HeuristicPolicy:
    """Not Kev: a fixed linear evaluator (Dellacherie-style weights). Lets the stream screen and the engine be tried
    without the model, and serves as a reference line on the progress chart."""

    W = {"lines": 0.76, "holes": -0.36, "agg_height": -0.51, "bumpiness": -0.18}

    def decide(self, game: Game) -> Decision:
        placements = game.placements()
        scores = []
        for p in placements:
            f = game.features(p)
            scores.append(sum(w * getattr(f, k) for k, w in self.W.items()))
        m = max(scores)
        ex = [math.exp((s - m) * 2) for s in scores]
        z = sum(ex)
        probs = {p.key: e / z for p, e in zip(placements, ex)}
        return Decision(placements[scores.index(m)], probs, 0.0)


class RandomPolicy:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def decide(self, game: Game) -> Decision:
        placements = game.placements()
        return Decision(self.rng.choice(placements), {p.key: 1 / len(placements) for p in placements}, 0.0)


class KevServer:
    """`python -m kev.serve --run <checkpoint>` as a child process, from the deployed Kev (kevenv). Use as a context manager."""

    def __init__(self, run: str, port: int = 8009, log: Path | None = None, env: dict | None = None):
        self.run, self.port, self.env = run, port, env or {}
        self.log = log or ROOT / "runs" / "serve.log"
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/v1/models", timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def start(self, timeout: float = 1800, tick=None) -> "KevServer":
        """Launch and wait until it answers. `tick()` is called about once a second while loading; if it raises, the
        server is stopped and the exception propagates (a stop request while the weights load)."""
        if self.ready(): raise RuntimeError(f"port {self.port} is already serving; stop that server first")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        f = self.log.open("a", encoding="utf-8")
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} serve {self.run} ===\n"); f.flush()
        self.proc = subprocess.Popen([kevenv.kev_python(), "-m", "kev.serve", "--run", kevenv.resolve_run(self.run), "--port", str(self.port)],
                                     stdout=f, stderr=subprocess.STDOUT, cwd=kevenv.kev_home(), env={**kevenv.kev_env(), **self.env})
        t0 = time.time()
        while not self.ready():
            if self.proc.poll() is not None: raise RuntimeError(f"kev.serve exited ({self.proc.returncode}); see {self.log}")
            if time.time() - t0 > timeout: self.stop(); raise TimeoutError(f"kev.serve not ready after {timeout}s")
            if tick:
                try: tick()
                except BaseException: self.stop(); raise
            time.sleep(1)
        return self

    def stop(self):
        if self.proc and self.proc.poll() is None:
            proc.kill(self.proc.pid)   # the whole tree: on Windows the venv launcher's child holds the GPU
            self.proc.wait()
        self.proc = None

    def __enter__(self): return self.start()
    def __exit__(self, *exc): self.stop()
