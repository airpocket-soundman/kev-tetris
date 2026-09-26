"""Kev's own lookahead (the RL phase): Kev proposes moves, Kev judges the boards they lead to.

For a position, one request asks Kev for its move probabilities. The `top_k` likeliest moves are played on copies of
the board, and for each resulting board Kev answers the value question ("how good is this board for the rest of the
game", 0-4) with its next piece known and the one after it unknown - what a player would see. A move's worth is
its immediate reward + gamma x the value of the board after it; the best one is the search's choice. Kev is then
trained to pick that choice, and its value answers are trained on the returns the games actually produced, so both
improve from Kev's own play (expert iteration, AlphaZero-style, one ply deep).

Bootstrap: before Kev's value answers have been trained (the first RL generation), `fallback_value` stands in.
"""
from __future__ import annotations

import copy, json, math, random, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

from .interface import read_answer, to_request, to_value_request
from .policy import Decision
from .tetris import Game, apply_placement, board_features


class KevSearchPolicy:
    def __init__(self, base_url: str, reward, value_of_level, top_k: int = 4, explore: float = 0.15, gamma: float = 0.97,
                 fallback_value=None, seed: int | None = None, timeout: float = 300, allow=None, value_scale: float = 1.0):
        self.base_url, self.reward, self.value_of_level = base_url.rstrip("/"), reward, value_of_level
        self.top_k, self.explore, self.gamma, self.fallback_value = top_k, explore, gamma, fallback_value
        self.rng, self.timeout, self.allow = random.Random(seed), timeout, allow
        self.value_scale = max(1e-6, value_scale)   # return units per nat of Kev's move prior (PUCT-like anchor)

    def _ask(self, req: dict) -> dict:
        r = urllib.request.Request(f"{self.base_url}/v1/systemone", json.dumps(req).encode(), {"content-type": "application/json"})
        with urllib.request.urlopen(r, timeout=self.timeout) as resp:
            return json.load(resp)

    def decide(self, game: Game) -> Decision:
        t0 = time.perf_counter()
        placements = game.placements()
        resp = self._ask(to_request(game, placements, value=True))
        kev_choice, probs = read_answer(resp["answers"], placements)
        pool = (self.allow(game, placements) if self.allow else None) or placements
        top = sorted(pool, key=lambda p: probs.get(p.key, 0.0), reverse=True)[:self.top_k]
        before = board_features(game.board)
        cands = []
        for p in top:
            board, cleared = apply_placement(game.board, p, 1)
            after = copy.copy(game)                                  # a look, not a move: nothing is drawn
            after.board, after.current, after.next = board, game.next, None
            after.lines = game.lines + cleared
            f = board_features(board)
            died = not after.placements()                            # the next piece could not even appear
            cands.append((p, after, f, self.reward(before, f, cleared, died), died))
        if self.fallback_value:
            values = [0.0 if died else self.fallback_value(f) for _, _, f, _, died in cands]
        else:
            def value(c):
                _, after, _, _, died = c
                if died: return 0.0
                return self.value_of_level(self._ask(to_value_request(after))["answers"]["value"]["probabilities"])
            with ThreadPoolExecutor(len(cands)) as ex: values = list(ex.map(value, cands))
        # Kev's move prior anchors the search: a young value head cannot overturn a confident move on its own
        q = [math.log(max(probs.get(p.key, 0.0), 1e-6)) + (r + self.gamma * v) / self.value_scale
             for (p, _, _, r, _), v in zip(cands, values)]
        best = cands[max(range(len(q)), key=q.__getitem__)][0]
        chosen = best
        if self.rng.random() < self.explore and len(cands) > 1:    # explore among the searched moves
            chosen = self.rng.choice([c[0] for c in cands])
        d = Decision(chosen, probs, resp.get("latency_ms", (time.perf_counter() - t0) * 1000))
        d.label, d.kev_choice = best.key, kev_choice.key           # train towards the search's choice
        return d
