"""Board imagination: can Kev picture the board after a move? (docs/plan.md 5.9)

Kev is shown the board and one move - the piece and the cells it would fill - but none of the outcome, and is asked
what the board looks like afterwards: how many rows clear, whether an empty cell gets covered, how high the stack
gets, whether a Tetris is then ready. The answers are computed here, so the probes are both a test (accuracy against
always giving the most common answer) and training data: Kev learns to imagine the outcome before it is asked to play
without being told it.
"""
from __future__ import annotations

import json, random, urllib.request
from collections import Counter

from . import interface
from .policy import HeuristicPolicy
from .tetris import Game, apply_placement, board_features

HEIGHT_BINS = ["0-4 rows", "5-8 rows", "9-12 rows", "13-16 rows", "17 rows or more"]
QUESTIONS = {
    "clears": {"type": "choice", "instructions": "After this move, how many rows are completely filled and clear?",
               "criteria": {"0": "none", "1": "one row", "2": "two rows", "3": "three rows", "4": "four rows (a Tetris)"}},
    "covers": {"type": "noul", "instructions": "Does this move leave an empty cell covered under the piece or the stack "
                                                "(a new hole or overhang)?"},
    "height": {"type": "score", "instructions": "After this move (and any cleared rows), how high is the tallest column?",
               "criteria": HEIGHT_BINS},
    "ready": {"type": "noul", "instructions": "After this move, would an I piece dropped upright into one column clear "
                                               "four rows at once?"},
}


def height_bin(h: int) -> int:
    return min(4, max(0, (h - 1) // 4)) if h > 0 else 0


def answers(game: Game, p) -> dict:
    """The true answers for a move on this board."""
    f = game.features(p)
    board, _ = apply_placement(game.board, p, 1)
    after = board_features(board)
    return {"clears": str(f.lines), "covers": (f.new_enclosed + f.new_overhang) > 0,
            "height": height_bin(after["max_height"]), "ready": bool(after["tetris_ready"])}


def probe(game: Game, p, with_labels: bool = True) -> dict:
    """A request (or a labelled record) asking about one move; the options' outcomes are never given."""
    state = (interface.state_text(game, level=interface_level()) +
             f"\nMove: the {game.current} piece goes to cells {interface.cells_text(game, p)} (column:row).")
    qs = {k: dict(v) for k, v in QUESTIONS.items()}
    if with_labels:
        for k, v in answers(game, p).items(): qs[k]["label"] = v
    return {"state": state, "questions": qs}


def interface_level() -> int:
    return 3   # the board without any computed summary of it (no hole count)


def pick_move(game: Game, rng: random.Random):
    """A move to ask about, by quota so the answers are spread out: a clearing move (a Tetris when there is one) a
    third of the time, a covering move a sixth, otherwise a clean move (neither)."""
    feats = [(p, game.features(p)) for p in game.placements()]
    r = rng.random()
    clearing = [p for p, f in feats if f.lines > 0]
    tetris = [p for p, f in feats if f.lines == 4]
    covering = [p for p, f in feats if f.lines == 0 and f.new_enclosed + f.new_overhang > 0]
    clean = [p for p, f in feats if f.lines == 0 and f.new_enclosed + f.new_overhang == 0]
    pool = ((tetris or clearing) if r < 0.15 else clearing if r < 0.35 else covering if r < 0.5 else clean)
    return rng.choice(pool or [p for p, _ in feats])


def _positions(args):
    n, seed, every = args
    from .rl import make_drill
    rng, pol, out = random.Random(seed), HeuristicPolicy(), []
    while len(out) < n:
        g = Game(seed=rng.randrange(1 << 30))
        if rng.random() < 0.6:   # a Tetris drill: rows full around one well (ready Tetrises, clears)
            drill = make_drill(rng)
            g.board = [[0] * len(drill[0]) for _ in range(len(g.board) - len(drill))] + [row[:] for row in drill]
            if rng.random() < 0.35: g.current = "I"
            out.append((g, pick_move(g, rng)))
            continue
        stop = rng.randrange(0, 90)   # a position from a game of the hand evaluator with 10% random moves
        while not g.over and g.pieces < stop:
            g.step(rng.choice(g.placements()) if rng.random() < 0.1 else pol.decide(g).placement)
        if not g.over: out.append((g, pick_move(g, rng)))
    return out


def positions(n: int, seed: int, every: int = 3, workers: int = 1):
    """n (game, move) pairs: 60% Tetris drills, 40% positions from games of a hand-written evaluator."""
    if workers <= 1: return _positions((n, seed, every))
    from concurrent.futures import ProcessPoolExecutor
    k = -(-n // workers)
    with ProcessPoolExecutor(workers) as ex:
        parts = ex.map(_positions, [(k, seed * 1000 + i, every) for i in range(workers)])
    return [x for part in parts for x in part][:n]


def _copy(g: Game) -> Game:
    import copy
    return copy.deepcopy(g)


def dataset(n: int, seed: int, workers: int = 1) -> list[dict]:
    return [probe(g, p) for g, p in positions(n, seed, workers=workers)]


EVAL_SEED = 777


def accuracy(base_url: str, n: int = 300, timeout: float = 300) -> dict:
    """Kev's accuracy on a fixed set of probes, per question, next to always answering the most common label."""
    items = positions(n, EVAL_SEED)
    right, truth = Counter(), {k: [] for k in QUESTIONS}
    for g, p in items:
        req = {**probe(g, p, with_labels=False), "model": "kev-latest"}
        r = urllib.request.Request(f"{base_url.rstrip('/')}/v1/systemone", json.dumps(req).encode(),
                                   {"content-type": "application/json"})
        with urllib.request.urlopen(r, timeout=timeout) as resp: ans = json.load(resp)["answers"]
        true = answers(g, p)
        got = {"clears": ans["clears"]["choice"], "covers": ans["covers"]["noul"] >= 0.5,
               "height": round(ans["height"]["score"]), "ready": ans["ready"]["noul"] >= 0.5}
        for k in QUESTIONS:
            right[k] += got[k] == true[k]; truth[k].append(true[k])
    out = {}
    for k in QUESTIONS:
        base = Counter(truth[k]).most_common(1)[0][1] / n
        out[k] = {"acc": round(right[k] / n, 3), "baseline": round(base, 3)}
    return out
