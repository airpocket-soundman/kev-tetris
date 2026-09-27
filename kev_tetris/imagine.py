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
    # steps towards the harder answers (covers above all): where the piece ends up, what is right under it
    "gap_under": {"type": "score", "instructions": "Once the piece has landed, how many empty cells are directly below "
                                                    "its cells (counting each cell right under a piece cell)?",
                  "criteria": ["none", "one", "two", "three", "four or more"]},
    "low_row": {"type": "score", "instructions": "Which row does the lowest cell of the piece end up in?",
                "criteria": ["rows 1-4", "rows 5-8", "rows 9-12", "rows 13-16", "row 17 or higher"]},
}
TARGETED = ("clears", "covers", "height", "ready")   # the answers a move choice needs (IMAGINE_TARGETS in rl.py)


def height_bin(h: int) -> int:
    return min(4, max(0, (h - 1) // 4)) if h > 0 else 0


def answers(game: Game, p) -> dict:
    """The true answers for a move on this board."""
    f = game.features(p)
    board, _ = apply_placement(game.board, p, 1)
    after = board_features(board)
    H, cells = len(game.board), set(p.cells)
    gap = min(4, sum(1 for x, y in p.cells if y + 1 < H and (x, y + 1) not in cells and not game.board[y + 1][x]))
    low = H - max(y for _, y in p.cells)                      # the row number of the piece's lowest cell (1 = floor)
    return {"clears": str(f.lines), "covers": (f.new_enclosed + f.new_overhang) > 0,
            "height": height_bin(after["max_height"]), "ready": bool(after["tetris_ready"]),
            "gap_under": gap, "low_row": min(4, (low - 1) // 4)}


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
    pool = ((tetris or clearing) if r < 0.3 else clearing if r < 0.45 else covering if r < 0.6 else clean)
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


def read(ans: dict) -> dict:
    """Kev's answers to the probe questions, in the form of answers()."""
    return {k: (v["choice"] if v["type"] == "choice" else v["noul"] >= 0.5 if v["type"] == "noul" else round(v["score"]))
            for k, v in ans.items()}


def ask(base_url: str, req: dict, timeout: float = 300) -> dict:
    r = urllib.request.Request(f"{base_url.rstrip('/')}/v1/systemone", json.dumps({**req, "model": "kev-latest"}).encode(),
                               {"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp: return json.load(resp)["answers"]


def mined_dataset(base_url: str, n: int, seed: int, workers: int = 1, pool: int = 3) -> tuple[list[dict], dict]:
    """Hard-example mining: pool x n candidate probes, Kev (the parent) answers them all; every probe it gets wrong on a
    targeted question is kept, then as many it gets right (so the easy answers are not forgotten), up to n.
    -> (labelled probes, the candidates' error rate per question)."""
    from concurrent.futures import ThreadPoolExecutor
    cands = positions(n * pool, seed, workers=workers)
    def grade(gp):
        g, p = gp
        return read(ask(base_url, probe(g, p, with_labels=False))), answers(g, p)
    with ThreadPoolExecutor(4) as ex: graded = list(ex.map(grade, cands))
    wrong, right, errs = [], [], Counter()
    for (g, p), (got, true) in zip(cands, graded):
        bad = [k for k in QUESTIONS if got.get(k) != true[k]]
        errs.update(bad)
        (wrong if any(k in TARGETED for k in bad) else right).append(probe(g, p))
    rng = random.Random(seed)
    rng.shuffle(right)
    out = wrong[:n] + right[:max(0, n - len(wrong[:n]))]
    rng.shuffle(out)
    return out, {k: round(errs[k] / len(cands), 3) for k in QUESTIONS}


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
        got = read(ans)
        for k in QUESTIONS:
            right[k] += got[k] == true[k]; truth[k].append(true[k])
    out = {}
    for k in QUESTIONS:
        base = Counter(truth[k]).most_common(1)[0][1] / n
        out[k] = {"acc": round(right[k] / n, 3), "baseline": round(base, 3)}
    return out


DRILL_SHARE = 0.4   # Tetris drills among the records without outcomes (0.2 until gen 59, when Tetrises were still missed)


def _own_moves(args):
    """Positions from the teacher's own games (its two-piece search, 10% random moves), each labelled with the
    teacher's move, as records without the computed outcomes (option level 3)."""
    n, seed = args
    from . import rl, teacher
    reward = lambda b, f, c, d: rl.shaped_reward(b, f, c, d)
    rng, out = random.Random(seed), []
    while len(out) < n:
        g = Game(seed=rng.randrange(1 << 30))
        if rng.random() < DRILL_SHARE:
            # a Tetris drill (full rows around one well), often with an I piece in hand: finding the Tetris from the
            # board alone - without "clears 4" in the option text - is what the weaning games still lack
            drill = rl.make_drill(rng)
            g.board = [[0] * len(drill[0]) for _ in range(len(g.board) - len(drill))] + [row[:] for row in drill]
            if rng.random() < 0.5: g.current = "I"
            ps = g.placements()
            key = next((p.key for p in ps if g.features(p).lines == 4), None) or teacher.search_label(g, ps, reward, rl.potential, 8)
            out.append(interface.to_record(g, ps, key, level=3))
            continue
        stops = set(range(rng.randrange(2), 300, 2))   # every other position of a teacher game (cheap: one search per move)
        last = max(stops)
        while not g.over and g.pieces <= last and len(out) < n:
            ps = g.placements()
            key = teacher.search_label(g, ps, reward, rl.potential, 8)
            if g.pieces in stops:
                out.append(interface.to_record(g, ps, key, level=3))
            move = rng.choice(ps) if rng.random() < 0.1 else next(p for p in ps if p.key == key)
            g.step(move)
    return out


def own_moves(n: int, seed: int, workers: int = 1) -> list[dict]:
    """n teacher-labelled move records without computed outcomes, made on the CPU (weaning data at scale)."""
    if workers <= 1: return _own_moves((n, seed))
    from concurrent.futures import ProcessPoolExecutor
    k = -(-n // workers)
    with ProcessPoolExecutor(workers) as ex:
        parts = ex.map(_own_moves, [(k, seed * 1000 + i) for i in range(workers)])
    return [x for part in parts for x in part][:n]
