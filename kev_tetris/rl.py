"""Reinforcement learning for Kev: on-policy policy improvement with advantage-filtered imitation.

One generation:
  1. collect   the current generation plays `episodes` games, sampling its move from its own probabilities at
               `temperature` (exploration). Every decision is logged with its request.
  2. credit    each decision gets a shaped reward (lines cleared, new holes, stack growth, game over) and a discounted
               return G_t. A linear value baseline V(s) fitted on board features gives the advantage A_t = G_t - V(s_t).
  3. improve   the decisions with the highest positive advantage become labelled Choice records ("in this state, this
               move turned out better than expected") and Kev is fine-tuned on them with kev.train --init_from
               <previous generation>. This is the binary-weight form of advantage-weighted regression (filtered
               behaviour cloning / expert iteration), which fits Kev because kev.train optimises log loss on labels.
  4. evaluate  greedy play on fixed seeds; the result goes into runs/generations.json.

Kev cannot generate or train in the same process as it serves, and the GPU holds one model at a time, so the loop
starts kev.serve for collection and evaluation and stops it before training.
"""
from __future__ import annotations

import argparse, copy, json, os, random, shutil, socket, statistics, subprocess, sys, threading, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import generations, kevenv, live, proc, replays, teacher
from . import interface
from .interface import VALUE_QUESTION, to_record, to_value_request
from .policy import GATE, KevPolicy, KevServer, is_front, set_front
from .search import KevSearchPolicy, teacher_rollout
from .tetris import RULES, Game, board_features

ROOT = Path(__file__).resolve().parent.parent
# reward "v2": a Tetris is worth twice the game's own ratio (1/3/5/8), and stack height is not penalised: building for
# a Tetris means stacking high on purpose. Records from different reward versions are never trained on together.
REWARD_VERSION = "v5"   # v5 (gen 37): holes and second wells first, then Tetrises only (docs/plan.md 5.8c).  v4: v3 + Cold Clear style line rewards, B2B, T-spins, Dellacherie/BCTS potential, elite games (docs/plan.md 5.7)
LINE_REWARD = {0: 0.0, 1: 1.0, 2: 3.0, 3: 5.0, 4: 16.0}
# v4 (Cold Clear style): while the stack is safe, singles and doubles are worth little - build for a Tetris instead
LINE_REWARD_SAFE = {0: 0.0, 1: 0.2, 2: 0.8, 3: 3.0, 4: 16.0}
SAFE_HEIGHT = 10
# v5: a clean board (no hole, no overhang, no second well) below SAFE_HEIGHT builds for Tetrises: smaller clears cost
TETRIS_READY = 4.0    # v5: a move that completes a Tetris setup (from gen 41)
MISSED_TETRIS = 8.0
IMAGINE_TARGETS = {"clears": 0.95, "covers": 0.9, "height": 0.8, "ready": 0.9}   # board imagination done
OWN_LEVEL = 3   # option text without any computed outcome: only the cells a move fills (interface.option_text)   # v5: an I piece placed elsewhere while a Tetris was ready
LINE_REWARD_CLEAN = {0: 0.0, 1: -3.0, 2: -3.0, 3: -1.5, 4: 16.0}
SURVIVE_HEIGHT = 10   # v4 from gen 19: above this the potential falls with the square of the excess height
CONTROL = ROOT / "runs" / "rl_control.json"   # written by the control page: {"command": "run" | "pause" | "stop"}
STATUS = ROOT / "runs" / "rl_status.json"     # written here, read by the control page and the stream screen


class Stopped(Exception):
    """A stop request: the loop ends at the next safe point. The generation in progress is not registered, so the
    next start redoes it."""


def _write_json(path: Path, data: dict, retries: int = 50):
    """Atomic replace. On Windows it fails while a reader has the file open (the pages poll it every second), so retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    for i in range(retries):
        try:
            os.replace(tmp, path); return
        except PermissionError:
            if i == retries - 1: raise
            time.sleep(0.02)


def set_command(command: str):
    _write_json(CONTROL, {"command": command, "at": time.time()})


def get_command() -> str:
    try: return json.loads(CONTROL.read_text(encoding="utf-8")).get("command", "run")
    except (OSError, ValueError): return "run"


class Control:
    """Pause / stop between safe points, and a status file with a heartbeat.

    checkpoint() is called after every move, while a server loads and while kev.train runs. On "pause" it blocks until
    "run" or "stop" (on_pause / on_resume let the caller suspend and resume a child process meanwhile); on "stop" it
    raises Stopped."""

    def __init__(self):
        self.status = {"state": "running", "phase": "", "gen": None, "detail": "", "progress": None}
        self.lock = threading.RLock()   # games run in parallel threads and all report here
        self.last = 0.0

    def report(self, force: bool = False, **kw):
        with self.lock: self._report(force, **kw)

    def _report(self, force: bool = False, **kw):
        self.status.update(kw)
        if force or time.time() - self.last > 1.0:
            self.last = time.time()
            # host: a loop in a container has a pid the Windows side cannot check, so readers fall back to `updated`
            try: _write_json(STATUS, {**self.status, "pid": os.getpid(), "host": socket.gethostname(), "updated": self.last})
            except PermissionError: pass   # display only: never stop training over it

    def checkpoint(self, on_pause=None, on_resume=None):
        cmd = get_command()
        if cmd == "stop": raise Stopped
        if cmd == "pause":
            if on_pause: on_pause()
            self.report(force=True, state="paused")
            while (cmd := get_command()) == "pause":
                time.sleep(0.5); self.report()
            self.report(force=True, state="running")
            if on_resume: on_resume()
            if cmd == "stop": raise Stopped
        self.report()


@dataclass
class Step:
    record: dict            # the request, labelled with the move actually played
    reward: float
    feats: list[float]      # board features before the move, for the value baseline
    ret: float = 0.0
    adv: float = 0.0
    phi_before: float = 0.0
    phi_after: float = 0.0
    tetris: bool = False
    new_enclosed: int = 0
    teacher_record: dict | None = None   # the same position labelled with the lookahead search's move
    agrees: bool = True                  # the search picked the move that was played
    value_request: dict | None = None    # "how good is this board" on this position, next piece unknown (RL)
    source: str | None = None            # where the label came from when not the teacher, e.g. "rollout"
    tetris_key: str | None = None        # an I piece could clear 4 rows here and did not: the Tetris move (a bad move)
    own_request: dict | None = None      # the same position without the computed outcomes (OWN_LEVEL), for weaning


@dataclass
class Episode:
    seed: int
    steps: list[Step] = field(default_factory=list)
    lines: int = 0
    tetrises: int = 0
    score: int = 0
    pieces: int = 0
    died: bool = False
    moves: list[dict] = field(default_factory=list)   # for replays: key, top probabilities, latency
    rules: int = 2


DANGER_HEIGHT = 16   # the top 4 rows: stacking into them is penalised, the rest of the height is free (Tetris setups)
DANGER_HEIGHT_V4 = 12  # v4 (from gen 15): gen 14 stacked to 18-20 around several deep wells and topped out


def danger_height() -> int:
    return DANGER_HEIGHT_V4 if REWARD_VERSION in ("v4", "v5") else DANGER_HEIGHT


def shaped_reward(before: dict, after: dict, cleared: int, died: bool, tspin: bool = False, b2b: int = 0) -> float:
    v5 = REWARD_VERSION == "v5"
    if v5:
        # low board: Tetrises only, unless the clear repairs something - fewer holes/overhangs, shallower buried holes,
        # or a second well filled (a shallow dip alone made gen 37's boards "unclean" and paid for plain singles)
        repairs = (after["enclosed"] + after["overhang"] < before["enclosed"] + before["overhang"]
                   or after["hole_depth"] < before["hole_depth"] or after["extra_wells"] < before["extra_wells"])
        r = (LINE_REWARD_CLEAN if before["max_height"] <= SAFE_HEIGHT and not repairs else LINE_REWARD)[cleared] + 0.05
    if REWARD_VERSION in ("v4", "v5"):
        if not v5: r = (LINE_REWARD_SAFE if before["max_height"] <= SAFE_HEIGHT else LINE_REWARD)[cleared] + 0.05
        if cleared and b2b >= 2: r += 8.0                  # back-to-back Tetris / T-spin clear
        if cleared and tspin: r += 4.0 * cleared           # T-spin single/double/triple
        # holes and overhangs cost every move they stay (gen 17: 0.15, gen 19: 0.35): repair them at once, then build
        r -= (0.6 if v5 else 0.35) * (after["enclosed"] + after["overhang"])
        # a Tetris set up (an I would clear 4 rows now) pays. Spoiling one is not penalised as such (with nowhere else to
        # go it can be right); an I that skips a ready Tetris is (MISSED_TETRIS), and a piece dropped into the well pays
        # through the holes, overhangs and small clears it makes
        if v5 and after["tetris_ready"] and not before["tetris_ready"]: r += TETRIS_READY
        if v5:   # a second well (or a 2-3 wide dip) costs every move it stays; filling it pays
            r -= 0.1 * after["extra_wells"]
            r += 0.3 * max(0, before["extra_wells"] - after["extra_wells"])
        # above half the board: survive first - lowering the stack pays (from gen 19)
        if before["max_height"] > SURVIVE_HEIGHT: r += 0.6 * max(0, before["max_height"] - after["max_height"])
    else:
        r = LINE_REWARD[cleared] + 0.05
    r -= (1.5 if v5 else 1.0) * max(0, after["enclosed"] - before["enclosed"])     # a hole no piece can reach any more
    r -= (1.2 if v5 else 0.8 if REWARD_VERSION == "v4" else 0.5) * max(0, after["overhang"] - before["overhang"])   # a slide can still fill it
    # resolved: filled by a slide, or uncovered because the rows above cleared. Less than the penalty, so creating a
    # hole and filling it again never pays
    r += (1.0 if v5 else 0.8) * max(0, (before["enclosed"] + before["overhang"]) - (after["enclosed"] + after["overhang"]))
    r -= 0.5 * max(0, after["max_height"] - danger_height())
    if died: r -= 10.0
    return r


def potential(f: dict) -> float:
    """How good a board is, for the window credit: few holes, rows ready for a Tetris, a well (capped at 4 deep)."""
    phi = -1.0 * f["enclosed"] - 0.5 * f["overhang"] + 0.3 * f["ready_rows"] + 0.2 * min(f["max_well"], 4) \
        - 0.5 * max(0, f["max_height"] - danger_height())
    if REWARD_VERSION in ("v4", "v5"):   # Dellacherie / BCTS terms: rugged and holey boards are worse than they look
        phi -= 0.1 * f["row_transitions"] + 0.1 * f["col_transitions"] + 0.2 * f["hole_depth"] + 0.5 * f["hole_rows"]
        phi -= 0.5 * f["overhang"]                                            # overhangs weigh -1.0 in all (gen 19)
        phi -= 0.08 * max(0, f["max_height"] - SURVIVE_HEIGHT) ** 2           # "survive first" above half the board
        # one well for the I piece; every other well is a liability that grows fast with its depth (gen 18: two-well towers)
        phi -= (0.3 if REWARD_VERSION == "v5" else 0.15) * f["extra_wells"]
    return phi


def value_features(f: dict) -> list[float]:
    return [1.0, f["enclosed"], f["overhang"], f["max_height"], f["agg_height"] / 10, f["bumpiness"], f["wells"],
            f["ready_rows"], min(f["max_well"], 4), f["row_transitions"] / 10, f["col_transitions"] / 10,
            f["hole_depth"], f["hole_rows"]]


def play_episode(policy, seed: int, max_pieces: int, on_step=None, on_move=None, board=None, search=None,
                 values: bool = False, own: bool = False) -> Episode:
    """on_step(game, decision) after every move; on_move(event) gets the move as the stream screen draws it.
    board: a starting board (a Tetris drill) instead of an empty one. A decision carrying `label` (Kev's own search,
    RL) is its own teacher. values: keep each position's value question for value labels."""
    game, ep = Game(seed=seed), Episode(seed)
    if board is not None:   # a 20-row drill under the hidden rows of a rules-3 board
        game.board = [[0] * len(board[0]) for _ in range(len(game.board) - len(board))] + [row[:] for row in board]
    while not game.over and game.pieces < max_pieces:
        placements = game.placements()
        d = policy.decide(game)
        rec = to_record(game, placements, d.placement.key, level=0 if own else None)
        own_req = to_record(game, placements, d.placement.key, level=OWN_LEVEL) if own else None
        t_rec, agrees, source = None, True, None
        vreq = {k: v for k, v in to_value_request(game).items() if k != "model"} if values else None
        own = getattr(d, "label", None)
        if search or own:   # the teacher's move for this position (expert iteration): same request, another label
            key = search(game, placements, d) if search else own
            if isinstance(key, tuple): key, source = key
            agrees = key == d.placement.key
            t_rec = rec if agrees else {**rec, "questions": {"move": {**rec["questions"]["move"], "label": key}}}
        before = board_features(game.board)
        pre, piece = [row[:] for row in game.board], game.current
        tetris_at = next((p.key for p in placements if game.features(p).lines == 4), None) if piece == "I" else None
        cleared = game.step(d.placement)
        missed = tetris_at if tetris_at and cleared != 4 else None
        if on_move: on_move(live.move_event(game, pre, placements, d, piece))
        after = board_features(game.board)
        # a Tetris left on the table is a bad move (v5, from gen 41): the I went elsewhere while 4 rows were ready
        r = shaped_reward(before, after, cleared, game.over, game.last_tspin, game.b2b) - (MISSED_TETRIS if missed else 0.0)
        ep.steps.append(Step(rec, r, value_features(before),
                             phi_before=potential(before), phi_after=potential(after), tetris=cleared == 4,
                             new_enclosed=after["enclosed"] - before["enclosed"] if not cleared else 0,
                             teacher_record=t_rec, agrees=agrees, value_request=vreq, source=source, tetris_key=missed,
                             own_request=own_req))
        ep.moves.append(replays.move_record(d))
        if on_step: on_step(game, d)
    ep.lines, ep.score, ep.pieces, ep.died, ep.tetrises, ep.rules = game.lines, game.score, game.pieces, game.over, game.tetrises, game.rules
    return ep


def make_drill(rng: random.Random):
    """A Tetris drill: 2-6 bottom rows full except one well column, under a ragged partial row; no holes."""
    from .tetris import HEIGHT, WIDTH
    b = [[0] * WIDTH for _ in range(HEIGHT)]
    well, rows = rng.randrange(WIDTH), rng.randint(2, 6)
    for y in range(HEIGHT - rows, HEIGHT):
        for x in range(WIDTH):
            if x != well: b[y][x] = rng.randint(1, 7)
    top = HEIGHT - rows - 1
    for x in range(WIDTH):
        if x != well and rng.random() < 0.5: b[top][x] = rng.randint(1, 7)
    return b


def assign_advantages(episodes: list[Episode], gamma: float = 0.97, truncate_tail: int = 60) -> list[Step]:
    """Discounted returns, then A = G - V with V a least-squares fit on board features. The last `truncate_tail` steps of
    an episode cut off by max_pieces are dropped: their return is missing the future."""
    usable = []
    for ep in episodes:
        g = 0.0
        for s in reversed(ep.steps):
            g = s.reward + gamma * g
            s.ret = g
        usable += ep.steps if ep.died else ep.steps[:max(0, len(ep.steps) - truncate_tail)]
    if not usable: return []
    X = np.array([s.feats for s in usable]); y = np.array([s.ret for s in usable])
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    for s, v in zip(usable, X @ w): s.adv = float(s.ret - v)
    return usable


def assign_window_advantages(episodes: list[Episode], window: int = 10, tetris_bonus: float = 1.5) -> list[Step]:
    """v3 credit: each move ends a window of the `window` moves up to it (sliding by one). A window's value is the
    rewards inside it plus the change in board potential over it; a least-squares baseline on the board at its start
    gives the window's advantage (x tetris_bonus when positive and it holds a Tetris). A move's advantage is the mean
    over the windows that contain it, so setup moves share the credit of the Tetris they lead to."""
    rows = []   # (episode steps, start index, end index, value)
    for ep in episodes:
        st = ep.steps
        for t in range(len(st)):
            s0 = max(0, t - window + 1)
            v = sum(x.reward for x in st[s0:t + 1]) + st[t].phi_after - st[s0].phi_before
            rows.append((st, s0, t, v))
    if not rows: return []
    X = np.array([r[0][r[1]].feats for r in rows]); y = np.array([r[3] for r in rows])
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    sums: dict[int, list] = {}
    for (st, s0, t, v), base in zip(rows, X @ w):
        adv = float(v - base)
        if adv > 0 and any(x.tetris for x in st[s0:t + 1]): adv *= tetris_bonus
        for x in st[s0:t + 1]:
            acc = sums.setdefault(id(x), [x, 0.0, 0]); acc[1] += adv; acc[2] += 1
    out = []
    for x, total, n in sums.values():
        x.adv = total / n; out.append(x)
    return out


def value_records(episodes: list[Episode], gamma: float, horizon: int = 10) -> tuple[list[dict], list[float]]:
    """Value labels (RL): each position's n-step return - the rewards of the next `horizon` moves plus the hand potential
    of the board after them (a game's end: the rewards only) - binned into the 5 levels of the value question by the
    targets' quintiles. -> (labelled value requests, the mean target of each level: what a level is worth to the search).
    Full discounted returns (gens 25-30) were dominated by whether a Tetris happened to come much later: Kev's answers
    correlated only 0.43 with them."""
    rows = []
    for ep in episodes:
        st = ep.steps
        for t, s in enumerate(st):
            if not s.value_request: continue
            end = min(t + horizon, len(st))
            g = sum(gamma ** (k - t) * st[k].reward for k in range(t, end))
            if not (ep.died and end == len(st)): g += gamma ** (end - t) * st[end - 1].phi_after
            rows.append((s.value_request, g))
    if not rows: return [], []
    rets = np.array([r for _, r in rows])
    edges = np.quantile(rets, [0.2, 0.4, 0.6, 0.8])
    lv = np.searchsorted(edges, rets, side="right")
    means = [float(rets[lv == i].mean()) if (lv == i).any() else float(np.quantile(rets, 0.1 + 0.2 * i)) for i in range(5)]
    recs = [{**q, "questions": {"value": {**q["questions"]["value"], "label": int(l)}}} for (q, _), l in zip(rows, lv)]
    return recs, means


def level_value(means: list[float]):
    """Kev's answer to the value question (probabilities by level) -> an expected return."""
    return lambda probs: sum(float(probs.get(str(i), 0.0)) * m for i, m in enumerate(means))


MAX_RECORD_CHARS = 5200   # ~2.2 chars per token: keeps a record inside kev.train's context at --max_state 2048


def select_records(steps: list[Step], keep_frac: float) -> list[dict]:
    # a move that sealed a hole is never imitated, whatever the window around it earned (v3)
    # records too long for the training context are left out (kev.train stops on them instead of skipping)
    pos = sorted((s for s in steps if s.adv > 0 and s.new_enclosed <= 0 and len(json.dumps(s.record)) <= MAX_RECORD_CHARS),
                 key=lambda s: s.adv, reverse=True)
    return [s.record for s in pos[:max(1, int(len(steps) * keep_frac))]]


def run_games(n_parallel: int, jobs: list) -> list:
    """Run play_episode jobs (zero-argument callables) n at a time, in order. While Kev computes on the GPU for one game,
    the others do their CPU work (search, features, requests); with CUDA graphs kev.serve also batches their requests.
    A stop request reaches every game through Control.checkpoint, so the first exception is re-raised."""
    if n_parallel <= 1: return [job() for job in jobs]
    with ThreadPoolExecutor(n_parallel) as ex:
        return [f.result() for f in [ex.submit(job) for job in jobs]]


class Showcase:
    """Which of the parallel games the stream screen follows: the lowest-numbered one still playing."""
    def __init__(self):
        self.lock, self.running = threading.Lock(), set()

    def start(self, i):
        with self.lock: self.running.add(i)

    def end(self, i):
        with self.lock: self.running.discard(i)

    def shown(self, i) -> bool:
        with self.lock: return bool(self.running) and i == min(self.running)


def evaluate(policy, seeds: list[int], max_pieces: int, on_step=None, replay_gen: int | None = None, on_move=None,
             parallel: int = 1, level: int | None = None) -> dict:
    """Greedy games on fixed seeds. With replay_gen the games are saved for the stream screen (kev_tetris.replays).
    policy: a policy, or a zero-argument factory giving one per game (needed when games run in parallel)."""
    show = Showcase()
    make = policy if callable(policy) and not hasattr(policy, "decide") else (lambda: policy)

    def job(i, s):
        def run():
            show.start(i); set_front(show.shown(i)); interface.set_level(level)
            def step(game, d):
                set_front(show.shown(i))
                if on_step: on_step(i, game)
            try:
                return play_episode(make(), s, max_pieces, step,
                                    (lambda ev: on_move(i, ev) if show.shown(i) else None) if on_move else None)
            finally:
                show.end(i)
        return run
    eps = run_games(parallel, [job(i, s) for i, s in enumerate(seeds)])
    if replay_gen is not None:
        replays.save(replay_gen, [{"seed": e.seed, "lines": e.lines, "score": e.score, "pieces": e.pieces, "died": e.died,
                                   "rules": e.rules, "moves": e.moves} for e in eps])
    return {"games": len(eps), "mean_lines": round(statistics.mean(e.lines for e in eps), 2),
            "mean_score": round(statistics.mean(e.score for e in eps), 1),
            "mean_pieces": round(statistics.mean(e.pieces for e in eps), 1), "best_lines": max(e.lines for e in eps),
            "best_score": max(e.score for e in eps), "max_pieces": max_pieces,
            # how well lines are cleared, whatever the length of the games: 100 = only singles, 200 = only Tetrises
            "score_per_line": round(sum(e.score for e in eps) / max(1, sum(e.lines for e in eps)), 1),
            # per piece: comparable across test caps (the survivors of a capped game count at the cap)
            "score_per_piece": round(sum(e.score for e in eps) / max(1, sum(e.pieces for e in eps)), 2),
            "lines_per_piece": round(sum(e.lines for e in eps) / max(1, sum(e.pieces for e in eps)), 4),
            "mean_tetrises": round(statistics.mean(e.tetrises for e in eps), 2)}


def train_generation(data: Path, init_from: str, out: Path, a, ctl: Control) -> None:
    """kev.train in the deployed Kev's venv. A pause suspends its whole process tree (its GPU memory stays allocated),
    a stop kills it."""
    if a.demo:   # stand-in: a process that just takes time, so pause / resume / stop can be tried without a GPU
        cmd, cwd = [sys.executable, "-c", f"import time\nfor i in range({a.demo_train_seconds}): print(i, flush=True); time.sleep(1)"], ROOT
    else:
        # on Windows, kev_train_win.py = `-m kev.train` with stand-ins for the Unix-only modules it imports (see that file)
        entry = [str(Path(__file__).with_name("kev_train_win.py"))] if sys.platform == "win32" else ["-m", "kev.train"]
        cmd = [kevenv.kev_python(), *entry, "--data", str(data), "--base", a.base,
               "--init_from", kevenv.resolve_run(init_from), "--out", str(out), "--epochs", str(a.epochs), "--lr", str(a.lr),
               "--batch", "1", "--accum", str(a.accum), "--dtype", "bf16", "--weights_dtype", "bf16", "--checkpointing", "1",
               "--device", "cuda", "--max_state", str(a.max_state)]
        if a.replay: cmd += ["--suite", a.replay_suite, "--replay", str(a.replay)]
        cwd = kevenv.kev_home()
    log = out.parent / f"{out.name}.train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[train] {' '.join(cmd)}\n        log: {log}", flush=True)
    t0, paused_for = time.time(), 0.0
    with log.open("w", encoding="utf-8") as f:
        child = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=kevenv.kev_env())
        try:
            while child.poll() is None:
                p0 = time.time()
                ctl.checkpoint(on_pause=lambda: proc.suspend(child.pid), on_resume=lambda: proc.resume(child.pid))
                paused_for += time.time() - p0
                sec = int(time.time() - t0 - paused_for)
                ctl.report(detail=f"学習中 {sec // 60}分{sec % 60:02d}秒")
                time.sleep(0.5)
        except BaseException:
            proc.kill(child.pid); child.wait()
            raise
    if child.returncode: raise RuntimeError(f"kev.train failed ({child.returncode}); see {log}")


def gpu_free_gb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split(",")
        return (float(out[1]) - float(out[0])) / 1024
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def wait_for_gpu(ctl: Control, need_gb: float, what: str):
    """Kev-4B needs most of a 16 GB card: before loading a server or starting kev.train, wait until enough memory is free
    (the stream screen gives its own servers up while training runs; a manually started Kev must be stopped by hand)."""
    if need_gb <= 0: return
    before = dict(ctl.status)
    while (free := gpu_free_gb()) is not None and free < need_gb:
        ctl.report(detail=f"GPUの空き待ち({what}): 空き {free:.1f} GB / 必要 {need_gb:.0f} GB。"
                          f"手動で起動したKev(8009番)など、GPUを使うプロセスを止めてください", waiting_gpu=True)
        ctl.checkpoint()
        time.sleep(3)
    ctl.report(force=True, detail=before.get("detail", ""), waiting_gpu=False)


class _Serving:
    """KevServer as a context manager that honours pause / stop while the weights load."""

    def __init__(self, run, port, ctl, need_gb=0.0, env=None):
        self.srv, self.ctl, self.need_gb = KevServer(run, port, env=env), ctl, need_gb

    def __enter__(self):
        wait_for_gpu(self.ctl, self.need_gb, "Kevの読み込み")
        return self.srv.start(tick=self.ctl.checkpoint)

    def __exit__(self, *exc):
        self.srv.stop()


def run_loop(a, ctl: Control):
    runs, data_dir = ROOT / "runs", ROOT / "runs" / ("demo-data" if a.demo else "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    eval_seeds = [10_000 + i for i in range(a.eval_games)]
    rng = random.Random(a.seed)
    branch_file = runs / "branch.json"   # remembers --branch_from, so a restart without it still branches correctly
    if a.branch_from is not None and not a.demo:
        branch_file.write_text(json.dumps({"reward": REWARD_VERSION, "from": a.branch_from}), encoding="utf-8")
    elif branch_file.exists():
        b = json.loads(branch_file.read_text(encoding="utf-8"))
        if b.get("reward") == REWARD_VERSION: a.branch_from = b["from"]
    demo_gens: list[dict] = []   # --demo keeps its generations in memory: runs/generations.json is left alone
    load = (lambda: demo_gens) if a.demo else generations.load

    def serving(run, other_model=False):
        """A Kev server for `run`. CUDA graphs only for the current model when --cuda_graphs (e.g. 0.8B), never for the
        other model a distillation reads from (4B on 16 GB is unstable with them)."""
        if a.demo: return nullcontext(None)
        env = {"KEV_CUDA_GRAPHS": "1"} if a.cuda_graphs and not other_model else {"KEV_CUDA_GRAPHS": "0"}
        return _Serving(run, a.port, ctl, a.need_teacher_serve_gb if other_model else a.need_serve_gb, env)

    model_of = lambda x: x.get("model", "Kev-4B")

    def policy(srv, gen, temperature=0.0):
        if a.demo:
            from .stream import NoisyHeuristic
            return NoisyHeuristic(max(0.0, 0.6 - 0.12 * gen), rng.randrange(1 << 30))
        # practice: sample only among moves that seal no hole while there are any (5% of moves unrestricted). A move
        # that clears lines is never masked (gen 37: an I into a deep well cleared 3 rows but left one hole, the mask
        # hid it and the stack topped out), and a stack above SAFE_HEIGHT is not masked at all
        high = lambda game: board_features(game.board)["max_height"] > SAFE_HEIGHT
        tetris_ready = lambda game: game.current == "I" and any(game.features(p).lines == 4 for p in game.placements())
        allow = (lambda game, ps: None if rng.random() < a.hole_free_eps or high(game) else
                 [p for p in ps if (f := game.features(p)).new_enclosed <= 0 or f.lines > 0]) if temperature > 0 else None
        if temperature > 0 and rl_means:   # RL practice: Kev's own lookahead, judged by Kev's own value answers
            return KevSearchPolicy(srv.url, lambda b, f, c, dd: shaped_reward(b, f, c, dd), level_value(rl_means),
                                   top_k=a.search_k, explore=a.explore, gamma=a.gamma, seed=rng.randrange(1 << 30), allow=allow,
                                   value_scale=(rl_means[-1] - rl_means[0]) / 4)
        # no exploration above SAFE_HEIGHT: a high stack is played for survival with Kev's best move
        return KevPolicy(srv.url, temperature=temperature, seed=rng.randrange(1 << 30), allow=allow,
                         explore=a.explore, top_k=a.top_k, explore_if=lambda game: not high(game) and not tetris_ready(game))

    feed = live.Feed()   # every move played here also goes to the stream screen

    def strength(x):
        """A tested generation's strength: score per piece (the test's piece cap changed: gens 37-38 ran to 1000 pieces,
        so their mean scores are about twice the others'). Older entries without it: mean score / pieces."""
        ev = x["eval"]
        return ev.get("score_per_piece") or ev["mean_score"] / max(1, ev.get("mean_pieces", 1))
    collapsed: set[int] = set()
    rng_ro = random.Random(a.seed + 7 if hasattr(a, "seed") and a.seed is not None else None)

    def imagined(x):
        """The board-imagination targets are met (docs/plan.md 5.9), or the imagination rounds stopped improving: the
        mean accuracy on the targeted questions rose less than a point over the last 3 rounds."""
        acc = x.get("imagine_acc")
        if not acc: return False
        if all(acc[k]["acc"] >= t for k, t in IMAGINE_TARGETS.items()): return True
        score = lambda e: sum(e["imagine_acc"][k]["acc"] for k in IMAGINE_TARGETS) / len(IMAGINE_TARGETS)
        chain = sorted((e for e in load() if e.get("kind") == "imagine" and e["gen"] <= x["gen"]), key=lambda e: e["gen"])
        return len(chain) >= 4 and score(chain[-1]) - score(chain[-4]) < 0.01

    def imagination_round(g, prev):
        """A generation that learns to picture the board after a move instead of playing: probes with computed answers
        (kev_tetris.imagine) plus the parent's move records (so it keeps playing), then the imagination test and the
        usual game test."""
        from . import imagine
        t0 = time.time()
        ctl.report(force=True, phase="train", gen=g, detail="盤面推論の問題を作成中", progress=None)
        print(f"[gen {g}] board imagination round from gen {prev['gen']}", flush=True)
        # hard-example mining with the parent: the probes it gets wrong first
        ctl.report(force=True, phase="train", detail="盤面推論: 苦手な問題を探しています", progress=None)
        with serving(prev["run"]) as srv:
            mined, errs = imagine.mined_dataset(srv.url, a.imagine_n, 1000 + g, workers=a.rollout_workers, pool=a.imagine_pool)
        print(f"[gen {g}] imagination mining: parent error rates {errs}", flush=True)
        probes = [json.dumps(r, ensure_ascii=False) for r in mined]
        # move records from the latest game generations (not the imagination rounds, whose files hold probes and reused
        # moves): gen 44 trained on 1500 probes + 600 moves and its tests fell from 500 to 221 pieces - as many moves as probes
        games = sorted((x["gen"] for x in load() if x.get("kind") != "imagine" and x.get("learner") == "rl"), reverse=True)[:3]
        moves = [l for k in games for src in [data_dir / f"gen-{k:03d}.jsonl"] if src.exists()
                 for l in src.read_text(encoding="utf-8").splitlines() if l and '"move"' in l]
        rng.shuffle(moves)
        lines = probes + moves[:a.imagine_moves]
        rng.shuffle(lines)
        data = data_dir / f"gen-{g:03d}.jsonl"
        data.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = runs / f"gen-{g:03d}"
        if out.exists(): shutil.rmtree(out)
        ctl.report(force=True, phase="train", detail=f"盤面推論 {len(probes)} 問 + 手 {len(lines) - len(probes)} 件で学習", progress=None)
        wait_for_gpu(ctl, a.need_train_gb, "学習")
        train_generation(data, prev["run"], out, argparse.Namespace(**{**vars(a), "epochs": a.imagine_epochs}), ctl)
        run = str(out.relative_to(ROOT)).replace("\\", "/")
        ctl.report(force=True, phase="test", detail="盤面推論テスト", progress=0.0)
        with serving(run) as srv:
            acc = imagine.accuracy(srv.url, a.imagine_eval)
            print(f"[gen {g}] imagination {acc}", flush=True)
            ev = evaluate(lambda: policy(srv, g), eval_seeds, a.eval_max_pieces, tester(g), g, test_feed(g),
                          parallel=a.test_parallel, level=0)
        generations.upsert({"gen": g, "run": run, "parent": prev["gen"], "model": a.model_name, "reward": REWARD_VERSION,
                            "rules": RULES, "teacher": False, "learner": "rl", "kind": "imagine", "imagine_acc": acc,
                            "option_level": 0,
                            "train": {"episodes": 0, "decisions": 0, "records": len(lines), "trained_on": len(lines),
                                      "mean_lines": 0, "minutes": round((time.time() - t0) / 60, 1)},
                            "eval": ev})
        ctl.report(force=True, last_eval={"gen": g, **ev})
        print(f"[gen {g}] eval {ev}", flush=True)

    def distill(g, gens):
        """Distillation into a new model (e.g. Kev-0.8B). The best generation so far (same rules) plays practice games,
        every position it meets gets the lookahead search's move as its label, the earlier teacher generations' records
        are added, and the new model's released checkpoint is fine-tuned on all of it. Then it is tested like any
        generation and registered under the new model's name."""
        src = max([x for x in gens if x.get("eval") and x.get("rules", 1) == RULES], key=strength)
        t0 = time.time()
        ctl.report(force=True, phase="collect", gen=g, detail=f"蒸留: 第{src['gen']}世代({model_of(src)})を読み込み中", progress=0.0)
        print(f"[gen {g}] distilling {model_of(src)} gen {src['gen']} into {a.model_name}", flush=True)
        search = lambda game, ps, d=None: teacher.search_label(game, ps, lambda b, f, c, dd: shaped_reward(b, f, c, dd),
                                                       potential, a.teacher_first_k)
        show, done = Showcase(), []
        with serving(src["run"], other_model=True) as srv:
            def job(i, seed, pol):
                def on_step(game, d):
                    ctl.checkpoint()
                    ctl.report(detail=f"蒸留用の対局 {len(done)}/{a.distill_games} ゲーム完了", progress=len(done) / a.distill_games)
                def on_move(ev):
                    if show.shown(i):
                        feed.publish({"gen": src["gen"], "phase": "practice", "game": i + 1, "games": a.distill_games,
                                      "max_pieces": a.max_pieces, "training_gen": g, "parallel": a.parallel}, ev)
                def run():
                    show.start(i)
                    try: return play_episode(pol, seed, a.max_pieces, on_step, on_move, search=search)
                    finally: show.end(i); done.append(i)
                return run
            eps = run_games(a.parallel, [job(i, rng.randrange(1 << 30), policy(srv, src["gen"], a.temperature))
                                         for i in range(a.distill_games)])
        fits = lambda r: r and len(json.dumps(r)) <= MAX_RECORD_CHARS
        recs = [s.teacher_record for e in eps for s in e.steps if fits(s.teacher_record)]
        rng.shuffle(recs)
        older = [p for x in gens if x.get("teacher") for p in [data_dir / f"gen-{x['gen']:03d}.jsonl"] if p.exists()]
        lines = [json.dumps(r, ensure_ascii=False) for r in recs] + [l for p in older for l in p.read_text(encoding="utf-8").splitlines() if l]
        lines = lines[:a.distill_cap]
        data = data_dir / f"gen-{g:03d}.jsonl"
        data.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[gen {g}] distillation set: {len(recs)} new + older teacher records -> {len(lines)}", flush=True)
        out = runs / f"gen-{g:03d}"
        if out.exists(): shutil.rmtree(out)
        ctl.report(force=True, phase="train", detail=f"{len(lines)} 手ぶんのデータで蒸留", progress=None)
        wait_for_gpu(ctl, a.need_train_gb, "学習")
        train_generation(data, a.start, out, a, ctl)
        run = str(out.relative_to(ROOT)).replace("\\", "/")
        ctl.report(force=True, phase="test", detail=f"第{g}世代 を読み込み中", progress=0.0)
        with serving(run) as srv:
            ev = evaluate(lambda: policy(srv, g), eval_seeds, a.eval_max_pieces, tester(g), g, test_feed(g), parallel=a.test_parallel)
        generations.upsert({"gen": g, "run": run, "parent": src["gen"], "model": a.model_name, "reward": REWARD_VERSION,
                            "rules": RULES, "teacher": False, "distilled_from": src["gen"],
                            "train": {"episodes": len(eps), "decisions": sum(len(e.steps) for e in eps), "records": len(lines),
                                      "trained_on": len(lines), "mean_lines": round(statistics.mean(e.lines for e in eps), 2),
                                      "minutes": round((time.time() - t0) / 60, 1)},
                            "eval": ev})
        ctl.report(force=True, last_eval={"gen": g, **ev})
        print(f"[gen {g}] distilled {a.model_name}: eval {ev}", flush=True)

    def tester(gen):
        pieces_now = {}
        def on_step(i, game):
            ctl.checkpoint()
            pieces_now[i] = game.pieces if not game.over else a.eval_max_pieces
            ctl.report(detail=f"テスト {len(pieces_now)}/{a.eval_games} ゲーム目・{game.lines}ライン",
                       progress=min(1.0, sum(min(1.0, p / a.eval_max_pieces) for p in pieces_now.values()) / a.eval_games))
        return on_step

    def test_feed(gen):
        return lambda i, ev: feed.publish({"gen": gen, "phase": "test", "game": i + 1, "games": a.eval_games,
                                           "max_pieces": a.eval_max_pieces, "training_gen": gen, "parallel": a.test_parallel}, ev)

    if not any(g["gen"] == 0 for g in load()):
        start = "demo:0" if a.demo else a.start
        ctl.report(force=True, phase="test", gen=0, detail=f"{start} を読み込み中", progress=0.0)
        print(f"[gen 0] evaluating {start}", flush=True)
        with serving(start) as srv:
            ev = evaluate(lambda: policy(srv, 0), eval_seeds, a.eval_max_pieces, tester(0), None if a.demo else 0, test_feed(0),
                          parallel=a.test_parallel)
        entry = {"gen": 0, "run": start, "parent": None, "model": a.model_name, "train": None, "eval": ev}
        demo_gens.append(entry) if a.demo else generations.upsert(entry)
        ctl.report(force=True, last_eval={"gen": 0, **ev})
        print(f"[gen 0] {ev}", flush=True)

    while True:
        gens = load()
        g = max(x["gen"] for x in gens) + 1
        # --branch_from starts a new line from an older generation; after that the latest one is the parent
        # parent: the best tested generation among the latest few of this reward version; the first one of a version
        # starts from the best generation so far (or --branch_from)
        if not a.demo and not any(model_of(x) == a.model_name for x in gens if x.get("eval")):
            distill(g, gens)    # the first generation of a new model: learned from the best one so far
            continue
        gens = [x for x in gens if model_of(x) == a.model_name]   # a model's lineage continues within itself
        mine = [x for x in gens if x.get("reward") == REWARD_VERSION and x.get("eval")]
        same_rules = [x for x in gens if x.get("eval") and x.get("rules", 1) == RULES]   # scores under other rules don't compare
        if a.learner == "rl" and not a.demo and any(x.get("learner") == "rl" for x in mine):
            mine = [x for x in mine if x.get("learner") == "rl"]   # RL continues from generations whose value answers were trained
        lv = getattr(a, "option_level", 0)
        if any(x.get("option_level", 0) == lv for x in mine):
            # a new option text level starts from the best generation so far, then continues within its own level (the
            # scores drop right after each step: the older levels would otherwise always win)
            mine = [x for x in mine if x.get("option_level", 0) == lv]
        if mine: pool = mine[-a.parent_window:]
        elif a.branch_from is not None: pool = [x for x in gens if x["gen"] == a.branch_from]
        else: pool = same_rules or [x for x in gens if x.get("eval")]
        prev = max(pool, key=strength)
        weaning = [x for x in gens if x.get("own_share") and x.get("eval_own")]
        if weaning and not a.demo:
            # while weaning, the lineage continues within the weaning generations, chosen by how well they play without
            # the computed outcomes (gen 51 went back to gen 46 by the full test and lost gen 50's weaning)
            own = lambda x: x["eval_own"].get("score_per_piece", 0)
            best = max(weaning[-a.parent_window:], key=lambda x: (own(x), strength(x)))
            # the latest one continues the lineage unless it is clearly worse: the own-judgement test is a few games, and
            # picking its maximum kept going back to one lucky generation (gens 56-58 all restarted from gen 55)
            prev = weaning[-1] if own(weaning[-1]) >= own(best) - a.own_tolerance else best
        latest = max((x for x in gens if x.get("eval")), key=lambda x: x["gen"], default=None)
        if a.imagine and latest and latest.get("kind") == "imagine" and not imagined(latest):
            prev = latest   # board-imagination rounds continue from each other until the targets are met
        if a.generations and g > a.generations: break   # 0 = until stopped
        if (a.imagine and latest and latest.get("kind") == "imagine" and imagined(latest)
                and not any(x.get("own_share") for x in gens) and strength(latest) >= 0.95 * strength(prev)):
            prev = latest   # the weaning starts from the imagination rounds' result, not an older, slightly stronger player
        if a.imagine and not a.demo and not imagined(prev) and not any(imagined(x) for x in gens):
            imagination_round(g, prev)
            continue
        rl = a.learner == "rl" and not a.demo
        # the parent's value levels, once an RL generation has trained them; before that (the first RL generation) the
        # lookahead search labels the moves and the value answers are only being learned
        # Kev's own search only after `value_warmup` generations of value labels (gen 26: after one, the search's
        # choices were noise and practice games lasted ~40 pieces), and never right after a collapsed attempt
        n_rl = sum(1 for x in gens if x.get("learner") == "rl")
        rl_means = prev.get("value_means") if rl and n_rl >= a.value_warmup and g not in collapsed else None
        # weaning off the computed outcomes (docs/plan.md 5.9): the share of games and records without them grows by
        # 0.25 once the parent's own-judgement test reached own_advance of its full test
        share = prev.get("own_share")
        if share is None: share = a.own_share_start
        elif (prev.get("eval_own") or {}).get("score_per_piece", 0) >= a.own_advance * prev["eval"]["score_per_piece"]:
            share = min(1.0, share + 0.25)
        if a.demo or not rl: share = 0.0
        if share: print(f"[gen {g}] own-judgement share {share:.2f}", flush=True)
        t0 = time.time()

        ctl.report(force=True, phase="collect", gen=g, detail=f"第{prev['gen']}世代 を読み込み中", progress=0.0)
        print(f"[gen {g}] collecting {a.episodes} episodes with gen {prev['gen']} (T={a.temperature})", flush=True)
        with serving(prev["run"]) as srv:
            show, pieces_now, done = Showcase(), {}, []
            hand = lambda game, ps: teacher.search_label(game, ps, lambda b, f, c, dd: shaped_reward(b, f, c, dd),
                                                         potential, a.teacher_first_k)
            ro = {"n": 0, "same": 0}
            pending, pending_lock = [], threading.Lock()

            def search(game, ps, d=None):
                # RL: a share of the positions is set aside for rollouts after the games (CPU, many cores: the stream's
                # Kev answers are not slowed); until then they carry the teacher's label
                key = hand(game, ps)
                if rl and a.rollout_rate and d is not None and rng_ro.random() < a.rollout_rate:
                    top = [p.key for p in sorted(ps, key=lambda p: d.probs.get(p.key, 0.0), reverse=True)[:a.rollout_k]]
                    with pending_lock:
                        token = f"ro{len(pending)}"
                        pending.append((copy.deepcopy(game), list(dict.fromkeys(top + [key])), rng_ro.randrange(1 << 30),
                                        a.rollout_depth, a.rollout_n, a.gamma, a.teacher_first_k))
                    return key, token
                return key
            if not ((a.teacher or rl) and not a.demo and not rl_means): search = None

            def practice_job(i, seed, drill, pol, own_game=False):
                def on_step(game, d):
                    ctl.checkpoint()
                    set_front(show.shown(i))
                    pieces_now[i] = game.pieces
                    ctl.report(detail=f"練習試合 {len(done)}/{a.episodes} ゲーム完了・{a.parallel}ゲーム同時",
                               progress=min(1.0, sum(min(1.0, p / a.max_pieces) for p in pieces_now.values()) / a.episodes))
                def on_move(ev):
                    if show.shown(i):
                        feed.publish({"gen": prev["gen"], "phase": "practice", "game": i + 1, "games": a.episodes,
                                      "max_pieces": a.max_pieces, "training_gen": g, "parallel": a.parallel}, ev)
                def run():
                    show.start(i); set_front(show.shown(i)); interface.set_level(OWN_LEVEL if own_game else 0)
                    try:
                        ep = play_episode(pol, seed, a.max_pieces, on_step, on_move, board=drill, search=search, values=rl,
                                          own=share > 0)
                    finally:
                        show.end(i)
                    done.append(i); pieces_now[i] = a.max_pieces
                    return ep
                return run
            jobs = []
            for i in range(a.episodes):   # seeds, drills and policies drawn here, in order: runs stay reproducible
                drill = make_drill(rng) if a.drill_frac and rng.random() < a.drill_frac else None
                own_game = share > 0 and rng.random() < share
                jobs.append(practice_job(i, rng.randrange(1 << 30), drill, policy(srv, prev["gen"], a.temperature), own_game))
            eps = run_games(a.parallel, jobs)
        if rl_means and statistics.mean(e.pieces for e in eps) < a.collapse_pieces:
            print(f"[gen {g}] Kev's search collapsed (mean {statistics.mean(e.pieces for e in eps):.0f} pieces): "
                  f"collecting again with the hand search", flush=True)
            collapsed.add(g); continue
        if rl and pending:
            ctl.report(force=True, phase="train", detail=f"ロールアウトで {len(pending)} 局面を検討中", progress=None)
            t_ro = time.time()
            with ProcessPoolExecutor(a.rollout_workers) as ex:
                best = list(ex.map(teacher_rollout, pending, chunksize=2))
            for e in eps:
                for s in e.steps:
                    if s.source and s.source.startswith("ro"):
                        key = best[int(s.source[2:])]; teacher_key = s.teacher_record["questions"]["move"]["label"]
                        ro["n"] += 1; ro["same"] += key == teacher_key
                        s.teacher_record = {**s.record, "questions": {"move": {**s.record["questions"]["move"], "label": key}}}
                        s.agrees, s.source = key == s.record["questions"]["move"]["label"], "rollout"
            print(f"[gen {g}] rollouts took {(time.time() - t_ro) / 60:.1f} min", flush=True)
        steps = assign_window_advantages(eps, a.window)
        means = None
        if rl:
            # RL: moves labelled by the search (Kev's own, or the hand one in the first RL generation), disagreements
            # first; plus value labels from the returns the games actually produced
            fits = lambda r: r and len(json.dumps(r)) <= MAX_RECORD_CHARS
            pool = [s for e in eps for s in e.steps if fits(s.teacher_record)]
            for s in pool:   # a missed Tetris: whatever the teacher or the rollouts said, the Tetris is the answer
                if s.tetris_key:
                    s.teacher_record = {**s.record, "questions": {"move": {**s.record["questions"]["move"], "label": s.tetris_key}}}
            missed = [s for s in pool if s.tetris_key]
            if missed: print(f"[gen {g}] missed Tetrises: {len(missed)} (trained x3)", flush=True)
            pool = [s for s in pool if not s.tetris_key]
            rolled = [s for s in pool if s.source == "rollout"]
            diff = [s for s in pool if not s.agrees and not s.source]
            same = [s for s in pool if s.agrees and not s.source]
            rng.shuffle(diff); rng.shuffle(same)
            if ro["n"]: print(f"[gen {g}] rollouts: {ro['n']} positions, same as the teacher {ro['same']}", flush=True)
            vrecs, means = value_records(eps, a.gamma)
            vrecs = [r for r in vrecs if fits(r)]; rng.shuffle(vrecs)
            # moves filled up to teacher_cap with agreements: with Kev's own search few moves differ (gen 29: 129), and
            # a set of mostly value records (279 moves vs 1200 values) let the move answers drift - 207 pieces in the test
            chosen = missed * 3 + (rolled + diff + same)[:max(a.teacher_cap, len(rolled))]
            moves = []
            for s in chosen:
                if s.own_request:
                    # weaning: the same position without the computed outcomes, with the same answer - what Kev picks
                    # with the outcomes spelled out it learns to pick from the board alone. The spelled-out copy stays
                    # for a share of the positions that shrinks as own_share grows
                    label = s.teacher_record["questions"]["move"]["label"]
                    moves.append({**s.own_request, "questions": {"move": {**s.own_request["questions"]["move"], "label": label}}})
                    if rng.random() >= share: moves.append(s.teacher_record)
                else:
                    moves.append(s.teacher_record)
            recs = moves + vrecs[:min(a.value_cap, len(moves))]
            if a.imagine_mix:   # board-imagination probes keep training the predictions Kev now has to make itself
                from . import imagine
                recs += imagine.dataset(a.imagine_mix, 5000 + g, workers=a.rollout_workers)
            if share and a.own_extra:   # weaning at scale: teacher-labelled positions without computed outcomes (CPU)
                from . import imagine
                ctl.report(force=True, phase="train", detail=f"予測なしの学習データを {a.own_extra} 件作成中", progress=None)
                recs += imagine.own_moves(a.own_extra, 7000 + g, workers=a.rollout_workers)
            rng.shuffle(recs)
            print(f"[gen {g}] rl: {len(diff)} disagreements, {len(same)} agreements, {len(vrecs)} value positions, "
                  f"level means {[round(m, 1) for m in means]}", flush=True)
        elif a.teacher and not a.demo:
            # expert iteration: the search's moves on the positions Kev reached, disagreements first (they teach most)
            fits = lambda s: s.teacher_record and len(json.dumps(s.teacher_record)) <= MAX_RECORD_CHARS
            pool = [s for s in steps if fits(s)]
            diff = [s.teacher_record for s in pool if not s.agrees]; same = [s.teacher_record for s in pool if s.agrees]
            rng.shuffle(diff); rng.shuffle(same)
            recs = (diff + same[:max(0, a.teacher_cap // 4)])[:a.teacher_cap]
            print(f"[gen {g}] teacher: {len(diff)} disagreements, {len(same)} agreements", flush=True)
        else:
            recs = select_records(steps, a.keep_frac)
        data = data_dir / f"gen-{g:03d}.jsonl"
        data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")

        # elite games: the moves that went well in the practice games with the best score per piece are kept across
        # generations (capped) and trained on again (self-imitation)
        elite = data_dir / f"elite-{REWARD_VERSION}.jsonl"
        if a.elite_games and not a.demo and not a.teacher:   # the teacher's labels replace the elite set
            best = sorted((e for e in eps if e.pieces >= 50), key=lambda e: e.score / e.pieces, reverse=True)[:a.elite_games]
            keep = {id(s) for e in best for s in e.steps}
            new = [json.dumps(s.record, ensure_ascii=False) for s in steps
                   if id(s) in keep and s.adv > 0 and s.new_enclosed <= 0 and len(json.dumps(s.record)) <= MAX_RECORD_CHARS]
            old = elite.read_text(encoding="utf-8").splitlines() if elite.exists() else []
            elite.write_text("\n".join((old + new)[-a.elite_cap:]) + "\n" if old or new else "", encoding="utf-8")
        merged = data_dir / f"train-{g:03d}.jsonl"
        # replay buffer: records of the latest generations made the same way (same reward version, teacher or not)
        same = sorted(x["gen"] for x in gens if x.get("reward", "v1") == REWARD_VERSION
                      and bool(x.get("teacher")) == bool(a.teacher) and x.get("learner") == (a.learner if rl else None)
                      )[-a.buffer_gens:] if a.buffer_gens else []
        parts = [data_dir / f"gen-{k:03d}.jsonl" for k in same + [g]]   # only records chosen under this reward
        if a.elite_games and not a.teacher and elite.exists(): parts.append(elite)
        merged.write_text("".join(p.read_text(encoding="utf-8") for p in parts if p.exists()), encoding="utf-8")
        n_merged = sum(1 for _ in merged.open(encoding="utf-8"))
        print(f"[gen {g}] {len(recs)} records from {sum(len(e.steps) for e in eps)} decisions; training on {n_merged}", flush=True)
        if not n_merged:   # every game hit max_pieces and was cut before its tail: nothing to learn from, play again
            ctl.report(force=True, detail="学習データが0件のため、練習試合をやり直します(--max_pieces を増やすと解消します)")
            continue

        out = runs / (f"demo-gen-{g:03d}" if a.demo else f"gen-{g:03d}")
        if out.exists():   # left by an attempt that was stopped or failed (gen g is not registered): kev.train refuses to overwrite
            shutil.rmtree(out)
        ctl.report(force=True, phase="train", detail=f"{n_merged} 手ぶんのデータで学習開始", progress=None)
        if not a.demo: wait_for_gpu(ctl, a.need_train_gb, "学習")
        train_generation(merged, prev["run"], out, a, ctl)

        run = f"demo:{g}" if a.demo else str(out.relative_to(ROOT)).replace("\\", "/")
        ctl.report(force=True, phase="test", detail=f"第{g}世代 を読み込み中", progress=0.0)
        ev_own = None
        with serving(run) as srv:
            # the test: with the computed outcomes until the weaning is done, then without; while weaning, a few more
            # games without them measure how far Kev judges the board on its own
            ev = evaluate(lambda: policy(srv, g), eval_seeds, a.eval_max_pieces, tester(g), None if a.demo else g, test_feed(g),
                          parallel=a.test_parallel, level=OWN_LEVEL if share >= 1 else 0)
            if 0 < share < 1:
                ev_own = evaluate(lambda: policy(srv, g), eval_seeds[:a.own_eval_games], a.eval_max_pieces, tester(g), None,
                                  test_feed(g), parallel=a.test_parallel, level=OWN_LEVEL)
                print(f"[gen {g}] own-judgement test {ev_own}", flush=True)
        entry = {"gen": g, "run": run, "parent": prev["gen"], "model": prev.get("model", a.model_name), "reward": REWARD_VERSION,
                 "rules": RULES, "teacher": bool(a.teacher) and not rl, "option_level": OWN_LEVEL if share >= 1 else getattr(a, "option_level", 0),
                 **({"own_share": share, "eval_own": ev if share >= 1 else ev_own} if share else {}),
                 **({"learner": "rl", "value_means": means, "search": "kev" if rl_means else "hand"} if rl else {}),
                 "train": {"episodes": len(eps), "decisions": sum(len(e.steps) for e in eps), "records": len(recs),
                           "trained_on": n_merged, "mean_lines": round(statistics.mean(e.lines for e in eps), 2),
                           "minutes": round((time.time() - t0) / 60, 1)},
                 "eval": ev}
        demo_gens.append(entry) if a.demo else generations.upsert(entry)
        ctl.report(force=True, last_eval={"gen": g, **ev})
        print(f"[gen {g}] eval {ev}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def main(argv=None):
    import argparse
    # run as `python -m kev_tetris.rl` this module is __main__: register it as kev_tetris.rl too, so the rollout workers
    # (forked) use the code that is running instead of importing the file on disk again (gen 41: a newer file on disk
    # met the running tetris module and the workers failed)
    sys.modules.setdefault("kev_tetris.rl", sys.modules[__name__])
    ap = argparse.ArgumentParser(description="Train Kev at Tetris, one generation at a time")
    ap.add_argument("--start", default="jaredpalmer/kev-4b", help="generation 0: a released Kev checkpoint (hub id) or a run directory")
    ap.add_argument("--model_name", default="Kev-4B", help="shown on the stream screen")
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B-Base", help="the backbone of --start (kev.train --base)")
    ap.add_argument("--need_serve_gb", type=float, default=10.0, help="free GPU memory to wait for before loading Kev (0 = do not check; ~3 for Kev-0.8B)")
    ap.add_argument("--need_train_gb", type=float, default=13.0, help="free GPU memory to wait for before kev.train (0 = do not check; ~6 for Kev-0.8B)")
    ap.add_argument("--generations", type=int, default=10, help="train until this generation exists (0 = keep going until stopped)")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--max_pieces", type=int, default=400, help="per collection episode (long enough to also meet deaths)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--explore", type=float, default=0.15, help="practice: share of moves that explore; the others play Kev's best allowed move")
    ap.add_argument("--top_k", type=int, default=3, help="practice: an exploring move samples among Kev's k likeliest allowed moves")
    ap.add_argument("--hole_free_eps", type=float, default=0.05, help="practice: share of moves sampled without the no-new-hole restriction")
    ap.add_argument("--drill_frac", type=float, default=0.25, help="practice: share of games starting from a Tetris drill board")
    ap.add_argument("--window", type=int, default=10, help="moves per credit window (v3)")
    ap.add_argument("--parent_window", type=int, default=3, help="the parent is the best tested of this many latest generations")
    ap.add_argument("--cuda_graphs", type=int, choices=[0, 1], default=0, help="CUDA graphs for this model's Kev servers (0.8B: yes; 4B on 16 GB: no)")
    ap.add_argument("--need_teacher_serve_gb", type=float, default=10.0, help="free GPU memory before loading the model a distillation reads from")
    ap.add_argument("--distill_games", type=int, default=24, help="games the best generation plays to build a distillation set")
    ap.add_argument("--distill_cap", type=int, default=4000, help="records in a distillation set")
    ap.add_argument("--test_parallel", type=int, default=1, help="test games at the same time: 1 = one after another, so the stream shows each whole game")
    ap.add_argument("--parallel", type=int, default=4, help="games played at the same time (practice and tests); more is faster overall "
                    "but each game waits longer for Kev (8 games: ~2.3 s per move on 4B without CUDA graphs)")
    ap.add_argument("--teacher", type=int, choices=[0, 1], default=1, help="train on a two-piece lookahead search's moves (v4 part 2)")
    ap.add_argument("--learner", choices=["teacher", "rl"], default="teacher",
                    help="rl: Kev's own lookahead + value answers (expert iteration); set in runs/model.json")
    ap.add_argument("--search_k", type=int, default=4, help="RL: Kev's likeliest moves the lookahead judges")
    ap.add_argument("--value_warmup", type=int, default=3, help="RL: generations of value labels before Kev's own search")
    ap.add_argument("--collapse_pieces", type=float, default=150, help="RL: practice mean pieces below which Kev's search is dropped for the generation")
    ap.add_argument("--rollout_rate", type=float, default=0.0, help="RL: share of practice positions labelled by Kev's own rollouts")
    ap.add_argument("--imagine", type=int, choices=[0, 1], default=0, help="board-imagination rounds before the weaning (docs/plan.md 5.9)")
    ap.add_argument("--imagine_n", type=int, default=2500, help="imagination probes per round")
    ap.add_argument("--imagine_pool", type=int, default=2, help="candidate probes per kept one (hard-example mining)")
    ap.add_argument("--imagine_epochs", type=int, default=2, help="epochs over an imagination round's probes")
    ap.add_argument("--imagine_moves", type=int, default=0, help="move records kept in a round (0: imagination only)")
    ap.add_argument("--imagine_mix", type=int, default=0, help="imagination probes added to each game generation's records")
    ap.add_argument("--imagine_eval", type=int, default=300, help="probes in the imagination test")
    ap.add_argument("--own_share_start", type=float, default=0.0, help="weaning: first share of games/records without computed outcomes (0 = off)")
    ap.add_argument("--own_advance", type=float, default=0.9, help="weaning: advance when the own-judgement test reaches this share of the full test")
    ap.add_argument("--own_extra", type=int, default=2500, help="weaning: teacher-labelled records without computed outcomes made on the CPU per generation")
    ap.add_argument("--own_tolerance", type=float, default=3.0, help="weaning: the latest generation stays the parent within this many points of the best own-judgement score")
    ap.add_argument("--own_eval_games", type=int, default=5, help="weaning: test games without computed outcomes")
    ap.add_argument("--rollout_workers", type=int, default=12, help="RL: CPU processes for the rollouts")
    ap.add_argument("--rollout_k", type=int, default=3, help="RL: Kev's likeliest moves the rollouts compare")
    ap.add_argument("--rollout_depth", type=int, default=10, help="RL: pieces Kev plays on after each move")
    ap.add_argument("--rollout_n", type=int, default=2, help="RL: rollouts per move")
    ap.add_argument("--value_cap", type=int, default=600, help="RL: value records per generation")
    ap.add_argument("--teacher_cap", type=int, default=600, help="teacher records per generation (disagreements first)")
    ap.add_argument("--teacher_first_k", type=int, default=8, help="first-ply placements the search expands")
    ap.add_argument("--elite_games", type=int, default=2, help="practice games per generation whose good moves join the elite set")
    ap.add_argument("--elite_cap", type=int, default=600, help="records kept in the elite set")
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--branch_from", type=int, default=None, help="start the current reward version from this generation (used until one of its generations exists)")
    ap.add_argument("--keep_frac", type=float, default=0.35, help="fraction of decisions kept as training records (top positive advantage)")
    ap.add_argument("--buffer_gens", type=int, default=1, help="also train on the records of this many previous generations")
    ap.add_argument("--eval_games", type=int, default=10, help="test games (10 since v4: per-piece results are compared, less noise)")
    ap.add_argument("--eval_max_pieces", type=int, default=500, help="test games stop here, so a strong generation still finishes; generations are compared by mean score")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_state", type=int, default=2048, help="training context; v3 prompts (slides, more features) need more than v2's 1536")
    ap.add_argument("--replay", type=int, default=0, help="mix in N records of --replay_suite so general skill is not forgotten")
    ap.add_argument("--replay_suite", default="evals/v7/decision-v7")
    ap.add_argument("--port", type=int, default=8019, help="the training loop's own Kev server (8009 = the manual one, 8011/8012 = the stream screen)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="no GPU: heuristic players and a dummy training process, runs/generations.json untouched (to try the controls)")
    ap.add_argument("--demo_train_seconds", type=int, default=30)
    a = ap.parse_args(argv)
    # runs/model.json switches the model without touching the container's command, e.g.
    # {"start": "jaredpalmer/kev-0.8b", "model_name": "Kev-0.8B", "base": "Qwen/Qwen3.5-0.8B-Base",
    #  "need_serve_gb": 3, "need_train_gb": 6, "cuda_graphs": 1}
    model_file = ROOT / "runs" / "model.json"
    if model_file.exists() and not a.demo:
        for k, v in json.loads(model_file.read_text(encoding="utf-8")).items(): setattr(a, k, v)

    from . import interface
    interface.OPTION_LEVEL = getattr(a, "option_level", 0)   # how much of each move's outcome Kev is told (interface.py)
    print(f"option text level {interface.OPTION_LEVEL}", flush=True)

    set_command("run")
    ctl = Control()
    ctl.report(force=True, state="running", phase="start", gen=None, detail="起動中", progress=None,
               started=time.time(), demo=a.demo, target=a.generations)
    try:
        run_loop(a, ctl)
        ctl.report(force=True, state="finished", phase="done", detail=f"第{a.generations}世代まで完了", progress=1.0)   # only with a limit
    except Stopped:
        ctl.report(force=True, state="stopped", phase="stopped", detail="停止しました(途中の世代は次回やり直し)")
        print("[stop] stopped by request", flush=True)
    except Exception as e:
        ctl.report(force=True, state="error", phase="error", detail=f"エラー: {e}")
        raise


if __name__ == "__main__":
    main()
