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

import json, os, random, shutil, socket, statistics, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import generations, kevenv, live, proc, replays, teacher
from .interface import to_record
from .policy import KevPolicy, KevServer
from .tetris import RULES, Game, board_features

ROOT = Path(__file__).resolve().parent.parent
# reward "v2": a Tetris is worth twice the game's own ratio (1/3/5/8), and stack height is not penalised: building for
# a Tetris means stacking high on purpose. Records from different reward versions are never trained on together.
REWARD_VERSION = "v4"   # v4: v3 + Cold Clear style line rewards, B2B, T-spins, Dellacherie/BCTS potential, elite games (docs/plan.md 5.7)
LINE_REWARD = {0: 0.0, 1: 1.0, 2: 3.0, 3: 5.0, 4: 16.0}
# v4 (Cold Clear style): while the stack is safe, singles and doubles are worth little - build for a Tetris instead
LINE_REWARD_SAFE = {0: 0.0, 1: 0.2, 2: 0.8, 3: 3.0, 4: 16.0}
SAFE_HEIGHT = 10
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
    return DANGER_HEIGHT_V4 if REWARD_VERSION == "v4" else DANGER_HEIGHT


def shaped_reward(before: dict, after: dict, cleared: int, died: bool, tspin: bool = False, b2b: int = 0) -> float:
    if REWARD_VERSION == "v4":
        r = (LINE_REWARD_SAFE if before["max_height"] <= SAFE_HEIGHT else LINE_REWARD)[cleared] + 0.05
        if cleared and b2b >= 2: r += 8.0                  # back-to-back Tetris / T-spin clear
        if cleared and tspin: r += 4.0 * cleared           # T-spin single/double/triple
        # holes and overhangs cost every move they stay (gen 17: 0.15, gen 19: 0.35): repair them at once, then build
        r -= 0.35 * (after["enclosed"] + after["overhang"])
        # above half the board: survive first - lowering the stack pays (from gen 19)
        if before["max_height"] > SURVIVE_HEIGHT: r += 0.6 * max(0, before["max_height"] - after["max_height"])
    else:
        r = LINE_REWARD[cleared] + 0.05
    r -= 1.0 * max(0, after["enclosed"] - before["enclosed"])     # a hole no piece can reach any more
    r -= (0.8 if REWARD_VERSION == "v4" else 0.5) * max(0, after["overhang"] - before["overhang"])   # a slide can still fill it
    # resolved: filled by a slide, or uncovered because the rows above cleared. Less than the penalty, so creating a
    # hole and filling it again never pays
    r += 0.8 * max(0, (before["enclosed"] + before["overhang"]) - (after["enclosed"] + after["overhang"]))
    r -= 0.5 * max(0, after["max_height"] - danger_height())
    if died: r -= 10.0
    return r


def potential(f: dict) -> float:
    """How good a board is, for the window credit: few holes, rows ready for a Tetris, a well (capped at 4 deep)."""
    phi = -1.0 * f["enclosed"] - 0.5 * f["overhang"] + 0.3 * f["ready_rows"] + 0.2 * min(f["max_well"], 4) \
        - 0.5 * max(0, f["max_height"] - danger_height())
    if REWARD_VERSION == "v4":   # Dellacherie / BCTS terms: rugged and holey boards are worse than they look
        phi -= 0.1 * f["row_transitions"] + 0.1 * f["col_transitions"] + 0.2 * f["hole_depth"] + 0.5 * f["hole_rows"]
        phi -= 0.5 * f["overhang"]                                            # overhangs weigh -1.0 in all (gen 19)
        phi -= 0.08 * max(0, f["max_height"] - SURVIVE_HEIGHT) ** 2           # "survive first" above half the board
        # one well for the I piece; every other well is a liability that grows fast with its depth (gen 18: two-well towers)
        phi -= 0.15 * f["extra_wells"]
    return phi


def value_features(f: dict) -> list[float]:
    return [1.0, f["enclosed"], f["overhang"], f["max_height"], f["agg_height"] / 10, f["bumpiness"], f["wells"],
            f["ready_rows"], min(f["max_well"], 4), f["row_transitions"] / 10, f["col_transitions"] / 10,
            f["hole_depth"], f["hole_rows"]]


def play_episode(policy, seed: int, max_pieces: int, on_step=None, on_move=None, board=None, search=None) -> Episode:
    """on_step(game, decision) after every move; on_move(event) gets the move as the stream screen draws it.
    board: a starting board (a Tetris drill) instead of an empty one."""
    game, ep = Game(seed=seed), Episode(seed)
    if board is not None:   # a 20-row drill under the hidden rows of a rules-3 board
        game.board = [[0] * len(board[0]) for _ in range(len(game.board) - len(board))] + [row[:] for row in board]
    while not game.over and game.pieces < max_pieces:
        placements = game.placements()
        d = policy.decide(game)
        rec = to_record(game, placements, d.placement.key)
        t_rec, agrees = None, True
        if search:   # the teacher's move for this position (expert iteration): same request, another label
            key = search(game, placements)
            agrees = key == d.placement.key
            t_rec = rec if agrees else {**rec, "questions": {"move": {**rec["questions"]["move"], "label": key}}}
        before = board_features(game.board)
        pre, piece = [row[:] for row in game.board], game.current
        cleared = game.step(d.placement)
        if on_move: on_move(live.move_event(game, pre, placements, d, piece))
        after = board_features(game.board)
        ep.steps.append(Step(rec, shaped_reward(before, after, cleared, game.over, game.last_tspin, game.b2b), value_features(before),
                             phi_before=potential(before), phi_after=potential(after), tetris=cleared == 4,
                             new_enclosed=after["enclosed"] - before["enclosed"] if not cleared else 0,
                             teacher_record=t_rec, agrees=agrees))
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
             parallel: int = 1) -> dict:
    """Greedy games on fixed seeds. With replay_gen the games are saved for the stream screen (kev_tetris.replays).
    policy: a policy, or a zero-argument factory giving one per game (needed when games run in parallel)."""
    show = Showcase()
    make = policy if callable(policy) and not hasattr(policy, "decide") else (lambda: policy)

    def job(i, s):
        def run():
            show.start(i)
            try:
                return play_episode(make(), s, max_pieces, (lambda game, d: on_step(i, game)) if on_step else None,
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
        # practice: sample only among moves that seal no hole while there are any (5% of moves unrestricted)
        allow = (lambda game, ps: None if rng.random() < a.hole_free_eps else
                 [p for p in ps if game.features(p).new_enclosed <= 0]) if temperature > 0 else None
        return KevPolicy(srv.url, temperature=temperature, seed=rng.randrange(1 << 30), allow=allow,
                         explore=a.explore, top_k=a.top_k)

    feed = live.Feed()   # every move played here also goes to the stream screen

    def distill(g, gens):
        """Distillation into a new model (e.g. Kev-0.8B). The best generation so far (same rules) plays practice games,
        every position it meets gets the lookahead search's move as its label, the earlier teacher generations' records
        are added, and the new model's released checkpoint is fine-tuned on all of it. Then it is tested like any
        generation and registered under the new model's name."""
        src = max([x for x in gens if x.get("eval") and x.get("rules", 1) == RULES], key=lambda x: x["eval"]["mean_score"])
        t0 = time.time()
        ctl.report(force=True, phase="collect", gen=g, detail=f"蒸留: 第{src['gen']}世代({model_of(src)})を読み込み中", progress=0.0)
        print(f"[gen {g}] distilling {model_of(src)} gen {src['gen']} into {a.model_name}", flush=True)
        search = lambda game, ps: teacher.search_label(game, ps, lambda b, f, c, dd: shaped_reward(b, f, c, dd),
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
        if mine: pool = mine[-a.parent_window:]
        elif a.branch_from is not None: pool = [x for x in gens if x["gen"] == a.branch_from]
        else: pool = same_rules or [x for x in gens if x.get("eval")]
        prev = max(pool, key=lambda x: x["eval"]["mean_score"])
        if a.generations and g > a.generations: break   # 0 = until stopped
        t0 = time.time()

        ctl.report(force=True, phase="collect", gen=g, detail=f"第{prev['gen']}世代 を読み込み中", progress=0.0)
        print(f"[gen {g}] collecting {a.episodes} episodes with gen {prev['gen']} (T={a.temperature})", flush=True)
        with serving(prev["run"]) as srv:
            show, pieces_now, done = Showcase(), {}, []
            search = (lambda game, ps: teacher.search_label(game, ps, lambda b, f, c, dd: shaped_reward(b, f, c, dd),
                                                            potential, a.teacher_first_k)) if a.teacher and not a.demo else None

            def practice_job(i, seed, drill, pol):
                def on_step(game, d):
                    ctl.checkpoint()
                    pieces_now[i] = game.pieces
                    ctl.report(detail=f"練習試合 {len(done)}/{a.episodes} ゲーム完了・{a.parallel}ゲーム同時",
                               progress=min(1.0, sum(min(1.0, p / a.max_pieces) for p in pieces_now.values()) / a.episodes))
                def on_move(ev):
                    if show.shown(i):
                        feed.publish({"gen": prev["gen"], "phase": "practice", "game": i + 1, "games": a.episodes,
                                      "max_pieces": a.max_pieces, "training_gen": g, "parallel": a.parallel}, ev)
                def run():
                    show.start(i)
                    try:
                        ep = play_episode(pol, seed, a.max_pieces, on_step, on_move, board=drill, search=search)
                    finally:
                        show.end(i)
                    done.append(i); pieces_now[i] = a.max_pieces
                    return ep
                return run
            jobs = []
            for i in range(a.episodes):   # seeds, drills and policies drawn here, in order: runs stay reproducible
                drill = make_drill(rng) if a.drill_frac and rng.random() < a.drill_frac else None
                jobs.append(practice_job(i, rng.randrange(1 << 30), drill, policy(srv, prev["gen"], a.temperature)))
            eps = run_games(a.parallel, jobs)
        steps = assign_window_advantages(eps, a.window)
        if a.teacher and not a.demo:
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
                      and bool(x.get("teacher")) == bool(a.teacher))[-a.buffer_gens:] if a.buffer_gens else []
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
        with serving(run) as srv:
            ev = evaluate(lambda: policy(srv, g), eval_seeds, a.eval_max_pieces, tester(g), None if a.demo else g, test_feed(g),
                          parallel=a.test_parallel)
        entry = {"gen": g, "run": run, "parent": prev["gen"], "model": prev.get("model", a.model_name), "reward": REWARD_VERSION,
                 "rules": RULES, "teacher": bool(a.teacher),
                 "train": {"episodes": len(eps), "decisions": sum(len(e.steps) for e in eps), "records": len(recs),
                           "trained_on": n_merged, "mean_lines": round(statistics.mean(e.lines for e in eps), 2),
                           "minutes": round((time.time() - t0) / 60, 1)},
                 "eval": ev}
        demo_gens.append(entry) if a.demo else generations.upsert(entry)
        ctl.report(force=True, last_eval={"gen": g, **ev})
        print(f"[gen {g}] eval {ev}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def main(argv=None):
    import argparse
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
