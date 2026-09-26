"""How far the training loop is, in percent: within the current phase, the current generation and the whole run.

Worked out from files only (runs/rl_status.json, runs/gen-XXX.train.log, runs/generations.json), so a stream screen
started after the training loop still reports it.

  practice / test   exact: games and pieces played (the status file's progress).
  train             kev.train prints "ep0 step 10/20 ... 1.234s/rec" every 10 optimizer steps. With the seconds per
                    record, progress = time since its first record / (records x seconds per record). Before its first
                    print the rate of the previous generation stands in; with none, the phase is "measuring" (None).
  generation        the three phases weighted by how long they took last time (defaults until measured).
"""
from __future__ import annotations

import json, math, re, time
from pathlib import Path

from . import generations, replays

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "runs" / "progress_history.json"
PHASES = [("collect", "練習試合"), ("train", "学習"), ("test", "実力テスト")]
DEFAULT_WEIGHTS = {"collect": 0.3, "train": 0.5, "test": 0.2}
STEP = re.compile(r"step (\d+)/(\d+) .*?([\d.]+)s/rec")
RECORDS = re.compile(r"^(\d+) training requests", re.M)


def _load(path: Path) -> dict:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError): return {}


def train_log_progress(log: Path, spr_hint: float | None, accum: int = 8, started: float | None = None) -> tuple[float | None, float | None, str]:
    """-> (fraction or None, seconds per record seen in this log or None, remaining-time text).
    `started`: when the records line was first seen (the training loop starts right after it)."""
    try: text, mtime = log.read_text(encoding="utf-8", errors="replace"), log.stat().st_mtime
    except OSError: return 0.0, None, ""
    if "saved " in text: return 1.0, None, ""
    rec = RECORDS.search(text)
    if not rec: return 0.0, None, "モデル読み込み中"
    records = int(rec.group(1))
    steps = STEP.findall(text)
    if steps:
        step, total, spr = int(steps[-1][0]), int(steps[-1][1]), float(steps[-1][2])
        t0 = mtime - spr * min(records, step * accum)   # the line was written at the file's last change
        frac = (time.time() - t0) / max(1e-6, spr * records)
        return max(step / total, min(0.99, frac)), spr, _eta(spr * records - (time.time() - t0))
    if spr_hint:   # no rate yet in this log: estimate with the previous generation's, from when the records line appeared
        t0 = started if started is not None else mtime
        left = spr_hint * records - (time.time() - t0)
        return min(0.99, (time.time() - t0) / max(1e-6, spr_hint * records)), None, _eta(left) + "(見込み)"
    return None, None, "計測中"


def _eta(seconds: float) -> str:
    if seconds <= 0: return "まもなく完了"
    m, s = divmod(int(seconds), 60)
    return f"残り 約{m}分{s:02d}秒" if m else f"残り 約{s}秒"


class Tracker:
    """Remembers phase durations (runs/progress_history.json) to weight the phases of a generation."""

    def __init__(self):
        self.history = _load(HISTORY)
        self.current: tuple | None = None   # (pid, gen, phase, started at)
        self.records_seen: dict[str, float] = {}   # train log -> when its records line was first seen

    def _save(self):
        try:
            HISTORY.parent.mkdir(parents=True, exist_ok=True)
            HISTORY.write_text(json.dumps(self.history, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def _observe(self, st: dict):
        key = (st.get("pid"), st.get("gen"), st.get("phase"))
        if self.current and self.current[:3] == key: return
        if self.current and self.current[2] in dict(PHASES) and self.current[1] and math.isfinite(self.current[3]):
            # a phase we saw from its start has ended: keep its duration (paused time included, good enough)
            self.history.setdefault("seconds", {})[self.current[2]] = round(time.time() - self.current[3], 1)
            self._save()
        started_here = self.current is not None   # the first phase seen may have begun before this screen
        self.current = (*key, time.time() if started_here else float("nan"))

    def weights(self) -> dict:
        secs = self.history.get("seconds", {})
        # NaN would reach the page as invalid JSON and every progress update would be dropped
        if all(isinstance(secs.get(k), (int, float)) and math.isfinite(secs[k]) and secs[k] > 0 for k, _ in PHASES):
            tot = sum(secs[k] for k, _ in PHASES)
            return {k: secs[k] / tot for k, _ in PHASES}
        return DEFAULT_WEIGHTS

    def view(self, st: dict) -> dict:
        """The progress block for the pages (empty when no training runs)."""
        if not st.get("active"): return {}
        self._observe(st)
        gen, phase, target = st.get("gen"), st.get("phase"), st.get("target") or 0   # 0 = until stopped
        if gen is None: return {"gen": None, "gen_pct": 0, "run_pct": 0, "phases": [], "note": st.get("detail", "")}

        pct, note = {"collect": 0.0, "train": 0.0, "test": 0.0}, ""
        if gen == 0:   # generation 0 is only tested (the untrained baseline)
            phases = [("test", "実力テスト")]
            pct["test"] = st.get("progress") or 0.0
        else:
            phases = PHASES
            order = [k for k, _ in PHASES]
            here = order.index(phase) if phase in order else 0
            for k in order[:here]: pct[k] = 1.0
            if phase in ("collect", "test"):
                pct[phase] = st.get("progress") or 0.0
            elif phase == "train":
                log = ROOT / "runs" / f"gen-{gen:03d}.train.log"
                try:
                    if str(log) not in self.records_seen and RECORDS.search(log.read_text(encoding="utf-8", errors="replace")):
                        self.records_seen[str(log)] = time.time()
                except OSError:
                    pass
                frac, spr, note = train_log_progress(log, self.history.get("sec_per_rec"), started=self.records_seen.get(str(log)))
                if spr: self.history["sec_per_rec"] = spr; self._save()
                pct["train"] = frac

        w = {"test": 1.0} if gen == 0 else self.weights()
        known = [pct[k] for k, _ in phases]
        gen_pct = None if any(v is None for v in known) else sum(w[k] * pct[k] for k, _ in phases)
        if gen_pct is None:   # the running phase is not measurable yet: count what is done
            gen_pct = sum(w[k] * (pct[k] or 0.0) for k, _ in phases)
        done_gens = max(0, gen - 1)
        run_pct = None if not target else 0.0 if gen == 0 else min(1.0, (done_gens + gen_pct) / target)
        def state_of(k):
            if gen == 0 and k != "test": return "skip"   # the baseline is only tested
            return "done" if pct[k] == 1.0 and k != phase else "current" if k == phase else "todo"
        # all three phases always, so the screen can show the whole cycle and where in it the loop is
        return {"gen": gen, "target": target, "gen_pct": round(gen_pct, 4), "run_pct": None if run_pct is None else round(run_pct, 4), "note": note,
                "phase": phase, "paused": st.get("state") == "paused", "detail": st.get("detail", ""),
                "phases": [{"key": k, "label": label, "pct": None if pct[k] is None else round(pct[k], 4), "state": state_of(k)}
                           for k, label in PHASES]}


def _best_score(g: dict):
    ev = g.get("eval") or {}
    if ev.get("best_score") is not None: return ev["best_score"]
    games = replays.load(g["gen"]) if ev else []   # generations tested before best_score was recorded
    return max((r["score"] for r in games), default=None)


def _best_pieces(g: dict):
    """The longest test game (pieces placed), from the recordings; the test stops at its max_pieces (500)."""
    games = replays.load(g["gen"]) if g.get("eval") else []
    return max((r["pieces"] for r in games), default=None)


def gens_event(gens: list[dict] | None = None) -> dict:
    """The "gens" event: the test results of every generation, for the score chart (generations are compared by score)."""
    gens = generations.load() if gens is None else gens
    return {"gens": [{"gen": g["gen"], "mean_lines": (g.get("eval") or {}).get("mean_lines"),
                      "best_lines": (g.get("eval") or {}).get("best_lines"),
                      "mean_score": (g.get("eval") or {}).get("mean_score"), "best_score": _best_score(g),
                      "mean_pieces": (g.get("eval") or {}).get("mean_pieces"), "best_pieces": _best_pieces(g)} for g in gens]}
