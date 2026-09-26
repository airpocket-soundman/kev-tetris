"""Live stream screen: Kev plays Tetris generation by generation, from the untrained model to the latest one.

Where a generation's moves come from (--source):
  live    the generation is served by Kev and decides every move now.
  replay  one of its recorded test games is played back (kev_tetris.replays): no GPU needed.
  auto    live while no training runs, replay while it does. Kev-4B takes ~9 GB and training most of the rest of a
          16 GB card, so while training the stream gives its GPU up (a live turn ends early and its server stops).

Whatever the source, while the training loop plays (its practice and test games, kev_tetris.live) the screen follows
those games in real time; recordings fill only the time it spends in kev.train.

    python -m kev_tetris.stream                 # generations from runs/generations.json, served by Kev
    python -m kev_tetris.stream --demo          # no GPU: stand-in players (NOT Kev) to check the screen

Open http://127.0.0.1:8765 (OBS: Browser Source, 1920x1080). The page is fed over Server-Sent Events.

Each generation plays one game, ended at --pieces_per_gen pieces so a strong generation does not hold the stage forever,
then the next generation takes over. With --preload the next generation's server is loaded on a second port while the
current one plays (two Kev-0.8B fit in 16 GB), so the switch is instant.
"""
from __future__ import annotations

import argparse, json, queue, random, shlex, socket, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import generations, kevenv, live, proc, progress, replays, rl, site
from .policy import ROOT, Decision, HeuristicPolicy, KevPolicy, KevServer, RandomPolicy
from .tetris import Game, PIECES

STATIC = Path(__file__).resolve().parent / "static"
MAINTENANCE = ROOT / "runs" / "maintenance.json"   # {"on": true, "title": ..., "message": ...}: the big notice on the stream
MAINTENANCE_DEFAULT = {"title": "高速化改良中",
                       "message": "Docker と高速化ライブラリを導入しています。これまでの学習の成果を引き継いで再開します"}


def maintenance() -> dict:
    try: return {**MAINTENANCE_DEFAULT, **json.loads(MAINTENANCE.read_text(encoding="utf-8"))}
    except (OSError, ValueError): return {**MAINTENANCE_DEFAULT, "on": False}


def set_maintenance(on: bool):
    MAINTENANCE.parent.mkdir(parents=True, exist_ok=True)
    MAINTENANCE.write_text(json.dumps({**maintenance(), "on": on}, ensure_ascii=False), encoding="utf-8")


class Hub:
    """Fan-out of events to every open page. New pages get the last event of each kind first."""

    def __init__(self):
        self.lock, self.clients, self.last = threading.Lock(), [], {}

    def publish(self, kind: str, data: dict):
        msg = {"type": kind, **data}
        with self.lock:
            self.last[kind] = msg
            for q in self.clients: q.put(msg)

    def subscribe(self) -> queue.Queue:
        q = queue.Queue()
        with self.lock:
            for kind in ("gens", "gen", "move", "train", "maintenance"):
                if kind in self.last: q.put(self.last[kind])
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients: self.clients.remove(q)


def make_handler(hub: Hub, trainer: "Trainer"):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            if self.path.startswith("/events"):
                self.send_response(200)
                self.send_header("content-type", "text/event-stream"); self.send_header("cache-control", "no-cache")
                self.end_headers()
                q = hub.subscribe()
                try:
                    while True:
                        try: msg = q.get(timeout=15); self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                        except queue.Empty: self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionError, OSError):
                    pass
                finally:
                    hub.unsubscribe(q)
                return
            if self.path.startswith("/api/train/status"):
                self._json({**trainer.status(), "maintenance": maintenance()}); return
            name = {"/": "index.html", "/index.html": "index.html", "/control": "control.html"}.get(self.path.split("?")[0])
            if not name: self.send_error(404); return
            body = (STATIC / name).read_bytes()
            self.send_response(200); self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("cache-control", "no-store")   # OBS's browser source otherwise keeps an old page
            self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            # a custom header forces a CORS preflight, which this server never answers: other web pages cannot press the buttons
            if self.headers.get("x-kev-control") != "1": self.send_error(403); return
            action = self.path.rsplit("/", 1)[-1]
            if self.path.startswith("/api/maintenance/") and action in ("on", "off"):
                set_maintenance(action == "on"); self._json(maintenance()); return
            fn = {"start": trainer.start, "pause": trainer.pause, "resume": trainer.resume, "stop": trainer.stop}.get(action)
            if not self.path.startswith("/api/train/") or not fn: self.send_error(404); return
            try: fn(); self._json(trainer.status())
            except RuntimeError as e: self._json({**trainer.status(), "message": str(e)}, 409)

        def _json(self, data, code=200):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code); self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
    return H


class Trainer:
    """The RL loop (kev_tetris.rl) as a child process, driven by the control page.

    Pause / resume / stop go through runs/rl_control.json, which the loop reads after every move and while kev.train
    runs (then it suspends kev.train's process tree). A loop started from a terminal is seen and controlled too: its pid
    is in runs/rl_status.json."""

    def __init__(self, args: list[str], launcher: str = "local"):
        self.args, self.child, self.launcher = args, None, launcher
        self.tracker = progress.Tracker()
        self.lock = threading.Lock()

    def _status_file(self) -> dict:
        try: return json.loads(rl.STATUS.read_text(encoding="utf-8"))
        except (OSError, ValueError): return {}

    def running(self) -> bool:
        if self.child and self.child.poll() is None: return True
        st = self._status_file()
        if st.get("state") not in ("running", "paused"): return False
        if st.get("host", socket.gethostname()) != socket.gethostname():   # in a container: trust the heartbeat
            return time.time() - st.get("updated", 0) < 60
        return proc.alive(st.get("pid"))

    def status(self) -> dict:
        st, run = self._status_file(), self.running()
        if not run and st.get("state") in ("running", "paused"): st["state"] = "stopped"   # died without saying so
        st["active"] = run
        with self.lock: prog = self.tracker.view(st)
        return {**st, "command": rl.get_command(), "args": self.args, "progress_view": prog,
                "generations": [{"gen": g["gen"], "eval": g.get("eval"), "train": g.get("train"), "created": g.get("created")}
                                for g in generations.load()]}

    def start(self):
        if self.running():
            if rl.get_command() == "pause": rl.set_command("run")
            return
        rl.set_command("run")
        if self.launcher == "docker":   # docker-compose.yml runs the loop (with its own arguments) in the kev service
            r = subprocess.run(["docker", "compose", "up", "-d", "kev"], cwd=ROOT, capture_output=True, text=True, timeout=300)
            if r.returncode: raise RuntimeError(f"docker compose up failed: {r.stderr.strip()[:300]}")
            return
        log = ROOT / "runs" / "rl.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        f = log.open("a", encoding="utf-8")
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start {' '.join(self.args)} ===\n"); f.flush()
        self.child = subprocess.Popen([sys.executable, "-m", "kev_tetris.rl", *self.args], cwd=ROOT, stdout=f,
                                      stderr=subprocess.STDOUT, env=kevenv.kev_env())

    def pause(self):
        if not self.running(): raise RuntimeError("学習は動いていません")
        rl.set_command("pause")

    def resume(self):
        if not self.running(): raise RuntimeError("学習は動いていません")
        rl.set_command("run")

    def stop(self):
        if not self.running(): raise RuntimeError("学習は動いていません")
        rl.set_command("stop")


def watch_training(hub: Hub, trainer: Trainer, stop):
    """Forward the training status and its progress to the stream screen, and new test results to its chart."""
    last, last_gens, last_maint = None, None, None
    while not stop.is_set():
        m = maintenance()
        if m != last_maint:
            hub.publish("maintenance", m); last_maint = m
        st = trainer.status()
        view = {k: st.get(k) for k in ("state", "phase", "gen", "detail", "progress", "active", "progress_view")}
        if view != last:
            hub.publish("train", view); last = view
        try: mtime = generations.REGISTRY.stat().st_mtime
        except OSError: mtime = None
        if mtime != last_gens:
            last_gens = mtime
            if mtime:
                hub.publish("gens", progress.gens_event())
                # a generation was registered (or the server just started): refresh the GitHub Pages record
                threading.Thread(target=lambda: site.log(f"auto: {site.publish()}"), daemon=True).start()
        time.sleep(1)


# --- who plays ---------------------------------------------------------------------------------------------------

class NoisyHeuristic:
    """--demo only: the heuristic with a random move `eps` of the time, so demo generations visibly improve."""
    def __init__(self, eps, seed=None):
        self.eps, self.h, self.r, self.rng = eps, HeuristicPolicy(), RandomPolicy(seed), random.Random(seed)

    def decide(self, game):
        d = self.h.decide(game)
        if self.rng.random() < self.eps:
            return Decision(self.r.decide(game).placement, d.probs, 0.0)
        return d


def demo_generations(n=6):
    gens = []
    for g in range(n):
        eps = max(0.0, 0.6 - 0.12 * g)
        gens.append({"gen": g, "run": f"demo:{eps}", "parent": g - 1 if g else None, "model": "DEMO(Kevではありません)",
                     "train": None if g == 0 else {"episodes": 16 * g, "records": 900 * g, "trained_on": 900 * g, "decisions": 2500 * g},
                     "eval": {"games": 5, "mean_lines": round(60 * (1 - eps) ** 3, 1), "mean_score": 0, "mean_pieces": 0, "best_lines": 0}})
    return gens


class Seat:
    """One generation, ready to play: a Kev server (or a demo policy) plus its registry entry."""

    def __init__(self, entry, port, demo, reuse=(), replay=None):
        self.entry, self.port, self.demo, self.reuse, self.replay = entry, port, demo, reuse, replay
        self.server, self.url, self.error = None, None, None
        self.ready = threading.Event()

    @property
    def mode(self) -> str:
        return "demo" if self.demo else "replay" if self.replay else "live"

    @property
    def seed(self):
        return self.replay["seed"] if self.replay else None

    def _serving_elsewhere(self):
        """The URL of an already running server (e.g. the manual one on 8009) that serves this generation's checkpoint."""
        want = (kevenv.resolve_run(self.entry["run"]), self.entry["run"])
        for url in self.reuse:
            try:
                with urllib.request.urlopen(f"{url}/v1/models", timeout=2) as r:
                    if json.load(r)["models"][0].get("run") in want: return url
            except (OSError, ValueError, KeyError, IndexError):
                pass
        return None

    def load(self):
        try:
            if self.mode == "live":
                self.url = self._serving_elsewhere()
                if not self.url:
                    self.server = KevServer(self.entry["run"], self.port, log=ROOT / "runs" / f"stream-serve-{self.port}.log").start()
                    self.url = self.server.url
        except Exception as e:   # noqa: BLE001 - reported on screen, the stream moves on
            self.error = str(e)
        finally:
            self.ready.set()
        return self

    def policy(self):
        if self.demo: return NoisyHeuristic(float(self.entry["run"].split(":")[1]))
        if self.replay: return replays.ReplayPolicy(self.replay)
        return KevPolicy(self.url)

    def close(self):
        if self.server: self.server.stop()


# --- the show ----------------------------------------------------------------------------------------------------

def gen_info(entry, gens):
    prev = next((g for g in gens if g["gen"] == entry.get("parent")), None)
    ev, pev = entry.get("eval") or {}, (prev or {}).get("eval") or {}
    return {"gen": entry["gen"], "model": entry.get("model", "Kev"), "run": entry["run"], "train": entry.get("train"),
            "eval": ev, "spp": round(ev["mean_score"] / ev["mean_pieces"], 1) if ev.get("mean_pieces") else None,
            "delta_spp": round(ev["mean_score"] / ev["mean_pieces"] - pev["mean_score"] / pev["mean_pieces"], 1)
            if ev.get("mean_pieces") and pev.get("mean_pieces") else None,
            "cumulative_episodes": sum((g.get("train") or {}).get("episodes", 0) for g in gens if g["gen"] <= entry["gen"]),
            "total_gens": len(gens), "latest": entry["gen"] == max(g["gen"] for g in gens)}


def play(hub, seat, gens, a, stop, next_entry, should_yield=lambda: False):
    """One turn. -> result dict; "ended" says why it ended early (yield to training, lost server)."""
    entry = seat.entry
    seed = seat.seed if seat.seed is not None else a.seed if a.seed is not None else random.randrange(1 << 30)
    game, pol, ended = Game(seed=seed), seat.policy(), None
    hub.publish("gen", {**gen_info(entry, gens), "mode": seat.mode})
    hub.publish("move", {**game.snapshot(), "post": game.board, "cells": [], "piece": None, "thinking": [],
                         "latency_ms": 0, "left": a.pieces_per_gen, "next_gen": next_entry["gen"] if next_entry else None})
    while not game.over and game.pieces < a.pieces_per_gen and not stop.is_set():
        if should_yield():
            ended = "training"; break
        t0 = time.time()
        pre = [row[:] for row in game.board]
        placements = game.placements()
        try:
            d = pol.decide(game)
        except StopIteration:     # the recording ends (it was capped shorter than this turn)
            break
        except OSError as e:      # the server went away (e.g. the manual Kev was restarted with another model)
            ended = f"Kevとの通信が切れました: {e}"; break
        top = sorted(d.probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        by_key = {p.key: p for p in placements}
        thinking = [{"key": k, "p": round(p, 3), "cells": by_key[k].cells if k in by_key else [],
                     "chosen": k == d.placement.key} for k, p in top]
        piece = game.current
        full = [y for y in range(len(pre)) if all(pre[y][x] or (x, y) in d.placement.cells for x in range(len(pre[0])))]
        game.step(d.placement)
        hub.publish("move", {**game.snapshot(), "board": pre, "post": game.board, "cells": d.placement.cells, "piece": piece,
                             "color": PIECES.index(piece) + 1, "cleared": full, "thinking": thinking,
                             "latency_ms": round(d.latency_ms, 1), "left": a.pieces_per_gen - game.pieces,
                             "next_gen": next_entry["gen"] if next_entry else None})
        time.sleep(max(0.0, a.move_delay - (time.time() - t0)))
    return {"gen": entry["gen"], "lines": game.lines, "score": game.score, "pieces": game.pieces, "over": game.over,
            "mode": seat.mode, "ended": ended}


def follow_training(hub, a, stop, load_gens):
    """Relay the training loop's games (runs/rl_live.json) move by move until it stops playing."""
    last_seq, last_key, last_sent = None, None, 0.0
    while not stop.is_set():
        data = live.read()
        if not data: return
        # at most ~2 moves a second reach the page: OBS renders the page and encodes on the GPU Kev is busy with
        if (data["pid"], data["seq"]) != last_seq and time.time() - last_sent >= a.relay_interval:
            last_sent = time.time()
            last_seq = (data["pid"], data["seq"])
            info, mv = data["info"], data["move"]
            key = (data["pid"], info["gen"], info["phase"], info["game"])
            if key != last_key:
                last_key = key
                gens = load_gens()
                entry = next((g for g in gens if g["gen"] == info["gen"]), None) or \
                    {"gen": info["gen"], "run": "", "parent": info["gen"] - 1 if info["gen"] else None,
                     "model": a.model_name, "train": None, "eval": None}
                if all(g["gen"] != entry["gen"] for g in gens): gens = gens + [entry]
                hub.publish("gens", progress.gens_event(gens))
                hub.publish("gen", {**gen_info(entry, gens), "mode": info["phase"], "training_gen": info["training_gen"],
                                    "game": info["game"], "games": info["games"]})
            what = "練習試合" if info["phase"] == "practice" else "テスト"
            hub.publish("move", {**mv, "left": info["max_pieces"] - mv["pieces"], "next_gen": None,
                                 "turn_text": f"{what} {info['game']}/{info['games']} ゲーム目",
                                 "turn_progress": (info["game"] - 1 + mv["pieces"] / info["max_pieces"]) / info["games"]})
        time.sleep(0.05)


def run_show(hub, a, stop, trainer):
    results: dict[int, list] = {}   # results of this stream session per generation
    turns: dict[int, int] = {}   # which recorded game of a generation to replay next

    def training() -> bool:
        return trainer.running()

    def seat_for(entry, port, reuse):
        """-> a Seat, or None when this generation cannot be shown now (training runs and it has no recording)."""
        if a.demo: return Seat(entry, port, True)
        live_seat = Seat(entry, port, False, reuse)
        if a.source == "live": return live_seat
        if a.source == "auto":
            if live_seat._serving_elsewhere(): return live_seat            # e.g. the manual Kev on 8009: costs nothing extra
            free = rl.gpu_free_gb()
            if not training() and (free is None or free >= a.need_serve_gb): return live_seat
        games = replays.load(entry["gen"])
        if not games: return None
        k = turns.get(entry["gen"], 0); turns[entry["gen"]] = k + 1
        return Seat(entry, port, False, replay=games[k % len(games)])

    def publish_gens(gens):
        hub.publish("gens", {**progress.gens_event(gens), "live": results})

    def load_gens():
        gens = demo_generations() if a.demo else generations.load()
        if not gens:   # nothing trained yet: the untrained Kev plays as generation 0
            gens = [{"gen": 0, "run": a.start, "parent": None, "model": a.model_name, "train": None, "eval": None}]
        return gens

    def training_plays() -> bool:
        return not a.demo and live.read() is not None

    while not stop.is_set():
        if maintenance().get("on"):   # the GPU and Kev are being rebuilt: load nothing, the page shows the notice
            time.sleep(2); continue
        if training_plays():
            follow_training(hub, a, stop, load_gens)
            continue
        gens = load_gens()
        gens_order = sorted((g for g in gens if not a.only or g["gen"] in a.only), key=lambda g: g["gen"])
        publish_gens(gens)

        ports = [a.kev_port, a.kev_port + 1]
        reuse = [u for u in a.reuse.split(",") if u]
        shown = 0
        seat = None
        for i, entry in enumerate(gens_order):
            if stop.is_set() or training_plays() or maintenance().get("on"): break
            if seat is None or seat.entry is not entry:
                seat = seat_for(entry, ports[i % 2], reuse)
                if seat is None: continue
                if seat.mode == "live":
                    hub.publish("switch", {"to": gen_info(entry, gens), "message": f"第{entry['gen']}世代 を読み込み中…"})
                seat.load()
            if seat.error:
                hub.publish("switch", {"to": gen_info(entry, gens), "message": f"第{entry['gen']}世代 の起動に失敗: {seat.error}"})
                time.sleep(5); seat.close(); seat = None; continue
            nxt = gens_order[i + 1] if i + 1 < len(gens_order) else None
            pre = None
            if nxt and a.preload and seat.mode == "live" and not training():
                pre = Seat(nxt, ports[(i + 1) % 2], a.demo, reuse)
                threading.Thread(target=pre.load, daemon=True).start()
            owns_gpu = seat.server is not None
            yield_now = (lambda: (owns_gpu and a.source == "auto" and training()) or training_plays() or maintenance().get("on"))
            result = play(hub, seat, gens, a, stop, nxt, yield_now)
            shown += 1
            if result["ended"] == "training":
                pass   # the training loop is playing: the screen follows it next
            elif result["ended"]:
                hub.publish("switch", {"to": gen_info(entry, gens), "message": result["ended"]})
                time.sleep(5)
            else:
                results.setdefault(entry["gen"], []).append(result)
                publish_gens(gens)
                hub.publish("result", result)
                time.sleep(a.pause)
            if pre and not training():
                if not pre.ready.is_set():
                    hub.publish("switch", {"to": gen_info(nxt, gens), "message": f"第{nxt['gen']}世代 を読み込み中…"})
                pre.ready.wait()
                seat.close(); seat = pre
            else:
                if pre: threading.Thread(target=lambda p=pre: (p.ready.wait(), p.close()), daemon=True).start()
                seat.close(); seat = None
        if seat: seat.close()
        if not shown and not training_plays():   # training runs, nothing to follow and no recording yet
            hub.publish("switch", {"to": None, "message": "Kev を準備しています。まもなく練習の様子をお見せします"})
            for _ in range(10):
                if stop.is_set() or training_plays(): break
                time.sleep(0.5)
        if not a.loop: break


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tetris live stream screen: Kev generations take turns")
    ap.add_argument("--port", type=int, default=8765, help="the page")
    ap.add_argument("--kev_port", type=int, default=8011, help="Kev servers started by the stream use this port and the next")
    ap.add_argument("--reuse", default="http://127.0.0.1:8009", help="comma-separated Kev servers to use when they already serve a generation's checkpoint (the manually started one)")
    ap.add_argument("--start", default="jaredpalmer/kev-4b", help="generation 0 when nothing is trained yet")
    ap.add_argument("--model_name", default="Kev-4B")
    ap.add_argument("--source", choices=["auto", "live", "replay"], default="auto", help="live play, recorded test games, or live unless training runs (see the module doc)")
    ap.add_argument("--train_launcher", choices=["local", "docker"], default="local", help="where 学習開始 starts the loop: this Python, or the kev service of docker-compose.yml")
    ap.add_argument("--train_args", default="--generations 0", help="arguments for kev_tetris.rl when the control page starts training")
    ap.add_argument("--pieces_per_gen", type=int, default=150, help="a generation's turn ends after this many pieces (or game over)")
    ap.add_argument("--relay_interval", type=float, default=0.35, help="seconds between relayed training moves (lower = smoother, heavier for OBS)")
    ap.add_argument("--move_delay", type=float, default=0.18, help="seconds per move, so viewers can follow")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds to show a generation's result before the switch")
    ap.add_argument("--need_serve_gb", type=float, default=10.0, help="auto: start a generation's own Kev server only with this much free GPU memory, else replay (~3 for Kev-0.8B)")
    ap.add_argument("--preload", type=int, choices=[0, 1], default=0, help="load the next generation while this one plays: needs VRAM for two models (Kev-0.8B yes, Kev-4B not on 16 GB)")
    ap.add_argument("--only", type=lambda s: [int(x) for x in s.split(",")], default=None, help="comma-separated generations to show, e.g. 0,5,10")
    ap.add_argument("--loop", type=int, choices=[0, 1], default=1)
    ap.add_argument("--seed", type=int, default=None, help="same piece sequence for every generation (fair comparison)")
    ap.add_argument("--demo", action="store_true", help="stand-in players instead of Kev, to check the screen without a GPU")
    a = ap.parse_args(argv)

    hub, stop = Hub(), threading.Event()
    trainer = Trainer(shlex.split(a.train_args), a.train_launcher)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(hub, trainer))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    threading.Thread(target=watch_training, args=(hub, trainer, stop), daemon=True).start()
    print(f"stream screen: http://127.0.0.1:{a.port}   training controls: http://127.0.0.1:{a.port}/control  (Ctrl+C to stop)", flush=True)
    try:
        run_show(hub, a, stop, trainer)
        while not a.loop: time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); srv.shutdown()


if __name__ == "__main__":
    main()
