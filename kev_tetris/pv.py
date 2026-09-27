"""A ~65 s promo video: how Kev learned Tetris, with clips of recorded test games and a synthesized chiptune BGM
(runs/videos/kev-tetris-pv.mp4).

    python -m kev_tetris.pv
"""
from __future__ import annotations

import json, math, random, subprocess, wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from . import generations
from .tetris import HEIGHT, WIDTH, Game
from .video import COLORS, PIECE_OF, font

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS, CELL = 1280, 720, 20, 30
PANEL, GRID, TEXT, MUTED, ACCENT = (24, 22, 58), (44, 40, 88), (246, 244, 255), (176, 170, 226), (94, 231, 255)
PINK, YELLOW, GREEN, ORANGE = (255, 110, 180), (255, 214, 90), (96, 230, 150), (255, 160, 70)
STAGES = [  # (label, first gen, colour) for the chart and the step tags
    ("強化学習", 0, (126, 142, 255)), ("模倣学習", 16, ORANGE), ("0.8B・ロールアウト", 23, GREEN),
    ("盤面予測能力の練習", 44, PINK), ("ヒントなし", 58, YELLOW)]

# gen, seconds, step tag, title, lines, clip window, hints given (the computed outcome of each option)
SCENES = [
    (0, 5, "学習前", "Kev-4B、テトリスを知らない", ["置き場所の候補から1つ選ぶ決定モデル", "46手ほどでゲームオーバー"], "start", True),
    (6, 6, "STEP 1 強化学習", "自分の良い手を強化する", ["報酬で「期待より良かった手」を覚える", "生き残れるが、1列消しばかり"], "start", True),
    (15, 4, "壁", "公式ルールにしたら伸び悩み", ["SRS回転・出現位置のゲームオーバー", "約30点/ミノで足踏み"], "start", True),
    (20, 6, "STEP 2 模倣学習", "2手先読みの「先生」を真似る", ["手作りの評価式で探索する先生の手を覚える", "→ 59.6点/ミノ"], "tetris", True),
    (23, 4, "STEP 3 蒸留", "Kev-4B → Kev-0.8B", ["最良の4Bを先生にして小型化", "推論が速くなり、配信もなめらかに"], "tetris", True),
    (41, 6, "STEP 4 ロールアウト", "10手先まで試して選ぶ", ["候補手を実際に打ち進めて報酬で比較", "テトリス見逃しは悪手、準備完成は加点"], "tetris", True),
    (46, 6, "STEP 5 盤面予測能力の練習", "置いた後の盤面を当てる", ["消える行・穴・高さ・テトリス準備を当てる問題", "ヒントありで最高 63.3点/ミノ"], "tetris", True),
]
FINAL = (66, 9, "STEP 6 盤面予測能力を Kev に取り込む", "ヒントなしで、自分で読む",
         ["渡すのは 盤面・戦略の指示・置く場所だけ", "59.6点/ミノ・テトリス11回 → 合格!"], "tetris", False)


def spp_of(x):
    ev = x.get("eval") or {}
    return ev.get("score_per_piece") or (ev["mean_score"] / max(1, ev["mean_pieces"]) if ev.get("mean_score") is not None else None)


def stage_of(gen):
    return [s for s in STAGES if gen >= s[1]][-1]


def ease_back(t):   # 0..1 -> overshooting ease-out (the pop)
    t = min(1.0, max(0.0, t)); c = 1.70158
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


_BG = None


def background():
    """A deep indigo-to-violet gradient with soft colour blobs."""
    global _BG
    if _BG is None:
        y = np.linspace(0, 1, H)[:, None]; x = np.linspace(0, 1, W)[None, :]
        top, bot = np.array([16, 14, 44]), np.array([46, 18, 72])
        img = top + (bot - top) * (0.6 * y + 0.4 * x)[..., None]
        for cx, cy, r, col in [(0.85, 0.15, 0.35, (60, 40, 140)), (0.1, 0.9, 0.4, (90, 30, 110))]:
            d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / r
            img = img + (np.clip(1 - d, 0, 1) ** 2)[..., None] * (np.array(col) - img) * 0.5
        _BG = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    return _BG.copy()


def board_img(d, board, ox, oy, cells=(), piece=None, cell=CELL):
    top = len(board) - HEIGHT
    d.rounded_rectangle([ox - 8, oy - 8, ox + WIDTH * cell + 8, oy + HEIGHT * cell + 8], 10, fill=PANEL, outline=(120, 110, 220), width=2)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            c = board[top + y][x]
            r = [ox + x * cell + 1, oy + y * cell + 1, ox + (x + 1) * cell - 1, oy + (y + 1) * cell - 1]
            if c: d.rounded_rectangle(r, 4, fill=COLORS.get(PIECE_OF.get(c, "T"), (180, 180, 200)))
            else: d.rectangle(r, outline=GRID)
    for x, y in cells:
        if y >= top:
            d.rounded_rectangle([ox + x * cell + 1, oy + (y - top) * cell + 1, ox + (x + 1) * cell - 1, oy + (y - top + 1) * cell - 1],
                                4, fill=COLORS[piece], outline=(255, 255, 255), width=3)


def pill(d, x, y, text, col, size=26):
    w = int(d.textlength(text, font=font(size)))
    d.rounded_rectangle([x, y, x + w + 28, y + size + 18], (size + 18) // 2, fill=col)
    d.text((x + 14, y + 7), text, font=font(size), fill=(20, 16, 40))
    return w + 28


def replay_states(gen):
    """(board, piece cells, piece, score, lines, pieces, tetrises, over) after each move of the gen's best test game."""
    rec = json.loads((ROOT / "runs" / "replays" / f"gen-{gen:03d}.json").read_text(encoding="utf-8"))
    games = rec["games"] if isinstance(rec, dict) else rec
    entry = next((x for x in generations.load() if x["gen"] == gen), {})
    rg = max(games, key=lambda x: x["score"])
    g = Game(seed=rg["seed"], rules=rg.get("rules", entry.get("rules", 1)))
    out, tet = [], 0
    for m in rg["moves"]:
        ps = {p.key: p for p in g.placements()}
        p, piece = ps[m["key"]], g.current
        tet += g.step(p) == 4
        out.append(([r[:] for r in g.board], p.cells, piece, g.score, g.lines, g.pieces, tet, g.over))
        if g.over: break
    return out, entry


def window(states, n, mode):
    if mode == "start" or len(states) <= n: return states[:n]
    best = max(range(0, len(states) - n + 1, 5), key=lambda s: states[s + n - 1][6] - (states[s - 1][6] if s else 0))
    return states[best:best + n]


class Confetti:
    def __init__(self, n=120, seed=7):
        rng = random.Random(seed)
        cols = [PINK, YELLOW, GREEN, ACCENT, ORANGE, (180, 140, 255)]
        self.p = [(rng.uniform(0, W), rng.uniform(-H, 0), rng.uniform(120, 260), rng.uniform(-60, 60), rng.choice(cols),
                   rng.uniform(0, 6.28), rng.uniform(6, 12)) for _ in range(n)]

    def draw(self, d, t):
        for x0, y0, vy, vx, col, ph, s in self.p:
            x, y = x0 + vx * t + 18 * math.sin(ph + t * 3), y0 + vy * t
            if 0 <= y <= H:
                a = s * abs(math.cos(ph + t * 5))
                d.rectangle([x - s / 2, y - a / 2, x + s / 2, y + a / 2 + 1], fill=col)


def scene_frames(gen, secs, tag, title, lines, mode, hints, idx, confetti=False):
    states, entry = replay_states(gen)
    n = secs * FPS
    clip = window(states, n, mode)
    if len(clip) < n:   # a short game: slow it down to fill the scene, then hold the end
        k = max(1, n // max(1, len(clip)))
        clip = [s for s in clip for _ in range(min(k, 3))]
        clip += [clip[-1]] * (n - len(clip))
    col = stage_of(gen)[2]
    conf = Confetti() if confetti else None
    for t, (board, cells, piece, score, nlines, pieces, tet, over) in enumerate(clip[:n]):
        img = background(); d = ImageDraw.Draw(img)
        board_img(d, board, 90, 60, cells, piece)
        # the hint badge over the board: what Kev was given with each option
        if hints: pill(d, 96, 18, "ヒントあり: 置いた結果を計算して渡す", ORANGE, 20)
        else: pill(d, 96, 18, "ヒントなし: 結果は Kev が予測", YELLOW, 20)
        s = t / FPS
        x0 = 480 + int(60 * (1 - ease_back(s / 0.5)))
        pill(d, x0, 70, tag, col)
        d.text((x0, 130), title, font=font(46), fill=TEXT)
        for k, line in enumerate(lines):
            xk = 480 + int(80 * (1 - ease_back((s - 0.15 - 0.1 * k) / 0.5)))
            d.text((xk, 210 + k * 44), line, font=font(28), fill=MUTED)
        d.text((480, 340), f"{entry.get('model', 'Kev-4B')}  第{gen}世代  テストのベストゲーム", font=font(24), fill=MUTED)
        stats = [("得点", f"{score:,}"), ("ライン", nlines), ("手数", pieces), ("テトリス", tet)]
        for k, (lab, val) in enumerate(stats):
            d.text((480 + k * 190, 400), lab, font=font(22), fill=MUTED); d.text((480 + k * 190, 430), str(val), font=font(40), fill=ACCENT)
        spp = spp_of(entry)
        if spp: d.text((480, 510), f"1ミノあたり {spp:.1f} 点(テスト平均)", font=font(30), fill=col)
        if over and t > n - FPS * 2: d.text((480, 570), "GAME OVER", font=font(40), fill=(255, 92, 120))
        if conf and s > secs - 4: conf.draw(d, s - (secs - 4))
        progress(d, idx)
        yield img


def progress(d, idx):
    for k in range(len(SCENES) + 1):
        x = 480 + k * 36
        d.ellipse([x, 670, x + 14, 684], fill=ACCENT if k <= idx else GRID)


def compare_frames(secs):
    """What changed at step 6: the hint text the program computed for each option, then only where the piece goes."""
    n = secs * FPS
    for t in range(n):
        s = t / FPS
        img = background(); d = ImageDraw.Draw(img)
        d.text((W // 2, 60), "STEP 5 まで と STEP 6 から、Kev に渡す候補の違い", font=font(36), fill=TEXT, anchor="mm")
        for side, (x, head, col, body, note) in enumerate([
            (70, "STEP 5 まで: ヒントあり", ORANGE,
             [("rot 1, col 9:", TEXT), ("clears 4", ORANGE), ("holes 0 (+0)", ORANGE), ("height 8", ORANGE), ("well 0 …", ORANGE)],
             "置いた結果をプログラムが計算して渡す"),
            (670, "STEP 6 から: ヒントなし", YELLOW,
             [("rot 1,", TEXT), ("cells 9:1 9:2 9:3 9:4", TEXT)],
             "結果は Kev が盤面から自分で予測")]):
            a = ease_back((s - 0.2 - 0.5 * side) / 0.5)
            yy = 120 + int(60 * (1 - a))
            d.rounded_rectangle([x, yy, x + 540, yy + 430], 18, fill=PANEL, outline=col, width=3)
            pill(d, x + 20, yy + 20, head, col, 26)
            for k, (txt, c) in enumerate(body):
                d.text((x + 34, yy + 100 + k * 50), txt, font=font(34), fill=c)
            d.text((x + 34, yy + 370), note, font=font(26), fill=col)
        if s > 1.6:
            d.text((W // 2, 640), "盤面予測能力を Kev の中に取り込む = STEP 6", font=font(32), fill=ACCENT, anchor="mm")
        yield img


def text_frames(secs, big, small, sub=None, confetti=False):
    conf = Confetti(seed=3) if confetti else None
    for t in range(secs * FPS):
        s = t / FPS
        img = background(); d = ImageDraw.Draw(img)
        k = ease_back(s / 0.6)
        size = max(10, int(64 * (0.6 + 0.4 * k)))
        d.text((W // 2, 280), big, font=font(size), fill=TEXT, anchor="mm")
        if s > 0.4: d.text((W // 2, 370), small, font=font(30), fill=MUTED, anchor="mm")
        if sub and s > 0.7: d.text((W // 2, 430), sub, font=font(24), fill=ACCENT, anchor="mm")
        if conf: conf.draw(d, s)
        yield img


def chart_frames(secs):
    gens = sorted((x for x in generations.load() if spp_of(x)), key=lambda x: x["gen"])
    pts = [(x["gen"], spp_of(x)) for x in gens]
    gmax, ymax = max(p[0] for p in pts), 70
    X0, Y0, X1, Y1 = 120, 120, 1180, 590
    px = lambda g: X0 + (X1 - X0) * g / gmax
    py = lambda v: Y1 - (Y1 - Y0) * v / ymax
    n = secs * FPS
    for t in range(n):
        img = background(); d = ImageDraw.Draw(img)
        d.text((X0, 40), "1ミノあたりの得点(テスト平均)の推移", font=font(34), fill=TEXT)
        for v in (0, 20, 40, 60):
            d.line([X0, py(v), X1, py(v)], fill=GRID, width=1); d.text((X0 - 50, py(v) - 12), str(v), font=font(20), fill=MUTED)
        for label, g0, col in STAGES:
            d.text((px(g0), Y1 + 16), f"▲{label}", font=font(19), fill=col)
        d.text((px(58) - 150, Y1 + 46), "ここからヒントなしで測定", font=font(18), fill=YELLOW)
        shown = int(len(pts) * min(1.0, t / (n * 0.7))) + 1
        for (g1, v1), (g2, v2) in zip(pts[:shown], pts[1:shown]):
            d.line([px(g1), py(v1), px(g2), py(v2)], fill=stage_of(g2)[2], width=4)
        for g, v in pts[:shown]:
            d.ellipse([px(g) - 5, py(v) - 5, px(g) + 5, py(v) + 5], fill=stage_of(g)[2])
        yield img


def bgm(seconds: float, path: Path, sr: int = 44100):
    """An original upbeat chiptune loop (I-V-vi-IV in C, 132 BPM): square arpeggios, a triangle bass, noise drums."""
    bpm = 132; beat = 60 / bpm; n = int(seconds * sr)
    out = np.zeros(n)
    t_all = np.arange(n) / sr
    chords = [[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]]   # C G Am F
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)

    def tone(start, dur, f, kind, vol):
        i0, i1 = int(start * sr), min(n, int((start + dur) * sr))
        if i0 >= n: return
        tt = np.arange(i1 - i0) / sr
        ph = (f * tt) % 1
        w = np.sign(np.sin(2 * np.pi * f * tt)) * 0.6 if kind == "square" else 4 * np.abs(ph - 0.5) - 1
        env = np.minimum(1, tt / 0.005) * np.exp(-tt * (6 if kind == "square" else 2.5))
        out[i0:i1] += vol * w * env

    rng = np.random.default_rng(1)
    def noise(start, dur, vol, decay):
        i0, i1 = int(start * sr), min(n, int((start + dur) * sr))
        if i0 >= n: return
        tt = np.arange(i1 - i0) / sr
        out[i0:i1] += vol * rng.uniform(-1, 1, i1 - i0) * np.exp(-tt * decay)

    def kick(start):
        i0, i1 = int(start * sr), min(n, int((start + 0.18) * sr))
        if i0 >= n: return
        tt = np.arange(i1 - i0) / sr
        out[i0:i1] += 0.9 * np.sin(2 * np.pi * (50 + 110 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 18)

    bars = int(seconds / (4 * beat)) + 1
    melody = [0, 2, 1, 2, 0, 2, 1, 2]   # chord-tone arpeggio pattern (eighth notes)
    for b in range(bars):
        ch = chords[b % 4]; t0 = b * 4 * beat
        for k in range(8):
            note = ch[melody[k]] + 12 + (12 if (b // 4) % 2 and k in (3, 7) else 0)
            tone(t0 + k * beat / 2, beat / 2, hz(note), "square", 0.16)
        for k in range(4):
            tone(t0 + k * beat, beat * 0.9, hz(ch[0] - 24), "tri", 0.35)
        if b >= 2:   # drums after a two-bar intro
            for k in range(4):
                if k in (0, 2): kick(t0 + k * beat)
                else: noise(t0 + k * beat, 0.15, 0.35, 25)            # snare
            for k in range(8): noise(t0 + k * beat / 2 + 0.001, 0.04, 0.12, 120)   # hat
    fade = np.minimum(1, t_all / 1.0) * np.minimum(1, (seconds - t_all) / 2.0)
    out = out * fade
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.7
    data = (np.clip(out, -1, 1) * 32767).astype(np.int16)
    stereo = np.stack([data, data], axis=1).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes(stereo)


VOICE, RATE = "ja-JP-NanamiNeural", "+35%"   # the same voice as the IchiPing Solist PV (edge-tts)
NARRATION = {
    "title": "ローカルの決定モデル、ケブ。テトリスを覚えるまでの道のりです。",
    0: "学習前のケブは、置き場所の候補から選ぶだけ。すぐにゲームオーバー。",
    6: "まずは強化学習。うまくいった手を強化して、生き残れるようになりました。",
    15: "でも公式ルールにしたら、伸び悩みます。",
    20: "そこで先読みの先生をまねる模倣学習。一気に上達しました。",
    23: "軽いゼロてんはちビーのモデルに蒸留して、スピードアップ。",
    41: "候補手を十手先まで試して選ぶロールアウトで、テトリスを狙えるように。",
    46: "置いた後の盤面を当てる練習。ここまでは、置いた結果をヒントとして渡していました。",
    "compare": "ステップ6では、そのヒントをなくします。渡すのは、置く場所だけ。",
    66: "盤面予測能力をケブの中に取り込み、ヒントなしで合格ラインを突破!",
    "chart": "得点はここまで伸びてきました。",
    "end": "いまも学習中。次は、指示ひとつで打ち方を変えます。",
}


def narration(folder: Path, sr: int = 44100) -> dict:
    """Synthesize each line (edge-tts, sent to Microsoft's online TTS) -> {key: mono float samples at sr}."""
    import asyncio, edge_tts
    folder.mkdir(parents=True, exist_ok=True)
    async def synth():
        for k, text in NARRATION.items():
            f = folder / f"{k}.mp3"
            if not f.exists(): await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(f))
    asyncio.run(synth())
    out = {}
    for k in NARRATION:
        raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(folder / f"{k}.mp3"),
                              "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"], capture_output=True, check=True).stdout
        out[k] = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768
    return out


def main():
    out = ROOT / "runs" / "videos" / "kev-tetris-pv.mp4"
    silent, music = out.with_name("pv-silent.mp4"), out.with_name("pv-bgm.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio_ffmpeg.write_frames(str(silent), (W, H), fps=FPS, quality=8, macro_block_size=16)
    w.send(None)
    frames = 0
    def put(fs):
        nonlocal frames
        for f in fs: w.send(f.tobytes()); frames += 1
    sr = 44100
    voice = narration(out.parent / "narration", sr)
    starts = {}   # narration key -> start frame
    def secs(key, base):   # a scene lasts at least as long as its narration (+ a breath)
        return max(base, math.ceil(len(voice[key]) / sr + 0.6))
    def scene(key, frames_iter):
        starts[key] = frames; put(frames_iter)
    scene("title", text_frames(secs("title", 3), "Kev がテトリスを覚えるまで", "ローカルの決定モデル Kev × 強化学習・模倣学習", "Qwen3.5 + LoRA + ポインタヘッド / RTX 5060 Ti 16GB"))
    for i, sc in enumerate(SCENES):
        g = sc[0]; scene(g, scene_frames(g, secs(g, 3), *sc[2:], i))
    scene("compare", compare_frames(secs("compare", 5)))
    scene(FINAL[0], scene_frames(FINAL[0], secs(FINAL[0], 7), *FINAL[2:], len(SCENES), confetti=True))
    scene("chart", chart_frames(secs("chart", 5)))
    scene("end", text_frames(secs("end", 4), "デシジョンモデルとして合格!", "いまも学習中: 第67世代〜 ロールアウトを増やしてさらに上へ", "次は、指示ひとつで打ち方を変える / X でライブ配信中", confetti=True))
    w.close()
    seconds = frames / FPS
    bgm(seconds, music)
    # the narration over the BGM, which ducks while a line is spoken
    with wave.open(str(music)) as wf: bg = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).reshape(-1, 2)[:, 0] / 32768
    vo = np.zeros_like(bg)
    for k, f0 in starts.items():
        i0 = int((f0 / FPS + 0.3) * sr); v = voice[k][:max(0, len(vo) - i0)]
        vo[i0:i0 + len(v)] += v
    active = np.convolve(np.abs(vo) > 0.01, np.ones(int(0.25 * sr)) / (0.25 * sr), mode="same") > 0.02
    duck = np.convolve(np.where(active, 0.35, 1.0), np.ones(int(0.1 * sr)) / (0.1 * sr), mode="same")
    mix = bg * 0.6 * duck + vo * 1.0
    mix = mix / max(1e-9, np.max(np.abs(mix))) * 0.9
    data = (mix * 32767).astype(np.int16)
    with wave.open(str(music), "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(np.stack([data, data], 1).tobytes())
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(silent), "-i", str(music),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)], check=True)
    silent.unlink(missing_ok=True)
    print(out, f"{seconds:.1f} s")


if __name__ == "__main__":
    main()
