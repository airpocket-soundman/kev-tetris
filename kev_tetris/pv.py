"""A ~60 s promo video: how Kev learned Tetris, with clips of recorded test games (runs/videos/kev-tetris-pv.mp4).

    python -m kev_tetris.pv
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw

from . import generations
from .tetris import HEIGHT, WIDTH, Game
from .video import COLORS, PIECE_OF, font

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS, CELL = 1280, 720, 20, 30
BG, PANEL, GRID, TEXT, MUTED, ACCENT = (11, 14, 26), (20, 25, 44), (28, 34, 58), (232, 236, 248), (142, 152, 196), (94, 231, 255)
STAGES = [  # (label, first gen, colour) for the chart and the step tags
    ("強化学習", 0, (126, 142, 255)), ("模倣学習", 16, (250, 190, 80)), ("0.8B・ロールアウト", 23, (96, 224, 150)),
    ("盤面推論", 44, (244, 110, 160)), ("予測なし", 58, (255, 214, 90))]

SCENES = [  # gen, seconds, step tag, title, lines, clip window ("start" or "tetris")
    (0, 6, "学習前", "Kev-4B、テトリスを知らない", ["置き方の候補から選ぶだけ", "46手ほどでゲームオーバー"], "start"),
    (6, 7, "STEP 1 強化学習", "自分の良い手を強化する", ["報酬で「期待より良かった手」を覚える", "生き残れるが、1列消しばかり"], "start"),
    (15, 4, "壁", "公式ルールにしたら伸び悩み", ["SRS回転・出現位置のゲームオーバー", "約30点/ミノで足踏み"], "start"),
    (20, 7, "STEP 2 模倣学習", "2手先読みの「先生」を真似る", ["手作りの評価式で探索する先生の手を", "Kevが出会った局面で覚える → 59.6点/ミノ"], "tetris"),
    (23, 4, "STEP 3 蒸留", "Kev-4B → Kev-0.8B", ["最良の4Bを先生にして小型化", "推論 約1/2、配信もなめらかに"], "tetris"),
    (41, 7, "STEP 4 ロールアウト", "10手先まで試して選ぶ", ["候補手を実際に打ち進めて報酬で比較", "テトリス見逃しは悪手、準備完成は加点"], "tetris"),
    (46, 6, "STEP 5 盤面推論", "置いた後の盤面を思い描く", ["消える行・穴・高さ・テトリス準備を当てる練習", "計算した予測つきで 63.3点/ミノ"], "tetris"),
    (60, 9, "STEP 6 自分で読む", "予測なし、盤面と戦略だけで", ["置いた結果の計算は一切渡さない", "57.9点/ミノ・全ゲーム500手 → 合格"], "tetris"),
]


def spp_of(x):
    ev = x.get("eval") or {}
    return ev.get("score_per_piece") or (ev["mean_score"] / max(1, ev["mean_pieces"]) if ev.get("mean_score") is not None else None)


def stage_of(gen):
    return [s for s in STAGES if gen >= s[1]][-1]


def board_img(d, board, ox, oy, cells=(), piece=None, cell=CELL):
    top = len(board) - HEIGHT
    d.rectangle([ox - 5, oy - 5, ox + WIDTH * cell + 5, oy + HEIGHT * cell + 5], fill=PANEL)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            c = board[top + y][x]
            r = [ox + x * cell + 1, oy + y * cell + 1, ox + (x + 1) * cell - 1, oy + (y + 1) * cell - 1]
            if c: d.rectangle(r, fill=COLORS.get(PIECE_OF.get(c, "T"), (180, 180, 200)))
            else: d.rectangle(r, outline=GRID)
    for x, y in cells:
        if y >= top:
            d.rectangle([ox + x * cell + 1, oy + (y - top) * cell + 1, ox + (x + 1) * cell - 1, oy + (y - top + 1) * cell - 1],
                        fill=COLORS[piece], outline=(255, 255, 255), width=3)


def replay_states(gen):
    """(board, piece cells, piece, score, lines, pieces, tetrises) after each move of the gen's best test game."""
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


def scene_frames(gen, secs, tag, title, lines, mode, idx):
    states, entry = replay_states(gen)
    n = secs * FPS
    clip = window(states, n, mode)
    if len(clip) < n:   # a short game: slow it down to fill the scene, then hold the end
        k = max(1, n // max(1, len(clip)))
        clip = [s for s in clip for _ in range(min(k, 3))]
        clip += [clip[-1]] * (n - len(clip))
    col = stage_of(gen)[2]
    for t, (board, cells, piece, score, nlines, pieces, tet, over) in enumerate(clip[:n]):
        img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
        board_img(d, board, 90, 60, cells, piece)
        x0 = 480
        d.rounded_rectangle([x0, 70, x0 + 12 + 26 * len(tag), 110], 8, fill=col)
        d.text((x0 + 10, 74), tag, font=font(26), fill=BG)
        d.text((x0, 130), title, font=font(46), fill=TEXT)
        for k, line in enumerate(lines): d.text((x0, 210 + k * 44), line, font=font(28), fill=MUTED)
        d.text((x0, 340), f"{entry.get('model', 'Kev-4B')}  第{gen}世代  テストのベストゲーム", font=font(24), fill=MUTED)
        stats = [("得点", f"{score:,}"), ("ライン", nlines), ("手数", pieces), ("テトリス", tet)]
        for k, (lab, val) in enumerate(stats):
            d.text((x0 + k * 190, 400), lab, font=font(22), fill=MUTED); d.text((x0 + k * 190, 430), str(val), font=font(40), fill=ACCENT)
        spp = spp_of(entry)
        if spp: d.text((x0, 510), f"1ミノあたり {spp:.1f} 点(テスト平均)", font=font(30), fill=col)
        if over and t > n - FPS * 2: d.text((x0, 570), "GAME OVER", font=font(40), fill=(244, 92, 110))
        progress(d, idx)
        yield img


def progress(d, idx):
    for k in range(len(SCENES)):
        x = 480 + k * 36
        d.ellipse([x, 670, x + 14, 684], fill=ACCENT if k <= idx else GRID)


def text_frames(secs, big, small, sub=None):
    for t in range(secs * FPS):
        img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
        a = min(1.0, t / (FPS * 0.6))
        c = tuple(int(BG[i] + (TEXT[i] - BG[i]) * a) for i in range(3))
        d.text((W // 2, 280), big, font=font(64), fill=c, anchor="mm")
        d.text((W // 2, 370), small, font=font(30), fill=tuple(int(BG[i] + (MUTED[i] - BG[i]) * a) for i in range(3)), anchor="mm")
        if sub: d.text((W // 2, 430), sub, font=font(24), fill=tuple(int(BG[i] + (ACCENT[i] - BG[i]) * a) for i in range(3)), anchor="mm")
        yield img


def chart_frames(secs):
    gens = sorted((x for x in generations.load() if spp_of(x)), key=lambda x: x["gen"])
    pts = [(x["gen"], spp_of(x)) for x in gens]
    gmax, ymax = max(p[0] for p in pts), 70
    X0, Y0, X1, Y1 = 120, 120, 1180, 600
    px = lambda g: X0 + (X1 - X0) * g / gmax
    py = lambda v: Y1 - (Y1 - Y0) * v / ymax
    n = secs * FPS
    for t in range(n):
        img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
        d.text((X0, 40), "1ミノあたりの得点(テスト平均)の推移", font=font(34), fill=TEXT)
        for v in (0, 20, 40, 60):
            d.line([X0, py(v), X1, py(v)], fill=GRID, width=1); d.text((X0 - 50, py(v) - 12), str(v), font=font(20), fill=MUTED)
        for label, g0, col in STAGES:
            d.text((px(g0), Y1 + 16), f"▲{label}", font=font(20), fill=col)
        shown = int(len(pts) * min(1.0, t / (n * 0.7))) + 1
        for (g1, v1), (g2, v2) in zip(pts[:shown], pts[1:shown]):
            d.line([px(g1), py(v1), px(g2), py(v2)], fill=stage_of(g2)[2], width=4)
        for g, v in pts[:shown]:
            d.ellipse([px(g) - 5, py(v) - 5, px(g) + 5, py(v) + 5], fill=stage_of(g)[2])
        if shown >= len(pts):
            g, v = max(pts, key=lambda p: p[1])
            d.text((px(g) - 160, py(v) - 50), f"最高 {v:.1f}(第{g}世代)", font=font(26), fill=TEXT)
        yield img


def main():
    out = ROOT / "runs" / "videos" / "kev-tetris-pv.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio_ffmpeg.write_frames(str(out), (W, H), fps=FPS, quality=8, macro_block_size=16)
    w.send(None)
    def put(frames):
        for f in frames: w.send(f.tobytes())
    put(text_frames(4, "Kev がテトリスを覚えるまで", "ローカルの決定モデル Kev × 強化学習・模倣学習", "Qwen3.5 + LoRA + ポインタヘッド / RTX 5060 Ti 16GB"))
    for i, sc in enumerate(SCENES): put(scene_frames(*sc, i))
    put(chart_frames(6))
    put(text_frames(4, "デシジョンモデルとして合格", "次は、指示ひとつで打ち方を変える", "X で学習の様子をライブ配信中"))
    w.close()
    print(out)


if __name__ == "__main__":
    main()
