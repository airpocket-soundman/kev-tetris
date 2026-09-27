"""Render a recorded test game to an MP4 (for sharing a generation's best play).

    python -m kev_tetris.video --gen 46            # its best-scoring test game -> runs/videos/gen-046-best.mp4

Each move is drawn twice: the piece where it lands (outlined), then the board after it (cleared rows flash white).
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

from . import generations
from .tetris import HEIGHT, WIDTH, Game

ROOT = Path(__file__).resolve().parent.parent
CELL, W, H = 36, 1280, 800
BG, PANEL, GRID, TEXT, MUTED, ACCENT = (11, 14, 26), (20, 25, 44), (28, 34, 58), (232, 236, 248), (142, 152, 196), (94, 231, 255)
COLORS = {"I": (64, 220, 240), "O": (250, 214, 72), "T": (190, 110, 240), "S": (96, 224, 120), "Z": (244, 92, 110),
          "J": (80, 130, 245), "L": (250, 160, 64)}
PIECE_OF = {i + 1: p for i, p in enumerate("IOTSZJL")}
FONT = "C:/Windows/Fonts/YuGothB.ttc"


def font(size):
    try: return ImageFont.truetype(FONT, size)
    except OSError: return ImageFont.load_default()


def draw_frame(board, piece_cells, piece, stats, title, cleared_rows=()):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    ox, oy = 560, (H - HEIGHT * CELL) // 2
    d.rectangle([ox - 6, oy - 6, ox + WIDTH * CELL + 6, oy + HEIGHT * CELL + 6], fill=PANEL)
    top = len(board) - HEIGHT
    for y in range(HEIGHT):
        for x in range(WIDTH):
            c = board[top + y][x]
            rect = [ox + x * CELL + 1, oy + y * CELL + 1, ox + (x + 1) * CELL - 1, oy + (y + 1) * CELL - 1]
            if c: d.rectangle(rect, fill=COLORS.get(PIECE_OF.get(c, "T"), (180, 180, 200)))
            else: d.rectangle(rect, outline=GRID)
    for y in cleared_rows:
        if y >= top: d.rectangle([ox, oy + (y - top) * CELL, ox + WIDTH * CELL, oy + (y - top + 1) * CELL], fill=(245, 245, 255))
    for x, y in piece_cells:
        if y < top: continue
        rect = [ox + x * CELL + 1, oy + (y - top) * CELL + 1, ox + (x + 1) * CELL - 1, oy + (y - top + 1) * CELL - 1]
        d.rectangle(rect, fill=COLORS[piece], outline=(255, 255, 255), width=3)
    for k, line in enumerate(title.split("\n")): d.text((40, 30 + k * 40), line, font=font(30), fill=TEXT)
    y = 130
    for label, value in stats:
        d.text((40, y), label, font=font(22), fill=MUTED); d.text((40, y + 28), str(value), font=font(38), fill=ACCENT)
        y += 90
    d.text((ox + WIDTH * CELL + 40, H - 44), "Kev plays Tetris", font=font(20), fill=MUTED)
    return img


def render(gen: int, game_index: int | None, out: Path, fps: int = 10):
    rec = json.loads((ROOT / "runs" / "replays" / f"gen-{gen:03d}.json").read_text(encoding="utf-8"))
    games = rec["games"] if isinstance(rec, dict) else rec
    i = game_index if game_index is not None else max(range(len(games)), key=lambda k: games[k]["score"])
    rg = games[i]
    entry = next((x for x in generations.load() if x["gen"] == gen), {})
    title = f"{entry.get('model', 'Kev')}  第{gen}世代\nテスト{i + 1}(ベストプレイ)"
    g = Game(seed=rg["seed"], rules=rg.get("rules", 3))
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(out), (W, H), fps=fps, quality=8, macro_block_size=16)
    writer.send(None)
    tetrises = 0
    def stats():
        return [("得点", f"{g.score:,}"), ("ライン", g.lines), ("手数", g.pieces), ("テトリス", tetrises),
                ("1ミノあたり", f"{g.score / max(1, g.pieces):.1f}"), ("NEXT", g.next)]
    for m in rg["moves"]:
        ps = {p.key: p for p in g.placements()}
        p, piece = ps[m["key"]], g.current
        pre = [row[:] for row in g.board]
        writer.send(draw_frame(pre, p.cells, piece, stats(), title).tobytes())
        full = [y for y in range(len(pre)) if all(pre[y][x] or (x, y) in p.cells for x in range(WIDTH))]
        if full:
            writer.send(draw_frame(pre, p.cells, piece, stats(), title, full).tobytes())
        n = g.step(p)
        tetrises += n == 4
        writer.send(draw_frame(g.board, [], piece, stats(), title).tobytes())
        if g.over: break
    for _ in range(fps * 3): writer.send(draw_frame(g.board, [], "I", stats(), title).tobytes())
    writer.close()
    return i, rg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render a recorded test game to MP4")
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--game", type=int, default=None, help="test game index (default: the best score)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else ROOT / "runs" / "videos" / f"gen-{a.gen:03d}-best.mp4"
    i, rg = render(a.gen, a.game, out, a.fps)
    print(f"{out}  (test game {i + 1}: {rg['score']} points, {rg['lines']} lines, {rg['pieces']} pieces)")


if __name__ == "__main__":
    main()
