"""Exhaustive check of the engine's SRS rotations against an independent reference.

The engine rotates shapes inside their bounding box and keeps the wall-kick tables. The reference below follows the
Guideline definition instead: "true rotation" of the minos about a pivot mino, then up to five tests whose kicks are
the difference of two offset tables (offset[from] - offset[to]). Both must give the same cells, or both must fail,
for every piece, orientation, position and direction, on an empty board and on random boards that force kicks.

    python tests/test_srs.py
"""
import random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev_tetris.tetris import HEIGHT, WIDTH, SRS_SHAPES, _srs_fits, srs_rotate   # noqa: E402

# minos relative to the pivot in the spawn state, y UP (Guideline SRS)
PIVOT_MINOS = {
    "T": [(-1, 0), (0, 0), (1, 0), (0, 1)],
    "J": [(-1, 1), (-1, 0), (0, 0), (1, 0)],
    "L": [(1, 1), (-1, 0), (0, 0), (1, 0)],
    "S": [(0, 1), (1, 1), (-1, 0), (0, 0)],
    "Z": [(-1, 1), (0, 1), (0, 0), (1, 0)],
    "I": [(-1, 0), (0, 0), (1, 0), (2, 0)],
}
# offset data per state 0, R, 2, L (y UP); kick for a test = offset[from] - offset[to]
OFFSETS_JLSTZ = [[(0, 0)] * 5,
                 [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
                 [(0, 0)] * 5,
                 [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)]]
OFFSETS_I = [[(0, 0), (-1, 0), (2, 0), (-1, 0), (2, 0)],
             [(-1, 0), (0, 0), (0, 0), (0, 1), (0, -2)],
             [(-1, 1), (1, 1), (-2, 1), (1, 0), (-2, 0)],
             [(0, 1), (0, 1), (0, 1), (0, -1), (0, 2)]]


def ref_cells(piece, state, px, py):
    """Absolute cells (y DOWN, like the engine) of a piece in `state` with its pivot at (px, py)."""
    out = []
    for x, y in PIVOT_MINOS[piece]:
        for _ in range(state): x, y = y, -x             # true rotation, clockwise, y up
        out.append((px + x, py - y))
    return frozenset(out)


def ref_fits(board, cells):
    return all(0 <= x < WIDTH and 0 <= y < HEIGHT and not board[y][x] for x, y in cells)


def ref_rotate(board, piece, state, px, py, turn):
    to = (state + turn) % 4
    offs = OFFSETS_I if piece == "I" else OFFSETS_JLSTZ
    for (ax, ay), (bx, by) in zip(offs[state], offs[to]):
        nx, ny = px + ax - bx, py - (ay - by)            # offsets are y up
        cells = ref_cells(piece, to, nx, ny)
        if ref_fits(board, cells): return cells
    return None


def engine_cells(piece, st):
    r, bx, by = st
    return frozenset((bx + x, by + y) for x, y in SRS_SHAPES[piece][r])


def pivot_for(piece, state, cells):
    """The reference pivot that puts `piece` in `state` on exactly these cells."""
    xs = [x for x, _ in cells]; ys = [y for _, y in cells]
    for px in range(min(xs) - 3, max(xs) + 4):
        for py in range(min(ys) - 3, max(ys) + 4):
            if ref_cells(piece, state, px, py) == cells: return px, py
    raise AssertionError(f"{piece} state {state}: no pivot for {sorted(cells)}")


def check(board, stats):
    for piece in PIVOT_MINOS:
        for r in range(4):
            for bx in range(-4, WIDTH + 1):
                for by in range(-4, HEIGHT + 1):
                    st = (r, bx, by)
                    if not _srs_fits(board, piece, st): continue
                    px, py = pivot_for(piece, r, engine_cells(piece, st))
                    for turn in (1, -1):
                        got = srs_rotate(board, piece, st, turn)
                        got = engine_cells(piece, got) if got else None
                        want = ref_rotate(board, piece, r, px, py, turn)
                        stats["cases"] += 1
                        stats["kicked"] += want is not None and want != ref_cells(piece, (r + turn) % 4, px, py)
                        stats["blocked"] += want is None
                        if got != want:
                            stats["fail"].append((piece, r, bx, by, turn, sorted(got or []), sorted(want or [])))


def random_board(rng):
    b = [[0] * WIDTH for _ in range(HEIGHT)]
    top = rng.randint(4, HEIGHT - 2)
    density = rng.uniform(0.3, 0.75)
    for y in range(top, HEIGHT):
        for x in range(WIDTH):
            if rng.random() < density: b[y][x] = 1
    return b


def main():
    stats = {"cases": 0, "kicked": 0, "blocked": 0, "fail": []}
    check([[0] * WIDTH for _ in range(HEIGHT)], stats)
    rng = random.Random(0)
    for _ in range(200): check(random_board(rng), stats)
    print(f"cases {stats['cases']}, needed a kick {stats['kicked']}, no rotation possible {stats['blocked']}, "
          f"mismatches {len(stats['fail'])}")
    for f in stats["fail"][:10]: print("MISMATCH", f)
    # O never moves when rotated
    assert srs_rotate([[0] * WIDTH for _ in range(HEIGHT)], "O", (0, 4, 5), 1) is None
    return 1 if stats["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
