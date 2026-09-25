"""Tetris engine built around whole-piece placements (rotation + column, then hard drop).

A decision model does not steer a falling piece key by key: every turn it gets the list of legal final placements of the
current piece and picks one. That is how most Tetris AIs are framed, and it maps directly onto a Kev Choice question.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

WIDTH, HEIGHT = 10, 20
PIECES = "IOTSZJL"

# cells (x, y) of each rotation, y grows downward; normalised so min x = min y = 0
_BASE = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
}
LINE_SCORE = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}


def _normalise(cells):
    mx, my = min(x for x, _ in cells), min(y for _, y in cells)
    return tuple(sorted((x - mx, y - my) for x, y in cells))


def _rotations(cells):
    out, cur = [], _normalise(cells)
    for _ in range(4):
        if cur not in out: out.append(cur)
        cur = _normalise([(-y, x) for x, y in cur])   # 90 degrees clockwise
    return out


ROTATIONS = {p: _rotations(c) for p, c in _BASE.items()}


@dataclass(frozen=True)
class Placement:
    piece: str
    rotation: int
    x: int                      # left column of the piece
    y: int                      # top row of the piece after the drop
    cells: tuple                # absolute (x, y) cells

    @property
    def key(self) -> str:
        return f"r{self.rotation}x{self.x}"


@dataclass
class Features:
    lines: int                  # lines this placement clears
    holes: int                  # empty cells with a filled cell somewhere above, after the placement
    new_holes: int
    max_height: int
    agg_height: int
    bumpiness: int              # sum of |height difference| between neighbouring columns
    wells: int                  # total depth of wells (columns lower than both neighbours)
    landing: int                # height of the piece's lowest cell above the floor
    max_well: int = 0           # the deepest well: a column kept open this deep is ready for a Tetris


def column_heights(board):
    hs = []
    for x in range(WIDTH):
        h = 0
        for y in range(HEIGHT):
            if board[y][x]:
                h = HEIGHT - y
                break
        hs.append(h)
    return hs


def count_holes(board, heights=None):
    heights = heights or column_heights(board)
    return sum(1 for x in range(WIDTH) for y in range(HEIGHT - heights[x], HEIGHT) if not board[y][x])


def wells_depth(heights):
    total = 0
    for x in range(WIDTH):
        left = heights[x - 1] if x > 0 else HEIGHT
        right = heights[x + 1] if x < WIDTH - 1 else HEIGHT
        d = min(left, right) - heights[x]
        if d > 0: total += d
    return total


def max_well_depth(heights):
    return max(max(0, min(heights[x - 1] if x > 0 else HEIGHT, heights[x + 1] if x < WIDTH - 1 else HEIGHT) - heights[x])
               for x in range(WIDTH))


def board_features(board):
    hs = column_heights(board)
    return {"heights": hs, "holes": count_holes(board, hs), "max_height": max(hs), "agg_height": sum(hs),
            "bumpiness": sum(abs(hs[i] - hs[i + 1]) for i in range(WIDTH - 1)), "wells": wells_depth(hs)}


def _fits(board, cells, ox, oy):
    for cx, cy in cells:
        x, y = ox + cx, oy + cy
        if x < 0 or x >= WIDTH or y >= HEIGHT: return False
        if y >= 0 and board[y][x]: return False
    return True


def apply_placement(board, p: Placement, color: int):
    """-> (new board, lines cleared). The input board is not modified."""
    b = [row[:] for row in board]
    for x, y in p.cells: b[y][x] = color
    kept = [row for row in b if not all(row)]
    cleared = HEIGHT - len(kept)
    return [[0] * WIDTH for _ in range(cleared)] + kept, cleared


@dataclass
class Game:
    seed: int | None = None
    board: list = field(default_factory=lambda: [[0] * WIDTH for _ in range(HEIGHT)])
    score: int = 0
    lines: int = 0
    pieces: int = 0
    tetrises: int = 0
    over: bool = False

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.bag: list[str] = []
        self.current = self._draw()
        self.next = self._draw()

    def _draw(self) -> str:
        if not self.bag:
            self.bag = list(PIECES)
            self.rng.shuffle(self.bag)
        return self.bag.pop()

    def placements(self, piece: str | None = None, board=None) -> list[Placement]:
        """Every final resting place reachable by rotating at the top and hard-dropping, entirely inside the board."""
        piece, board = piece or self.current, board or self.board
        out = []
        for r, cells in enumerate(ROTATIONS[piece]):
            w = max(x for x, _ in cells) + 1
            for ox in range(WIDTH - w + 1):
                oy = -4
                if not _fits(board, cells, ox, oy): continue
                while _fits(board, cells, ox, oy + 1): oy += 1
                if oy < 0: continue      # would stick out of the top: not legal
                out.append(Placement(piece, r, ox, oy, tuple((ox + x, oy + y) for x, y in cells)))
        return out

    def features(self, p: Placement) -> Features:
        before = count_holes(self.board)
        nb, cleared = apply_placement(self.board, p, 1)
        f = board_features(nb)
        return Features(lines=cleared, holes=f["holes"], new_holes=f["holes"] - before, max_height=f["max_height"],
                        agg_height=f["agg_height"], bumpiness=f["bumpiness"], wells=f["wells"],
                        landing=HEIGHT - max(y for _, y in p.cells), max_well=max_well_depth(f["heights"]))

    def step(self, p: Placement) -> int:
        """Place the current piece. -> lines cleared. Ends the game when the next piece has no legal placement."""
        assert not self.over and p.piece == self.current
        self.board, cleared = apply_placement(self.board, p, PIECES.index(p.piece) + 1)
        self.pieces += 1
        self.lines += cleared
        self.tetrises += cleared == 4
        self.score += LINE_SCORE[cleared]
        self.current, self.next = self.next, self._draw()
        if not self.placements(): self.over = True
        return cleared

    def snapshot(self) -> dict:
        return {"board": [row[:] for row in self.board], "current": self.current, "next": self.next, "score": self.score,
                "lines": self.lines, "pieces": self.pieces, "tetrises": self.tetrises, "over": self.over}
