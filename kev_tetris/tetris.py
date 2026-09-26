"""Tetris engine built around whole-piece placements.

Placements are every resting place the piece can reach from above by moving left/right, rotating (with a one-column
kick) and dropping one row at a time, so slides and tucks under overhangs are included. A placement reachable by a
plain hard drop keeps its short key r{rotation}x{column}; the others are keyed r{rotation}x{column}y{row}.

A decision model does not steer a falling piece key by key: every turn it gets the list of legal final placements of the
current piece and picks one. That is how most Tetris AIs are framed, and it maps directly onto a Kev Choice question.
"""
from __future__ import annotations

import random
from collections import deque
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
    slide: bool = False         # needs moves after the drop (a slide or tuck), not reachable by a plain hard drop

    @property
    def key(self) -> str:
        return f"r{self.rotation}x{self.x}y{self.y}" if self.slide else f"r{self.rotation}x{self.x}"


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
    enclosed: int = 0           # covered empty cells no piece can reach any more (true holes)
    overhang: int = 0           # covered empty cells still open to the side (a slide can fill them)
    new_enclosed: int = 0
    new_overhang: int = 0
    ready_rows: int = 0         # rows missing exactly one cell that is open from above (Tetris setup)


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


def cover_split(board, heights=None):
    """-> (enclosed, overhang): covered empty cells, split by whether open air reaches them through empty cells."""
    heights = heights or column_heights(board)
    open_ = set((x, y) for x in range(WIDTH) for y in range(HEIGHT - heights[x]))
    q = deque(open_)
    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and (nx, ny) not in open_ and not board[ny][nx]:
                open_.add((nx, ny)); q.append((nx, ny))
    covered = [(x, y) for x in range(WIDTH) for y in range(HEIGHT - heights[x], HEIGHT) if not board[y][x]]
    over = sum(1 for c in covered if c in open_)
    return len(covered) - over, over


def ready_rows(board, heights=None):
    """Rows with exactly one empty cell whose column is open above that row: one I piece away from clearing."""
    heights = heights or column_heights(board)
    n = 0
    for y in range(HEIGHT):
        empty = [x for x in range(WIDTH) if not board[y][x]]
        if len(empty) == 1 and HEIGHT - heights[empty[0]] > y: n += 1
    return n


def max_well_depth(heights):
    return max(max(0, min(heights[x - 1] if x > 0 else HEIGHT, heights[x + 1] if x < WIDTH - 1 else HEIGHT) - heights[x])
               for x in range(WIDTH))


def board_features(board):
    hs = column_heights(board)
    enclosed, overhang = cover_split(board, hs)
    return {"heights": hs, "holes": count_holes(board, hs), "max_height": max(hs), "agg_height": sum(hs),
            "bumpiness": sum(abs(hs[i] - hs[i + 1]) for i in range(WIDTH - 1)), "wells": wells_depth(hs),
            "enclosed": enclosed, "overhang": overhang, "ready_rows": ready_rows(board, hs), "max_well": max_well_depth(hs)}


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
        """Every final resting place reachable from above, entirely inside the board: first the plain hard drops, then
        the places only a slide, tuck or rotation after the drop reaches (breadth-first over rotation/column/row)."""
        piece, board = piece or self.current, board or self.board
        rots = ROTATIONS[piece]
        out, seen = [], set()
        starts = []
        for r, cells in enumerate(rots):
            w = max(x for x, _ in cells) + 1
            for ox in range(WIDTH - w + 1):
                oy = -4
                if not _fits(board, cells, ox, oy): continue
                starts.append((r, ox, oy))
                while _fits(board, cells, ox, oy + 1): oy += 1
                if oy < 0: continue      # would stick out of the top: not legal
                abs_cells = tuple(sorted((ox + x, oy + y) for x, y in cells))
                if abs_cells in seen: continue
                seen.add(abs_cells)
                out.append(Placement(piece, r, ox, oy, abs_cells))
        visited, q = set(starts), deque(starts)
        while q:
            r, x, y = q.popleft()
            cells = rots[r]
            if not _fits(board, cells, x, y + 1) and y >= 0:
                abs_cells = tuple(sorted((x + cx, y + cy) for cx, cy in cells))
                if abs_cells not in seen:
                    seen.add(abs_cells)
                    out.append(Placement(piece, r, x, y, abs_cells, slide=True))
            nxt = [(r, x - 1, y), (r, x + 1, y), (r, x, y + 1)]
            if len(rots) > 1:
                for r2 in ((r + 1) % len(rots), (r - 1) % len(rots)):
                    nxt += [(r2, x + dx, y) for dx in (0, -1, 1)]
            for st in nxt:
                if st not in visited and _fits(board, rots[st[0]], st[1], st[2]):
                    visited.add(st); q.append(st)
        return out

    def features(self, p: Placement) -> Features:
        if getattr(self, "_fb_board", None) is not self.board:   # the board's own features, once per turn
            self._fb_board, self._fb = self.board, board_features(self.board)
        b = self._fb
        nb, cleared = apply_placement(self.board, p, 1)
        f = board_features(nb)
        return Features(lines=cleared, holes=f["holes"], new_holes=f["holes"] - b["holes"], max_height=f["max_height"],
                        agg_height=f["agg_height"], bumpiness=f["bumpiness"], wells=f["wells"],
                        landing=HEIGHT - max(y for _, y in p.cells), max_well=f["max_well"],
                        enclosed=f["enclosed"], overhang=f["overhang"], new_enclosed=f["enclosed"] - b["enclosed"],
                        new_overhang=f["overhang"] - b["overhang"], ready_rows=f["ready_rows"])

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
