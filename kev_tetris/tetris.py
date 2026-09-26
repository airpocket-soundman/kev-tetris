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
RULES = 2   # 1: pieces enter from above any column, simplified rotation (generations 0-11); 2: SRS, spawn at the top


# --- rules 2: the standard rotation system (SRS) and a spawn at the top ------------------------------------------------
# Shapes in their SRS bounding boxes (y down), spawn orientation first; rotating turns the box clockwise.
_SRS_SPAWN = {
    "I": (4, [(0, 1), (1, 1), (2, 1), (3, 1)]),
    "O": (2, [(0, 0), (1, 0), (0, 1), (1, 1)]),
    "T": (3, [(1, 0), (0, 1), (1, 1), (2, 1)]),
    "S": (3, [(1, 0), (2, 0), (0, 1), (1, 1)]),
    "Z": (3, [(0, 0), (1, 0), (1, 1), (2, 1)]),
    "J": (3, [(0, 0), (0, 1), (1, 1), (2, 1)]),
    "L": (3, [(2, 0), (0, 1), (1, 1), (2, 1)]),
}


def _srs_states(n, cells):
    out, cur = [], list(cells)
    for _ in range(4):
        out.append(tuple(cur))
        cur = [(n - 1 - y, x) for x, y in cur]
    return out


SRS_SHAPES = {p: _srs_states(n, c) for p, (n, c) in _SRS_SPAWN.items()}
# wall kicks (x right, y UP as in the SRS tables) per (from, to) state: 0 spawn, 1 R, 2 two, 3 L
_KICKS_JLSTZ = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)], (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)], (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)], (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)], (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
}
_KICKS_I = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)], (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)], (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)], (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)], (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
}


def _srs_spawn(piece):
    """(rotation state, box x, box y) where a piece appears: top row, centred."""
    n = _SRS_SPAWN[piece][0]
    return 0, (WIDTH - n) // 2 if piece != "O" else 4, -1 if piece == "I" else 0


def _srs_fits(board, piece, st):
    r, bx, by = st
    for cx, cy in SRS_SHAPES[piece][r]:
        x, y = bx + cx, by + cy
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT or board[y][x]: return False
    return True


def _srs_moves(board, piece, st, down=True):
    """States one move away: left, right, (soft drop), and both rotations with the SRS kicks."""
    r, bx, by = st
    out = [(r, bx - 1, by), (r, bx + 1, by)] + ([(r, bx, by + 1)] if down else [])
    for turn in (1, -1):
        st2 = srs_rotate(board, piece, st, turn)
        if st2: out.append(st2)
    return [s for s in out if _srs_fits(board, piece, s)]


def srs_rotate(board, piece, st, turn):
    """One rotation (turn 1 = clockwise, -1 = counter-clockwise) with the SRS kicks. -> the new state, or None."""
    if piece == "O": return None                         # O keeps its cells: rotating it changes nothing
    r, bx, by = st
    r2 = (r + turn) % 4
    kicks = _KICKS_I if piece == "I" else _KICKS_JLSTZ
    for dx, dy in kicks[(r, r2)]:
        cand = (r2, bx + dx, by - dy)                    # the tables count y upward
        if _srs_fits(board, piece, cand): return cand      # the first kick that fits is the rotation
    return None


def _as_placement(piece, cells, slide):
    """A resting place as a Placement; rotation and x name the shape the way rules 1 did (so keys stay comparable)."""
    shape = _normalise(cells)
    return Placement(piece, ROTATIONS[piece].index(shape), min(x for x, _ in cells), min(y for _, y in cells),
                     tuple(sorted(cells)), slide=slide)


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
    rules: int = RULES          # recordings made under rules 1 replay with Game(rules=1)
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
        piece, board = piece or self.current, board or self.board
        return self._placements_srs(piece, board) if self.rules >= 2 else self._placements_v1(piece, board)

    def _placements_srs(self, piece, board) -> list[Placement]:
        """Rules 2. The piece spawns at the top centre; if it overlaps the stack there, there is no placement (game over).
        Plain drops: rotate and shift at the spawn height, then drop straight. Then everything else reachable by
        moving, rotating (SRS kicks, also after landing) and soft-dropping: slides, tucks and spins."""
        spawn = _srs_spawn(piece)
        if not _srs_fits(board, piece, spawn): return []
        cells_of = lambda st: [(st[1] + cx, st[2] + cy) for cx, cy in SRS_SHAPES[piece][st[0]]]
        top, q = {spawn}, deque([spawn])
        while q:                                           # moves before any drop
            for st in _srs_moves(board, piece, q.popleft(), down=False):
                if st not in top: top.add(st); q.append(st)
        out, seen = [], set()
        for st in sorted(top):
            r, bx, by = st
            while _srs_fits(board, piece, (r, bx, by + 1)): by += 1
            cells = tuple(sorted(cells_of((r, bx, by))))
            if cells not in seen: seen.add(cells); out.append(_as_placement(piece, cells, slide=False))
        visited, q = set(top), deque(top)
        while q:                                           # everything reachable with drops, moves and rotations
            st = q.popleft()
            if not _srs_fits(board, piece, (st[0], st[1], st[2] + 1)):
                cells = tuple(sorted(cells_of(st)))
                if cells not in seen: seen.add(cells); out.append(_as_placement(piece, cells, slide=True))
            for nxt in _srs_moves(board, piece, st):
                if nxt not in visited: visited.add(nxt); q.append(nxt)
        return out

    def _placements_v1(self, piece, board) -> list[Placement]:
        """Rules 1 (generations 0-11): pieces enter from above any column; simplified rotation with a one-column kick."""
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
