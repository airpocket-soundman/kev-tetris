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
RULES = 3   # 1: pieces enter from above any column, simplified rotation (generations 0-11); 2: SRS, spawn at the top;
            # 3: rules 2 with the Guideline's hidden rows above the field, block out / lock out, and 15 lock-delay resets
BUFFER = 4       # rules 3: hidden rows above the 20 visible ones (the Guideline spawns pieces there)
LOCK_RESETS = 15  # rules 3: moves/rotations allowed while on the ground; reaching a lower row gives them back


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


def _srs_spawn(piece, top=0):
    """(rotation state, box x, box y) where a piece appears: its top at row `top`, centred (left of centre)."""
    n = _SRS_SPAWN[piece][0]
    return 0, (WIDTH - n) // 2 if piece != "O" else 4, top - (1 if piece == "I" else 0)


def _srs_fits(board, piece, st):
    r, bx, by = st
    for cx, cy in SRS_SHAPES[piece][r]:
        x, y = bx + cx, by + cy
        if x < 0 or x >= WIDTH or y < 0 or y >= len(board) or board[y][x]: return False
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
    hs, H = [], len(board)
    for x in range(WIDTH):
        h = 0
        for y in range(H):
            if board[y][x]:
                h = H - y
                break
        hs.append(h)
    return hs


def count_holes(board, heights=None):
    heights = heights or column_heights(board)
    H = len(board)
    return sum(1 for x in range(WIDTH) for y in range(H - heights[x], H) if not board[y][x])


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
    H = len(board)
    open_ = set((x, y) for x in range(WIDTH) for y in range(H - heights[x]))
    q = deque(open_)
    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < WIDTH and 0 <= ny < H and (nx, ny) not in open_ and not board[ny][nx]:
                open_.add((nx, ny)); q.append((nx, ny))
    covered = [(x, y) for x in range(WIDTH) for y in range(H - heights[x], H) if not board[y][x]]
    over = sum(1 for c in covered if c in open_)
    return len(covered) - over, over


def ready_rows(board, heights=None):
    """Rows with exactly one empty cell whose column is open above that row: one I piece away from clearing."""
    heights = heights or column_heights(board)
    n = 0
    H = len(board)
    for y in range(H):
        empty = [x for x in range(WIDTH) if not board[y][x]]
        if len(empty) == 1 and H - heights[empty[0]] > y: n += 1
    return n


def extra_wells_cumulative(heights):
    """Dellacherie's cumulative wells (a well d deep counts 1+2+...+d) over every well but the deepest: a second deep
    well grows costly fast, so it gets filled instead of built around."""
    depths = []
    for x in range(WIDTH):
        d = min(heights[x - 1] if x > 0 else HEIGHT, heights[x + 1] if x < WIDTH - 1 else HEIGHT) - heights[x]
        if d > 0: depths.append(d)
    depths.sort()
    return sum(d * (d + 1) // 2 for d in depths[:-1])


def max_well_depth(heights):
    return max(max(0, min(heights[x - 1] if x > 0 else HEIGHT, heights[x + 1] if x < WIDTH - 1 else HEIGHT) - heights[x])
               for x in range(WIDTH))


def transitions(board):
    """Dellacherie's row and column transitions: filled/empty changes along each row (walls count as filled) and down
    each column (the floor counts as filled). Rugged, holey boards have many."""
    H = len(board)
    rows = sum(sum(1 for a, b in zip([1] + row, row + [1]) if bool(a) != bool(b)) for row in board if any(row))
    cols = 0
    for x in range(WIDTH):
        col = [board[y][x] for y in range(H)] + [1]
        cols += sum(1 for a, b in zip(col, col[1:]) if bool(a) != bool(b))
    return rows, cols


def hole_stats(board, heights=None):
    """BCTS: total hole depth (filled cells above each hole in its column) and the number of rows with a hole."""
    heights = heights or column_heights(board)
    H, depth, rows = len(board), 0, set()
    for x in range(WIDTH):
        above = 0
        for y in range(H - heights[x], H):
            if board[y][x]: above += 1
            else: depth += above; rows.add(y)
    return depth, len(rows)


def board_features(board):
    hs = column_heights(board)
    enclosed, overhang = cover_split(board, hs)
    row_t, col_t = transitions(board)
    hole_depth, hole_rows = hole_stats(board, hs)
    return {"heights": hs, "holes": count_holes(board, hs), "max_height": max(hs), "agg_height": sum(hs),
            "bumpiness": sum(abs(hs[i] - hs[i + 1]) for i in range(WIDTH - 1)), "wells": wells_depth(hs),
            "enclosed": enclosed, "overhang": overhang, "ready_rows": ready_rows(board, hs), "max_well": max_well_depth(hs),
            "row_transitions": row_t, "col_transitions": col_t, "hole_depth": hole_depth, "hole_rows": hole_rows,
            "extra_wells": extra_wells_cumulative(hs)}


def is_tspin(board, p) -> bool:
    """A T placed by rotating in (not a plain drop) with at least 3 of the 4 corners of its 3x3 box filled (walls and
    floor count) - the usual 3-corner rule, checked on the board before the lines clear."""
    if p.piece != "T" or not p.slide: return False
    xs = [x for x, _ in p.cells]; ys = [y for _, y in p.cells]
    cx = sorted(xs)[1]; cy = sorted(ys)[1]                          # the T's centre is its median cell
    corners = [(cx - 1, cy - 1), (cx + 1, cy - 1), (cx - 1, cy + 1), (cx + 1, cy + 1)]
    filled = sum(1 for x, y in corners if x < 0 or x >= WIDTH or y >= len(board) or (y >= 0 and board[y][x]))
    return filled >= 3


def _fits(board, cells, ox, oy):
    for cx, cy in cells:
        x, y = ox + cx, oy + cy
        if x < 0 or x >= WIDTH or y >= len(board): return False
        if y >= 0 and board[y][x]: return False
    return True


def apply_placement(board, p: Placement, color: int):
    """-> (new board, lines cleared). The input board is not modified."""
    b = [row[:] for row in board]
    for x, y in p.cells: b[y][x] = color
    kept = [row for row in b if not all(row)]
    cleared = len(board) - len(kept)
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
    b2b: int = 0                # consecutive Tetrises / T-spin clears (back-to-back chain), 0 after any other clear
    last_tspin: bool = False    # the last placement was a T-spin

    def __post_init__(self):
        if self.rules >= 3 and len(self.board) == HEIGHT:   # hidden rows above the visible field
            self.board = [[0] * WIDTH for _ in range(BUFFER)] + self.board
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
        if self.rules >= 3: return self._placements_guideline(piece, board)
        return self._placements_srs(piece, board) if self.rules >= 2 else self._placements_v1(piece, board)

    @property
    def hidden(self) -> int:
        """Rows above the visible field at the top of self.board."""
        return len(self.board) - HEIGHT

    def _placements_guideline(self, piece, board) -> list[Placement]:
        """Rules 3 (Guideline). The piece spawns in the hidden rows right above the field and drops one row at once if it
        can; overlapping there is a block out (no placement). Moves and rotations are free in the air; on the ground
        each one uses a lock-delay reset (15), and reaching a new lowest row gives them all back. Every place the piece
        can lock is a placement; the ones reached by rotating/shifting at the spawn and dropping straight are plain."""
        top = len(board) - HEIGHT - 2
        spawn = _srs_spawn(piece, top)
        if not _srs_fits(board, piece, spawn): return []
        if _srs_fits(board, piece, (spawn[0], spawn[1], spawn[2] + 1)): spawn = (spawn[0], spawn[1], spawn[2] + 1)
        fits = lambda st: _srs_fits(board, piece, st)
        grounded = lambda st: not fits((st[0], st[1], st[2] + 1))
        cells_of = lambda st: tuple(sorted((st[1] + cx, st[2] + cy) for cx, cy in SRS_SHAPES[piece][st[0]]))
        out, seen = [], set()
        # plain drops: shift/rotate at the spawn row while airborne, then straight down
        plain, q = {spawn}, deque([spawn])
        while q:
            st = q.popleft()
            if grounded(st): continue
            for nxt in _srs_moves(board, piece, st, down=False):
                if nxt[2] == spawn[2] and nxt not in plain: plain.add(nxt); q.append(nxt)
        for r, bx, by in sorted(plain):
            while fits((r, bx, by + 1)): by += 1
            c = cells_of((r, bx, by))
            if c not in seen: seen.add(c); out.append(_as_placement(piece, c, slide=False))
        # everything else: (state, lowest row reached) -> fewest resets used
        best = {(spawn, spawn[2]): 0}
        q = deque([(spawn, spawn[2], 0)])
        while q:
            st, low, used = q.popleft()
            if best.get((st, low), 99) < used: continue
            on_ground = grounded(st)
            if on_ground:
                c = cells_of(st)
                if c not in seen: seen.add(c); out.append(_as_placement(piece, c, slide=True))
            nexts = [(st[0], st[1], st[2] + 1)] if not on_ground else []
            if not (on_ground and used >= LOCK_RESETS):
                nexts += [n for n in _srs_moves(board, piece, st, down=False)]
            for nxt in nexts:
                if not fits(nxt): continue
                is_drop = nxt[1:] == (st[1], st[2] + 1) and nxt[0] == st[0]
                n_used = used + (1 if on_ground and not is_drop else 0)
                n_low = low
                if nxt[2] > low: n_low, n_used = nxt[2], 0          # a new lowest row: the resets come back
                key = (nxt, n_low)
                if best.get(key, 99) > n_used:
                    best[key] = n_used; q.append((nxt, n_low, n_used))
        return out

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
                        landing=len(self.board) - max(y for _, y in p.cells), max_well=f["max_well"],
                        enclosed=f["enclosed"], overhang=f["overhang"], new_enclosed=f["enclosed"] - b["enclosed"],
                        new_overhang=f["overhang"] - b["overhang"], ready_rows=f["ready_rows"])

    def step(self, p: Placement) -> int:
        """Place the current piece. -> lines cleared. Ends the game when the next piece has no legal placement."""
        assert not self.over and p.piece == self.current
        self.last_tspin = is_tspin(self.board, p)          # judged on the board before the lines clear
        self.board, cleared = apply_placement(self.board, p, PIECES.index(p.piece) + 1)
        self.pieces += 1
        self.lines += cleared
        self.tetrises += cleared == 4
        if cleared:   # back-to-back: a Tetris or a T-spin clear right after another one, singles/doubles break it
            difficult = cleared == 4 or self.last_tspin
            self.b2b = self.b2b + 1 if difficult else 0
        self.score += LINE_SCORE[cleared]
        self.current, self.next = self.next, self._draw()
        if self.rules >= 3 and not cleared and all(y < self.hidden for _, y in p.cells):
            self.over = True                                 # lock out: locked entirely above the visible field
        elif not self.placements(): self.over = True       # block out: the next piece cannot appear
        return cleared

    def visible(self, board=None) -> list:
        """The 20 visible rows of a board (rules 3 keeps hidden rows above them)."""
        board = self.board if board is None else board
        return [row[:] for row in board[len(board) - HEIGHT:]]

    def snapshot(self) -> dict:
        return {"board": self.visible(), "current": self.current, "next": self.next, "score": self.score,
                "lines": self.lines, "pieces": self.pieces, "tetrises": self.tetrises, "over": self.over}
