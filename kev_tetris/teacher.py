"""A two-piece lookahead search used as a teacher (expert iteration / DAgger).

For a position met in practice, every placement of the current piece is scored by the best follow-up placement of the
next piece: shaped reward of both moves + the potential of the board after them. Only the current and the next piece
are used - what Kev itself sees. Kev keeps choosing the moves played in practice; the search's choice becomes the
training label for that position, so Kev learns to see two moves ahead on the boards it actually reaches.

To stay cheap: the first ply keeps the top `first_k` placements by a one-move score, and the second ply only looks at
plain drops (shift/rotate at the spawn, then straight down).
"""
from __future__ import annotations

from .tetris import SRS_SHAPES, Game, Placement, _as_placement, _srs_fits, _srs_moves, _srs_spawn, apply_placement, board_features
from collections import deque


def plain_placements(board, piece) -> list[Placement]:
    """Rules-3 plain drops on `board` (hidden rows included): the cheap second-ply candidates. [] = block out."""
    top = len(board) - 20 - 2
    spawn = _srs_spawn(piece, top)
    if not _srs_fits(board, piece, spawn): return []
    if _srs_fits(board, piece, (spawn[0], spawn[1], spawn[2] + 1)): spawn = (spawn[0], spawn[1], spawn[2] + 1)
    seen_states, q = {spawn}, deque([spawn])
    while q:
        st = q.popleft()
        if not _srs_fits(board, piece, (st[0], st[1], st[2] + 1)): continue
        for nxt in _srs_moves(board, piece, st, down=False):
            if nxt[2] == spawn[2] and nxt not in seen_states: seen_states.add(nxt); q.append(nxt)
    out, seen = [], set()
    for r, bx, by in seen_states:
        while _srs_fits(board, piece, (r, bx, by + 1)): by += 1
        cells = tuple(sorted((bx + cx, by + cy) for cx, cy in SRS_SHAPES[piece][r]))
        if cells not in seen: seen.add(cells); out.append(_as_placement(piece, cells, slide=False))
    return out


def search_label(game: Game, placements: list[Placement], reward, potential, first_k: int = 8) -> str:
    """-> the key of the placement with the best two-move value. reward(before, after, cleared, died) and
    potential(features) are the training loop's own (so the teacher optimises what the loop rewards)."""
    before = board_features(game.board)
    first = []
    for p in placements:
        b1, c1 = apply_placement(game.board, p, 1)
        f1 = board_features(b1)
        r1 = reward(before, f1, c1, False)
        first.append((r1 + potential(f1), p, b1, f1, r1))
    first.sort(key=lambda t: t[0], reverse=True)
    best_key, best_val = first[0][1].key, float("-inf")
    for _, p, b1, f1, r1 in first[:first_k]:
        seconds = plain_placements(b1, game.next)
        if not seconds:                                  # the next piece could not even appear: a loss
            val = r1 + reward(before, f1, 0, True)
        else:
            val = float("-inf")
            for q2 in seconds:
                b2, c2 = apply_placement(b1, q2, 1)
                f2 = board_features(b2)
                val = max(val, r1 + reward(f1, f2, c2, False) + potential(f2))
        if val > best_val: best_val, best_key = val, p.key
    return best_key
