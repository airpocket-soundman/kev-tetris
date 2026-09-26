"""The Kev interface: a Tetris turn as one System One request.

state     the board drawn as text (only the rows that matter), the current and next piece, column heights and holes.
question  `move`, a Choice whose options are the legal placements of the current piece. Each option carries a short
          description of what the placement leads to (lines, holes, height, bumpiness), so the model decides from
          consequences it can read instead of simulating the drop itself.

The same shape is used to write training records (`to_record` with a label), so what Kev is trained on is exactly
what it is asked at play time.
"""
from __future__ import annotations

from .tetris import HEIGHT, WIDTH, Game, Placement, board_features, is_tspin

INSTRUCTIONS = ("You are playing Tetris for a high score. Choose where to place the current piece. A Tetris (4 lines "
                "at once) scores far more than single lines: stack flat, keep one column open as a deep well, and fill it "
                "with an I piece. Keep exactly one well: other deep gaps are dangerous. Keep the stack at about half the "
                "board height or lower. Enclosed holes are very bad; overhangs can still be filled by sliding a piece under "
                "them. When a hole or an overhang appears, repair it at once, then go back to building for a Tetris. "
                "A stack reaching the top loses the game.")


def board_text(board, margin: int = 2) -> str:
    H = len(board)                          # rules 3: hidden rows above the 20 visible ones
    top = next((y for y in range(H) if any(board[y])), H)
    start = max(0, top - margin)
    rows = [f"{H - y:2d} |" + "".join("#" if c else "." for c in board[y]) + "|" for y in range(start, H)]
    head = f"(rows {H} to {H - start + 1} are empty)" if start > 0 else ""
    cols = "    " + "".join(str(x) for x in range(WIDTH))
    return "\n".join(filter(None, [head, *rows, cols]))


VALUE_LEVELS = ["very bad: likely to lose soon", "bad", "fair", "good", "excellent: safe and scoring well"]
VALUE_QUESTION = {"type": "score", "instructions": "How good is this board for the rest of the game, counting future "
                  "points and the risk of losing?", "criteria": VALUE_LEVELS}


def state_text(game: Game, next_known: bool = True) -> str:
    """next_known=False: the board after a move, before its next piece is drawn (Kev's own lookahead): the piece that
    comes after the current one is not known yet, and the text says so."""
    f = board_features(game.board)
    hidden = len(game.board) - HEIGHT
    size = f"{WIDTH} columns x {HEIGHT} rows" + (f" plus {hidden} hidden rows above (pieces appear there)" if hidden else "")
    return (f"Tetris board, {size}, '#' filled, '.' empty, row 1 is the floor.\n"
            f"{board_text(game.board)}\n"
            f"Current piece: {game.current}. Next piece: {game.next if next_known else 'unknown'}.\n"
            f"Column heights: {' '.join(map(str, f['heights']))}. Holes: {f['holes']}. "
            f"Lines cleared so far: {game.lines}.")


def option_text(game: Game, p: Placement) -> str:
    f = game.features(p)
    cols = sorted({x for x, _ in p.cells})
    span = f"col {cols[0]}" if len(cols) == 1 else f"cols {cols[0]}-{cols[-1]}"
    how = (", slide" if p.slide else "") + (", T-spin" if is_tspin(game.board, p) else "")
    return (f"rot {p.rotation}, {span}{how}: clears {f.lines}, holes {f.enclosed} ({f.new_enclosed:+d}), "
            f"overhangs {f.overhang} ({f.new_overhang:+d}), height {f.max_height}, bumps {f.bumpiness}, "
            f"well {f.max_well}, ready {f.ready_rows}")


def to_request(game: Game, placements: list[Placement] | None = None, model: str = "kev-latest", value: bool = False) -> dict:
    """The move question; value=True adds "how good is this board" (a Score question) to the same pass."""
    placements = placements if placements is not None else game.placements()
    qs = {"move": {"type": "choice", "instructions": INSTRUCTIONS,
                   "criteria": {p.key: option_text(game, p) for p in placements}}}
    if value: qs["value"] = dict(VALUE_QUESTION)
    return {"state": state_text(game), "model": model, "questions": qs}


def to_value_request(game: Game, model: str = "kev-latest") -> dict:
    """Only the value question, on a board after a move (its current piece is the known next one; after that: unknown)."""
    return {"state": state_text(game, next_known=False), "model": model, "questions": {"value": dict(VALUE_QUESTION)}}


def to_record(game: Game, placements: list[Placement], label: str, value_label: int | None = None) -> dict:
    """A labelled training record in kev.data.load_records' format (value_label: the board's value level, 0-4)."""
    req = to_request(game, placements, value=value_label is not None)
    req.pop("model")
    req["questions"]["move"]["label"] = label
    if value_label is not None: req["questions"]["value"]["label"] = int(value_label)
    return req


def read_answer(answers: dict, placements: list[Placement]) -> tuple[Placement, dict[str, float]]:
    """-> (the chosen placement, probability per placement key) from a /v1/systemone response's `answers`."""
    move = answers["move"]
    by_key = {p.key: p for p in placements}
    return by_key[move["choice"]], {k: float(v) for k, v in move["probabilities"].items()}
