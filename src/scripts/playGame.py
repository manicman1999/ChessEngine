import asyncio
from typing import Optional

from cychess import Board, alg_to_square

# Assuming your engines are in these paths; adjust if needed
from src.engines.pstEngine.pstEngine import PstEngine
from src.engines.random.randomEngine import RandomEngine
from src.engines.chessEngineBase import ChessEngineBase


def parse_move_alg(move_str: str) -> tuple[int, int, int]:
    """
    Parse algebraic move like 'e2e4' or 'e7e8q' to (fr_sq, to_sq, promo).
    Promo: 0=none, 1=Q, 2=N, 3=B, 4=R.
    """
    if len(move_str) not in (4, 5):
        raise ValueError(f"Invalid move string: {move_str}")

    fr_alg = move_str[:2]
    to_alg = move_str[2:4]
    promo_char = move_str[4:] if len(move_str) == 5 else ""

    fr_sq = alg_to_square(fr_alg)
    to_sq = alg_to_square(to_alg)

    promo = 0
    if promo_char:
        promo_map = {"q": 1, "n": 2, "b": 3, "r": 4}
        promo = promo_map.get(promo_char.lower(), 0)

    return fr_sq, to_sq, promo


async def run_game(
    white_engine: ChessEngineBase,
    black_engine: ChessEngineBase,
    max_fullmoves: int = 200,
) -> dict[str, str]:
    board = Board()
    board.set_start_position()

    moves_history = []  # List of alg strs like 'e2e4'
    while len(moves_history) // 2 <= max_fullmoves:
        result = board.game_result()
        if result is not None:
            break

        if board.white_move():
            engine = white_engine
            color = "White"
        else:
            engine = black_engine
            color = "Black"

        move_str = engine.choose_move(board)
        if move_str is None:
            break

        fr, to, promo = parse_move_alg(move_str)
        succ = board.make_move(fr, to, promo)
        if not succ:
            print(f"Invalid move attempted: {move_str}")
            break

        moves_history.append(move_str)
        mat_bal = board.material_balance()
        print(f"{color} ({engine.name}): {move_str} ({len(moves_history) // 2 - 1}) (Mat Balance: {mat_bal})")

    if result is None:
        result_str = "1/2-1/2"
        winner = "Draw"
    else:
        result_str = {1: "1-0", -1: "0-1", 0: "1/2-1/2"}[result]
        winner = {1: "White", -1: "Black", 0: "Draw"}[result]
    print(
        f"\nGame over after {len(moves_history) // 2 - 1} full moves! {winner} ({result_str})"
    )

    # Generate simple PGN
    pgn_lines = [
        '[Event "{} vs {}"]'.format(white_engine.name, black_engine.name),
        '[White "{}"]'.format(white_engine.name),
        '[Black "{}"]'.format(black_engine.name),
        '[Result "{}"]'.format(result_str),
        "",
    ]
    move_text = []
    for i, move in enumerate(moves_history, 1):
        if i % 2 == 1:
            move_text.append(f"{(i + 1) // 2}. {move}")
        else:
            move_text[-1] += f" {move}"
    pgn_lines.extend(move_text)
    pgn_lines.append(result_str)
    pgn = "\n".join(pgn_lines)

    return {
        "pgn": pgn,
        "result": result_str,
        "winner": winner,
        "fullmoves": str(len(moves_history) // 2 - 1),
    }


async def main():
    white = RandomEngine()
    black = PstEngine()
    result = await run_game(white, black)

    print()
    print(result['pgn'])
    print()
    print(
        f"\nSummary: {result['winner']} ({result['result']}) after {result['fullmoves']} full moves"
    )


if __name__ == "__main__":
    asyncio.run(main())
