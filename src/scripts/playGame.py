import asyncio
import chess
import chess.pgn
from typing import Optional

from src.engines.attn.attnEngine import AttnEngine
from src.engines.random.randomEngine import RandomEngine
from src.engines.chessEngineBase import ChessEngineBase


async def run_game(
    white_engine: ChessEngineBase,
    black_engine: ChessEngineBase,
    start_fen: Optional[str] = None,
    max_fullmoves: int = 200,
) -> dict[str, str]:
    """
    Runs a full chess game between white_engine and black_engine.
    Prints board state after each move.
    Stops early if max_fullmoves reached (default 200).
    Returns {'pgn': ..., 'result': '1-0' | '0-1' | '1/2-1/2', 'winner': 'White' | 'Black' | 'Draw'}.
    """
    board = chess.Board(start_fen or chess.Board().fen())

    while not board.is_game_over() and board.fullmove_number <= max_fullmoves:
        if board.turn == chess.WHITE:
            engine = white_engine
            color = "White"
        else:
            engine = black_engine
            color = "Black"

        move = await engine.choose_move(board)
        if move is None:
            break
        print(
            f"{color} ({engine.name}): {move} ({move.xboard()}) {board.fullmove_number}"
        )
        board.push(move)

    if board.is_game_over():
        result = board.result()
        termination = (
            outcome.termination.name
            if (outcome := board.outcome(claim_draw=True))
            else "Unknown"
        )
    else:
        result = "1/2-1/2"
        termination = "Max fullmoves reached"

    winner = "White" if result == "1-0" else "Black" if result == "0-1" else "Draw"
    print(
        f"\nGame over after {board.fullmove_number - 1} full moves! {winner} ({result})"
    )
    print(f"Termination: {termination}")

    # Generate PGN
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = f"{white_engine.name} vs {black_engine.name}"
    game.headers["White"] = white_engine.name
    game.headers["Black"] = black_engine.name
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    pgn = str(game)

    return {
        "pgn": pgn,
        "result": result,
        "winner": winner,
        "fullmoves": str(board.fullmove_number - 1),
    }


async def main():
    white = RandomEngine()
    black = AttnEngine()
    result = await run_game(white, black)
    print(
        f"\nSummary: {result['winner']} ({result['result']}) after {result['fullmoves']} full moves"
    )


if __name__ == "__main__":
    asyncio.run(main())
