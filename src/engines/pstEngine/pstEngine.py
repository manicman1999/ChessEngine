import asyncio
import time
from typing import Any, Optional
from src.engines.chessEngineBase import ChessEngineBase
from cychess import Board, square_to_alg
from negamax import negamax
from math import inf

promo_chars = ['', 'q', 'n', 'b', 'r']


class PstEngine(ChessEngineBase):

    depth: int
    eval_cache: dict[Any, int]

    def __init__(self, depth: int = 4):
        self.depth = depth
        self.eval_cache = {}

    def choose_move(self, board: Board) -> Optional[str]:
        legal_moves = board.get_moves_list()
        legal_move_strs = [f"{square_to_alg(move[0])}{square_to_alg(move[1])}{promo_chars[move[2]]}" for move in legal_moves]

        start_time = time.time()

        start_pst = board.eval_pst()

        results = []
        for move in legal_moves:
            if board.make_move(*move):
                results.append(negamax(board, self.depth - 1))
                board.pop()
            else:
                results.append(8000)

        best_move = None
        best_score = -inf
        best_move_str = ""
        total_evals = 0
        for move, score in zip(legal_moves, results):
            total_evals += 0

            score *= -1

            if score > best_score:
                best_score = score
                best_move = move
                best_move_str = f"{square_to_alg(move[0])}{square_to_alg(move[1])}{promo_chars[move[2]]}"

            if not board.white_move():
                pass

        total_time = time.time() - start_time
        print(f"{total_evals} evals in {total_time:.4f}s.")

        if best_move and best_move_str:
            return best_move_str
