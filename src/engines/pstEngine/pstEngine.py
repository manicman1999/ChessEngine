import asyncio
import time
from typing import Optional
from src.engines.chessEngineBase import ChessEngineBase
from cychess import Board, square_to_alg
from math import inf

promo_chars = ['', 'q', 'n', 'b', 'r']


class PstEngine(ChessEngineBase):

    depth: int

    def __init__(self, depth: int = 4):
        self.depth = depth

    def choose_move(self, board: Board) -> Optional[str]:
        legal_moves = board.get_moves_list()

        start_time = time.time()

        results = []
        for move in legal_moves:
            board.make_move(*move)
            results.append(self.negamax(board, self.depth - 1))
            board.pop()

        best_move = None
        best_score = -inf
        total_evals = 0
        for move, (new_evals, score) in zip(legal_moves, results):
            total_evals += new_evals

            if -score > best_score:
                best_score = -score
                best_move = move

        total_time = time.time() - start_time
        print(f"{total_evals} evals in {total_time:.4f}s.")

        if best_move:
            return f"{square_to_alg(best_move[0])}{square_to_alg(move[1])}{promo_chars[move[2]]}"

    def eval_board(self, board: Board) -> int:
        if (result := board.game_result()) is not None:
            return int(1e10) * result

        x = board.eval_pst()
        return x

    def negamax(
        self, board: Board, depth: int, alpha: float = -inf, beta: float = inf
    ) -> tuple[int, float]:
        if depth == 0:
            return 1, self.eval_board(board)

        legal_moves = board.get_moves_list()
        if not legal_moves:
            return 1, self.eval_board(board)

        scores = []
        total_evals = 0

        for move in legal_moves:
            board.make_move(*move)
            new_evals, score = self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            total_evals += new_evals
            scores.append(score)

        max_score = -inf
        for score in scores:
            max_score = max(max_score, -score)
            alpha = max(alpha, -score)
            if alpha >= beta:
                break

        return total_evals, max_score
