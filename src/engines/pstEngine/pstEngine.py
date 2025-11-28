import asyncio
import time
from typing import Any, Optional
from src.engines.chessEngineBase import ChessEngineBase
from cychess import Board, square_to_alg
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

        start_score = self.eval_board(board)

        start_time = time.time()

        results = []
        for move in legal_moves:
            board.make_move(*move)
            results.append(self.negamax(board, self.depth - 1))
            board.pop()

        best_move = None
        best_score = -inf
        best_move_str = ""
        total_evals = 0
        for move, (new_evals, score) in zip(legal_moves, results):
            total_evals += new_evals

            if board.white_move():
                score *= -1

            if score > best_score:
                best_score = score
                best_move = move
                best_move_str = f"{square_to_alg(move[0])}{square_to_alg(move[1])}{promo_chars[move[2]]}"

        total_time = time.time() - start_time
        print(f"{total_evals} evals in {total_time:.4f}s.")

        if best_move and best_move_str:
            return best_move_str

    def eval_board(self, board: Board) -> int:
        bhash = board.zobrist_hash()
        if bhash in self.eval_cache:
            return self.eval_cache[bhash]

        if (result := board.game_result()) is not None:
            score = int(1e10) * result
        else:
            score = board.eval_pst()
        self.eval_cache[bhash] = score
        return score

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
