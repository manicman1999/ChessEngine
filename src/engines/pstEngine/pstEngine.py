import asyncio
import time
from typing import Any, Optional
from src.engines.chessEngineBase import ChessEngineBase
from cychess import Board, square_to_alg
from negamax import NegamaxSearch
from math import inf

promo_chars = ['', 'q', 'n', 'b', 'r']


class PstEngine(ChessEngineBase):

    depth: int
    seacher: NegamaxSearch

    def __init__(self, depth: int = 4):
        self.depth = depth
        self.searcher = NegamaxSearch(lambda b: b.material_balance())

    def choose_move(self, board: Board) -> Optional[str]:
        legal_moves = board.get_moves_list()
        legal_move_strs = [f"{square_to_alg(move[0])}{square_to_alg(move[1])}{promo_chars[move[2]]}" for move in legal_moves]

        start_time = time.time()

        start_pst = board.eval_pst()
        
        if board.white_move():
            board.set_eval_func(lambda b: b.eval_pst())
        else:
            board.set_eval_func(lambda b: -b.eval_pst())

        results = []
        for move in legal_moves:
            if board.make_move(*move):
                results.append(board.search(self.depth))
                board.pop()
            else:
                results.append(8000)

        best_move = None
        best_score = -inf
        best_move_str = ""
        for move, score in zip(legal_moves, results):
            score = -score
            if score > best_score:
                best_score = score
                best_move = move
                best_move_str = f"{square_to_alg(move[0])}{square_to_alg(move[1])}{promo_chars[move[2]]}"

            if not board.white_move():
                pass

        total_time = time.time() - start_time
        print(f"{total_time:.4f}s")

        if best_move and best_move_str:
            return best_move_str
