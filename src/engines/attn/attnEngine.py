import time
from typing import Optional
import torch
from src.engines.attn.attnModel import AttnModel
from src.engines.chessEngineBase import ChessEngineBase
from math import inf

from cychess import Board, square_to_alg
from attn_model import AttnModelCompiled

promo_chars = ['', 'q', 'n', 'b', 'r']


class AttnEngine(ChessEngineBase):

    depth: int
    compiled_model: AttnModelCompiled

    def __init__(self, model_path: Optional[str] = None, depth: int = 3):
        model = AttnModel()
        if model_path:
            model.load_state_dict(torch.load(model_path))
            
        self.compiled_model = model.attn_compile()
        self.depth = depth

    def choose_move(self, board: Board) -> Optional[str]:
        legal_moves = board.get_moves_list()
        legal_move_strs = [f"{square_to_alg(move[0])}{square_to_alg(move[1])}{promo_chars[move[2]]}" for move in legal_moves]

        start_time = time.time()
        
        if board.white_move():
            board.set_eval_func(lambda b: self.compiled_model.forward(b.tokenize()))
        else:
            board.set_eval_func(lambda b: -self.compiled_model.forward(b.tokenize()))

        results = []
        total_evals = 0
        for move in legal_moves:
            if board.make_move(*move):
                results.append(board.search(self.depth))
                board.pop()
                total_evals += board.get_eval_count()
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
        print(f"{total_time:.4f}s ({total_evals:,} evals)")

        if best_move and best_move_str:
            return best_move_str
