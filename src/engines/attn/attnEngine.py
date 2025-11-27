import random
import time
from typing import Optional
import chess
import torch
from src.engines.attn.attnModel import AttnModel, board_to_tokens
from src.engines.chessEngineBase import ChessEngineBase
from math import inf


class AttnEngine(ChessEngineBase):

    depth: int

    def __init__(self, model_path: Optional[str] = None):
        self.model = AttnModel()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.depth = 3

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        if board.is_game_over():
            return None

        start_time = time.time()

        best_move = None
        best_score = -inf

        total_evals = 0

        for move in list(board.legal_moves):
            board.push(move)
            new_evals, score = self.negamax(board, self.depth - 1)
            board.pop()

            total_evals += new_evals

            if -score > best_score:
                best_score = -score
                best_move = move

        total_time = time.time() - start_time
        print(f"{total_evals} evals in {total_time:.4f}s.")

        return best_move

    def eval_board(self, board: chess.Board) -> float:
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return inf
            elif result == "0-1":
                return -inf
            return 0.0  # Draw

        with torch.no_grad():
            tokens = board_to_tokens(board)
            logit = self.model(tokens).item()

        # Scale logit to centipawn-like range (arbitrary; tune based on training)
        score = logit * 1000.0

        # Perspective: positive for white's advantage
        if board.turn == chess.BLACK:
            score = -score
        return score

    def negamax(
        self, board: chess.Board, depth: int, alpha: float = -inf, beta: float = inf
    ) -> tuple[int, float]:
        if depth == 0 or board.is_game_over():
            return 1, self.eval_board(board)

        max_score = -inf
        total_evals = 0
        for move in list(board.legal_moves):
            board.push(move)
            new_evals, score = self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            total_evals += new_evals

            max_score = max(max_score, -score)
            alpha = max(alpha, -score)
            if alpha >= beta:
                break
        return total_evals, max_score
