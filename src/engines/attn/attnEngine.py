import asyncio
import time
from typing import Optional
import torch
from src.engines.attn.modelWorker import ModelWorker
from src.engines.attn.attnModel import AttnModel, board_to_tokens
from src.engines.chessEngineBase import ChessEngineBase
from math import inf

from cychess import Board
from attn_model import AttnModelCompiled


class AttnEngine(ChessEngineBase):

    depth: int

    def __init__(self, model_path: Optional[str] = None):
        self.model = AttnModel()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            
        

        self.modelWorker = ModelWorker(self.model)
        self.model.eval()
        # self.model = torch.compile(self.model, mode="reduce-overhead")
        self.depth = 4
        self._single_eval_cache = {}

    async def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        if board.is_game_over():
            return None

        start_time = time.time()

        best_move = None
        best_score = -inf

        total_evals = 0

        eval_tasks = []
        legal_moves = list(board.legal_moves)

        for move in legal_moves:
            board.push(move)
            eval_tasks.append(self.negamax(board.copy(), self.depth - 1))
            board.pop()

        eval_results = await asyncio.gather(*eval_tasks)

        for move, result in zip(legal_moves, eval_results):
            new_evals, score = result

            total_evals += new_evals

            if -score > best_score:
                best_score = -score
                best_move = move

        total_time = time.time() - start_time
        print(f"{total_evals} evals in {total_time:.4f}s.")

        return best_move

    async def eval_board(self, board: chess.Board) -> float:
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return inf
            elif result == "0-1":
                return -inf
            return 0.0

        fen = board.fen()
        if fen in self._single_eval_cache:
            return self._single_eval_cache[fen]

        tokens = board_to_tokens(board)  # [1,64]
        logit = await self.modelWorker.eval(tokens)

        score = logit * 1000.0
        if board.turn == chess.BLACK:
            score = -score

        self._single_eval_cache[fen] = score
        return score

    async def negamax(
        self, board: chess.Board, depth: int, alpha: float = -inf, beta: float = inf
    ) -> tuple[int, float]:
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 0, inf
            elif result == "0-1":
                return 0, -inf
            return 0, 0.0  # Draw

        if depth == 0:
            return 1, await self.eval_board(board)

        legal_moves = list(board.legal_moves)
        negamax_tasks = []

        for move in legal_moves:
            board.push(move)
            negamax_tasks.append(self.negamax(board.copy(), depth - 1, -beta, -alpha))
            board.pop()

        max_score = -inf
        total_evals = 0
        for future in asyncio.as_completed(negamax_tasks):
            try:
                new_evals, score = await future
            except asyncio.CancelledError:
                continue

            total_evals += new_evals

            max_score = max(max_score, -score)
            alpha = max(alpha, -score)
            if alpha >= beta:
                for task in negamax_tasks:
                    if not task.done():
                        task.cancel()
                break
        return total_evals, max_score
