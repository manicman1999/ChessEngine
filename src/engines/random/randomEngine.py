import chess
from src.engines.chessEngineBase import ChessEngineBase


class RandomEngine(ChessEngineBase):
    def choose_move(self, board: chess.Board) -> chess.Move:
        import random

        return random.choice(list(board.legal_moves))