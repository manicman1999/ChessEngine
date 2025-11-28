from typing import Optional
import random
from cychess import Board, square_to_alg
from src.engines.chessEngineBase import ChessEngineBase

promo_chars = ['', 'q', 'n', 'b', 'r']


class RandomEngine(ChessEngineBase):
    async def choose_move(self, board: Board) -> Optional[str]:
        legal_moves = board.get_moves_list()

        if len(legal_moves) == 0:
            return None
        else:
            move = random.choice(legal_moves)

            return f"{square_to_alg(move[0])}{square_to_alg(move[1])}{promo_chars[move[2]]}"