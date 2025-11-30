import asyncio
import time
from typing import Any, Optional
from src.engines.chessEngineBase import ChessEngineBase
from cychess import Board, square_to_alg
from negamax import NegamaxSearch
from math import inf

promo_chars = ['', 'q', 'n', 'b', 'r']


class PlayerEngine(ChessEngineBase):

    def choose_move(self, board: Board) -> Optional[str]:
        legal_moves = board.get_moves_list()
        legal_move_strs = [f"{square_to_alg(move[0])}{square_to_alg(move[1])}{promo_chars[move[2]]}" for move in legal_moves]
        legal_move_str = '\t'.join(legal_move_strs)
        
        if len(legal_moves) == 0:
            return None
        
        print(f"Legal moves: {legal_move_str}")
        
        move_choice = ""
        while move_choice not in legal_move_strs:
            move_choice = input("Move: ")

        return move_choice