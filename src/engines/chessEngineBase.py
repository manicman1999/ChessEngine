from dataclasses import dataclass
from typing import Optional
import chess

@dataclass
class ChessEngineBase:

    @property
    def name(self):
        return self.__class__.__name__

    async def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        raise NotImplementedError("Override in subclass")