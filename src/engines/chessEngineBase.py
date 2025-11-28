from dataclasses import dataclass
from typing import Optional

from cychess import Board

@dataclass
class ChessEngineBase:

    @property
    def name(self):
        return self.__class__.__name__

    async def choose_move(self, board: Board) -> Optional[str]:
        raise NotImplementedError("Override in subclass")