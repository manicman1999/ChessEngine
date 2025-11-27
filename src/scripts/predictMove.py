import argparse
import sys
import os
import chess
import chess.pgn

from src.engines.random.randomEngine import RandomEngine
from src.engines.chessEngineBase import ChessEngineBase

def load_board(fen: str) -> chess.Board:
    return chess.Board(fen)

def main():
    fen = input("FEN: ")
    board = chess.Board(fen)
    
    # Instantiate your engine here
    engine = RandomEngine()
    
    move = engine.choose_move(board)
    print(f"{engine.name} chooses: {move} ({move.uci()})")

if __name__ == "__main__":
    main()