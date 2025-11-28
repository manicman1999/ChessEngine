import random
import time
from cychess import Board, Move, square_to_alg, alg_to_square

def alg_move(move: Move):
    fr, to, promo = move
    s = f"{square_to_alg(fr)}{square_to_alg(to)}"
    if promo:
        promos = {1: 'q', 2: 'n', 3: 'b', 4: 'r'}
        s += promos[promo]
    return s

board = Board()
board.set_start_position()

moves_history = []
while len(moves_history) < 200:
    legal_moves = board.get_moves_list()
    result = board.game_result()
    if result is not None:
        winner = {1: "White wins", -1: "Black wins", 0: "Draw"}[result]
        print(f"\nGame end after {len(moves_history)} halfmoves: {winner}")
        break
    if len(legal_moves) == 0:
        print(f"\nNo legal moves after {len(moves_history)} halfmoves")
        break
    
    chosen = random.choice(legal_moves)
    succ = board.make_move(chosen[0], chosen[1], chosen[2])
    assert succ
    moves_history.append(alg_move(chosen))
    
    print(f"Move {len(moves_history)}: {alg_move(chosen)} (total legal now: {len(board.get_moves_list())})")

if len(moves_history) >= 200:
    print("\nHit move limit")
print("Moves:", " ".join(moves_history))