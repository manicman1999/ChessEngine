import time
from cychess import Board, square_to_alg, alg_to_square
import chess
board = Board()
board.set_start_position()
print(f"Start moves: {len(board.get_moves_list())}")  # 20

# Rook a1 test
board.clear()
board.set_piece(0, 4)  # a1 white rook, occ auto
print(f"Rook a1 moves: {len(board.get_moves_list())}")  # 14

# e2e4 then sliders unlock
board.set_start_position()
succ = board.make_move(alg_to_square('e2'), alg_to_square('e4'))  # e2(12)-e4(20)
print(f"After e2e4 moves ({succ}): {len(board.get_moves_list())}")  # 29? (pawn opens bishops/queen etc.)
succ = board.make_move(alg_to_square('e7'), alg_to_square('e5'))  # e2(12)-e4(20)
print(f"After e7e5 moves ({succ}): {len(board.get_moves_list())}")  # 29? (pawn opens bishops/queen etc.)
board.undo_move()
board.undo_move()
print(f"After some undos: {len(board.get_moves_list())}")

# Queen mid test
board.clear()
board.set_piece(27, 5)  # e4 queen
moves = board.get_moves_list()

print("Queen e4 moves: ", end='\t')

for move in moves:
    print(f"{square_to_alg(move[0])}{square_to_alg(move[1])}", end = ' ')
print()

board = Board()
board.set_start_position()
board.make_move(12, 20)  # e2-e4 (sq12→20)
print(f"Black pseudo-legal after e2e4: {len(board.get_moves_list())}")  # Expect ~35 (20 pawns + knights + some sliders)
board.undo_move()
print("Undid OK:", len(board.get_moves_list()) == 20)  # True

board = Board()
board.set_start_position()

print("Perft(1):", board.perft(1))  # 20
print("Perft(2):", board.perft(2))  # 400
print("Perft(3):", board.perft(3))  # 8902
print("Perft(4):", board.perft(4))  # 197281 (should take <1s)

start_time = time.time()
board.perft(5)
total_time = (time.time() - start_time) * 1000
print(f"Perft(5) done in {total_time:.2f} ms")

