from cychess import Board, py_sq_to_bit as sq_to_bit  # adjust import
board = Board()
board.set_start_position()
moves = board.get_moves_list()
print(f"Move count: {len(moves)}")  # Expect 20
print("Sample pawn moves:", [m for m in moves[:4] if m[0] // 8 == 1])  # e2-e3/e4 etc.
# Knights: e.g., (1, 16), (1, 18), (6, 17), (6, 21)

board.clear()
board.pieces[4] = sq_to_bit(0)  # a1 rook (need sq_to_bit Python wrapper?)
print(len(board.get_moves_list()))