from cychess import Board  # adjust import
board = Board()
board.set_start_position()
moves = board.get_moves_list()
print(f"Move count: {len(moves)}")  # Expect 20
print("Sample pawn moves:", [m for m in moves[:4] if m[0] // 8 == 1])  # e2-e3/e4 etc.
# Knights: e.g., (1, 16), (1, 18), (6, 17), (6, 21)