from cychess import Board, square_to_alg, alg_to_square
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

# Queen mid test
board.clear()
board.set_piece(27, 5)  # e4 queen
moves = board.get_moves_list()

print("Queen e4 moves: ")

for move in moves:
    print(f"{square_to_alg(move[0])}{square_to_alg(move[1])}", end = ' ')