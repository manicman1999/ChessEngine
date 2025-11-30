import time
from cychess import Board

board = Board()
board.set_start_position()

print("Perft(1):", board.perft(1))  # 20
print("Perft(2):", board.perft(2))  # 400
print("Perft(3):", board.perft(3))  # 8902
print("Perft(4):", board.perft(4))  # 197281 (should take <1s)

start_time = time.time()
perft5 = board.perft(5)
total_time = (time.time() - start_time) * 1000
print(f"Perft(5) done in {total_time:.2f} ms ({perft5})")

