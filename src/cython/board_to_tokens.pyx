# src/cython/board_to_tokens.pyx
import torch
import chess
import numpy as np
from libc.stdlib cimport malloc, free  # Not strictly needed here, but for future

# Constants (Python globals; Cython inlines them)
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
NUM_PIECES = 6
GROUP_SIZE = NUM_PIECES * 2 + 1  # 13
# Precompute index dict for fast lookup (avoids .index() Python call)
PT_TO_IDX = {pt: i for i, pt in enumerate(PIECE_TYPES)}

def board_to_tokens(board):  # Untyped: object board (chess.Board)
    # Init empties vectorized (still Python, but fast)
    offsets = torch.arange(64, dtype=torch.long) * GROUP_SIZE
    tokens_np = (offsets + 12).numpy()  # Convert to numpy for typed view (np.int64)
    
    cdef long long[:] tokens = tokens_np  # Fixed: Use long long for 64-bit match
    
    # Get piece_map (Python call, but only once)
    cdef object piece_map = board.piece_map()  # dict[sq: Piece]
    
    cdef int sq  # sq is 0-63, int is fine
    cdef object piece  # Untyped Piece (access .piece_type, .color as Python attrs)
    cdef long long offset, pt_idx, col_off  # Updated to long long for consistency
    
    # Typed loop: Only over occupied squares (~32 max)
    for sq_item in piece_map.items():
        sq = sq_item[0]  # int sq
        piece = sq_item[1]  # Piece object
        offset = sq * GROUP_SIZE
        pt_idx = PT_TO_IDX[piece.piece_type]  # Dict lookup (fast)
        col_off = NUM_PIECES if piece.color == chess.BLACK else 0
        tokens[sq] = offset + pt_idx + col_off
    
    # Back to Torch tensor (preserves dtype=torch.long)
    return torch.from_numpy(tokens_np).unsqueeze(0)