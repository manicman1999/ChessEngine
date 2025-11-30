# negamax_search.pyx
# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cdivision=True
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: embedsignature=True

from libc.math cimport INFINITY
from libc.stdint cimport uint64_t
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool
from math import inf  # For Python-side inf

from cychess import Board

cdef class NegamaxSearch:
    
    cdef object eval_func  # Python callable: Board -> float (white-relative score)
    cdef unordered_map[uint64_t, double] eval_cache  # Strict C++ cache: uint64_t -> double
    
    def __init__(self, eval_func):
        self.eval_func = eval_func
        # No explicit init needed for unordered_map; defaults to empty
    
    cpdef double search(self, object board, int depth):
        return self._search(board, depth)
    
    # Optional: For Python introspection (e.g., print cache size)
    cpdef size_t get_cache_size(self):
        return self.eval_cache.size()
    
    cdef double _search(self, object board, int depth):
        cdef uint64_t key
        cdef double eval_score
        if depth == 0:
            return <double>self.eval_func(board)
        
        cdef list moves = board._get_moves_list()
        if len(moves) == 0:
            if board.is_in_check():
                if board.white_move():
                    return -80000.0  # White mated: bad for white
                else:
                    return 80000.0   # Black mated: good for white
            return 0.0  # Stalemate
        
        cdef double max_score = -99999.0
        cdef object move
        cdef double score
        
        for move in moves:
            if board.make_move(move[0], move[1], move[2]):
                score = self._search(board, depth - 1)
                board.undo_move()
            else:
                score = 100000.0
            
            score = -score
            if score > max_score:
                max_score = score
        
        return max_score