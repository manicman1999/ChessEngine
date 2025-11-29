# negamax_search.pyx
# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: embedsignature=True

from libc.math cimport INFINITY
from libc.stdint cimport uint16_t

from cychess import Board

cpdef int negamax(object board, int depth, double alpha = -INFINITY, double beta = INFINITY):
    return _negamax(board, depth, alpha, beta)

cdef int _negamax(object board, int depth, double alpha, double beta):
    if depth == 0:
        return board.eval_pst()

    cdef int max_score = -8000
    cdef int i
    cdef object move
    cdef list moves = board.get_moves_list()
    cdef int score

    if len(moves) == 0:
        if board.is_in_check():
            if board.white_move():
                return 8000
            return -8000
        return 0

    for move in moves:
        if board.make_move(move[0], move[1], move[2]):
            score = _negamax(board, depth - 1, -beta, -alpha)
            board.undo_move()
        else:
            score = 0

        score = -score
        if score > max_score:
            max_score = score
        if score > alpha:
            alpha = <double>score
        if alpha >= beta:
            break

    return max_score