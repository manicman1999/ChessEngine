# src/cython/cychess.pyx
# distutils: language = c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: embedsignature=True

from libc.stdint cimport uint64_t, uint8_t, uint16_t, int8_t

# Piece constants (C-level)
cdef public uint8_t PIECE_NONE = 0
cdef public uint8_t PIECE_WP = 1, PIECE_WN = 2, PIECE_WB = 3, PIECE_WR = 4
cdef public uint8_t PIECE_WQ = 5, PIECE_WK = 6
cdef public uint8_t PIECE_BP = 7, PIECE_BN = 8, PIECE_BB = 9, PIECE_BR = 10
cdef public uint8_t PIECE_BQ = 11, PIECE_BK = 12

# Castling
cdef uint8_t CASTLE_WK = 1, CASTLE_WQ = 2, CASTLE_BK = 4, CASTLE_BQ = 8

# Knight Deltas
cdef int KNIGHT_DELTAS[8]
KNIGHT_DELTAS[:] = [15, 17, 10, 6, -6, -10, -17, -15]

# Pawn deltas by side (0=white/up, 1=black/down)
cdef int8_t PAWN_PUSH[2]
PAWN_PUSH[:] = [8, -8]
cdef int8_t PAWN_DOUBLE[2]
PAWN_DOUBLE[:] = [16, -16]
cdef int8_t PAWN_CAP_WEST[2]   # "Left" from player view: white west +7, black west -9
PAWN_CAP_WEST[:] = [7, -9]
cdef int8_t PAWN_CAP_EAST[2]   # "Right": white +9, black -7
PAWN_CAP_EAST[:] = [9, -7]

# King deltas (all 8 directions, single step)
cdef int KING_DELTAS[8]
KING_DELTAS[:] = [-9, -8, -7, -1, 1, 7, 8, 9]

# Slider directions
cdef int ROOK_DIRECTIONS[4]
ROOK_DIRECTIONS[:] = [8, -8, 1, -1]  # N S E W

cdef int BISHOP_DIRECTIONS[4]
BISHOP_DIRECTIONS[:] = [9, 7, -7, -9]  # NE NW SW SE

cdef int QUEEN_DIRECTIONS[8]
QUEEN_DIRECTIONS[:] = [8, -8, 1, -1, 9, 7, -7, -9]  # Rook + Bishop

# Free inline functions (pure C, GIL-free)
cdef inline uint64_t sq_to_bit(int sq) nogil:
    return 1ULL << sq

cpdef uint64_t py_sq_to_bit(int sq): return sq_to_bit(sq)

cdef struct Move:
    uint8_t fr_sq   # from square (0-63)
    uint8_t to_sq   # to square (0-63)

cdef class Board:
    cdef uint64_t pieces[12]        # 12 bitboards
    cdef uint64_t occupancy[3]      # 0=white, 1=black, 2=all
    cdef bint white_to_move
    cdef uint8_t castling
    cdef int8_t ep_square
    cdef uint16_t halfmove, fullmove
    cdef Move moves[256]     # Fixed buffer for pseudo-legal moves (plenty for chess max ~218)
    cdef int move_count

    def __cinit__(self):
        self._clear()

    cdef void _clear(self) nogil:
        cdef int i
        for i in range(12):
            self.pieces[i] = 0ULL
        self.occupancy[0] = self.occupancy[1] = self.occupancy[2] = 0ULL
        self.white_to_move = 1
        self.castling = 0
        self.ep_square = -1
        self.halfmove = 0
        self.fullmove = 1

    cpdef void clear(self):
        self._clear()

    cdef void _update_occupancy(self) nogil:
        cdef uint64_t white = 0ULL
        cdef uint64_t black = 0ULL
        cdef int i
        for i in range(6):
            white |= self.pieces[i]
            black |= self.pieces[6 + i]
        self.occupancy[0] = white
        self.occupancy[1] = black
        self.occupancy[2] = white | black

    cdef void _set_start_position(self) nogil:
        cdef int i
        for i in range(12):
            self.pieces[i] = 0ULL

        # White back rank
        self.pieces[PIECE_WR - 1] |= sq_to_bit(0) | sq_to_bit(7)
        self.pieces[PIECE_WN - 1] |= sq_to_bit(1) | sq_to_bit(6)
        self.pieces[PIECE_WB - 1] |= sq_to_bit(2) | sq_to_bit(5)
        self.pieces[PIECE_WQ - 1] |= sq_to_bit(3)
        self.pieces[PIECE_WK - 1] |= sq_to_bit(4)

        self.pieces[PIECE_WP - 1] = 0x000000000000FF00ULL

        self.pieces[PIECE_BR - 1] |= sq_to_bit(56) | sq_to_bit(63)
        self.pieces[PIECE_BN - 1] |= sq_to_bit(57) | sq_to_bit(62)
        self.pieces[PIECE_BB - 1] |= sq_to_bit(58) | sq_to_bit(61)
        self.pieces[PIECE_BQ - 1] |= sq_to_bit(59)
        self.pieces[PIECE_BK - 1] |= sq_to_bit(60)

        self.pieces[PIECE_BP - 1] = 0x00FF000000000000ULL

        self.white_to_move = 1
        self.castling = 15
        self.ep_square = -1
        self.halfmove = 0
        self.fullmove = 1
        self._update_occupancy()

    cpdef void set_start_position(self):
        self._set_start_position()

    # Python-accessible getters for testing
    cpdef uint64_t get_occupancy(self):
        return self.occupancy[2]

    cdef int _generate_knight_moves(self, uint64_t knights_bb, uint64_t own_occ, uint64_t opp_occ) nogil:
        cdef int sq, target, delta
        cdef uint64_t target_bb
        cdef int dx, dy

        for sq in range(64):
            if (sq_to_bit(sq) & knights_bb) == 0:
                continue
            for delta in range(8):
                if self.move_count >= 256:
                    return self.move_count
                target = sq + KNIGHT_DELTAS[delta]
                if 0 <= target < 64:
                    dx = (target % 8) - (sq % 8)
                    dy = (target // 8) - (sq // 8)
                    if ((abs(dx) == 1 and abs(dy) == 2) or (abs(dx) == 2 and abs(dy) == 1)):
                        target_bb = sq_to_bit(target)
                        if (target_bb & own_occ) == 0:
                            self.moves[self.move_count].fr_sq = <uint8_t>sq
                            self.moves[self.move_count].to_sq = <uint8_t>target
                            self.move_count += 1
        return self.move_count

    cdef int _generate_pawn_moves(self, uint64_t pawns_bb, uint64_t own_occ, uint64_t opp_occ, uint8_t side) nogil:
        """Pseudo-legal pawn moves: single/double pushes (quiet), diagonal captures. Fills moves[]."""
        cdef int sq, target
        cdef uint64_t target_bb
        cdef int push_delta = PAWN_PUSH[side]
        cdef int double_delta = PAWN_DOUBLE[side]
        cdef bint start_rank
        cdef int sq_file, tgt_file

        for sq in range(64):
            if (sq_to_bit(sq) & pawns_bb) == 0:
                continue

            
            if self.move_count >= 256:
                return self.move_count

            start_rank = (side == 0 and (sq // 8 == 1)) or (side == 1 and (sq // 8 == 6))

            # Single push (quiet)
            target = sq + push_delta
            if 0 <= target < 64:
                target_bb = sq_to_bit(target)
                if (target_bb & (own_occ | opp_occ)) == 0:  # Empty
                    self.moves[self.move_count].fr_sq = sq
                    self.moves[self.move_count].to_sq = target
                    self.move_count += 1

                    # Double push (from start rank, if single empty)
                    if start_rank:
                        double_target = sq + double_delta
                        double_bb = sq_to_bit(double_target)
                        if 0 <= double_target < 64 and (double_bb & (own_occ | opp_occ)) == 0:
                            self.moves[self.move_count].fr_sq = sq
                            self.moves[self.move_count].to_sq = double_target
                            self.move_count += 1

            # Captures
            # West cap
            target = sq + PAWN_CAP_WEST[side]
            if 0 <= target < 64:
                target_bb = sq_to_bit(target)
                if (target_bb & opp_occ) != 0:
                    sq_file = sq & 7
                    tgt_file = target & 7
                    if abs(tgt_file - sq_file) == 1:  # Diagonal: adjacent file
                        self.moves[self.move_count].fr_sq = sq
                        self.moves[self.move_count].to_sq = target
                        self.move_count += 1

            # East cap
            target = sq + PAWN_CAP_EAST[side]
            if 0 <= target < 64:
                target_bb = sq_to_bit(target)
                if (target_bb & opp_occ) != 0:
                    sq_file = sq & 7
                    tgt_file = target & 7
                    if abs(tgt_file - sq_file) == 1:
                        self.moves[self.move_count].fr_sq = sq
                        self.moves[self.move_count].to_sq = target
                        self.move_count += 1

        return self.move_count

    cdef int _generate_king_moves(self, uint64_t kings_bb, uint64_t own_occ, uint64_t opp_occ) nogil:
        cdef int sq, target, delta
        cdef uint64_t target_bb
        cdef int dx, dy

        for sq in range(64):
            if (sq_to_bit(sq) & kings_bb) == 0:
                continue
            for delta in range(8):
                if self.move_count >= 256:
                    return self.move_count
                target = sq + KING_DELTAS[delta]
                if 0 <= target < 64:
                    dx = (target % 8) - (sq % 8)
                    dy = (target // 8) - (sq // 8)
                    if abs(dx) <= 1 and abs(dy) <= 1 and (dx != 0 or dy != 0):
                        target_bb = sq_to_bit(target)
                        if (target_bb & own_occ) == 0:
                            self.moves[self.move_count].fr_sq = <uint8_t>sq
                            self.moves[self.move_count].to_sq = <uint8_t>target
                            self.move_count += 1
        return self.move_count

    cdef int _generate_rook_moves(self, uint64_t rooks_bb, uint64_t own_occ, uint64_t opp_occ) nogil:
        cdef uint64_t blockers = own_occ | opp_occ
        cdef int sq, dir_idx, target, dir_delta
        cdef uint64_t target_bb
        cdef int dfile, drank
        cdef bint vertical, horizontal

        for sq in range(64):
            if (sq_to_bit(sq) & rooks_bb) == 0:
                continue
            for dir_idx in range(4):
                if self.move_count >= 256:
                    return self.move_count
                dir_delta = ROOK_DIRECTIONS[dir_idx]
                target = sq + dir_delta
                while 0 <= target < 64:
                    dfile = (target & 7) - (sq & 7)
                    drank = (target // 8) - (sq // 8)
                    vertical = (dir_delta == 8 or dir_delta == -8)
                    horizontal = (dir_delta == 1 or dir_delta == -1)
                    if vertical:
                        if (target & 7) != (sq & 7):
                            break
                    elif horizontal:
                        if (target // 8) != (sq // 8):
                            break
                    else:
                        if abs(dfile) != abs(drank):
                            break  # Won't trigger for rook dirs

                    target_bb = sq_to_bit(target)
                    if (target_bb & blockers) != 0:
                        if (target_bb & opp_occ) != 0:
                            self.moves[self.move_count].fr_sq = <uint8_t>sq
                            self.moves[self.move_count].to_sq = <uint8_t>target
                            self.move_count += 1
                        break
                    else:
                        self.moves[self.move_count].fr_sq = <uint8_t>sq
                        self.moves[self.move_count].to_sq = <uint8_t>target
                        self.move_count += 1
                    target += dir_delta
        return self.move_count

    cdef int _generate_bishop_moves(self, uint64_t bishops_bb, uint64_t own_occ, uint64_t opp_occ) nogil:
        cdef uint64_t blockers = own_occ | opp_occ
        cdef int sq, dir_idx, target, dir_delta
        cdef uint64_t target_bb
        cdef int dfile, drank
        cdef bint vertical, horizontal

        for sq in range(64):
            if (sq_to_bit(sq) & bishops_bb) == 0:
                continue
            for dir_idx in range(4):
                if self.move_count >= 256:
                    return self.move_count
                dir_delta = BISHOP_DIRECTIONS[dir_idx]
                target = sq + dir_delta
                while 0 <= target < 64:
                    dfile = (target & 7) - (sq & 7)
                    drank = (target // 8) - (sq // 8)
                    vertical = (dir_delta == 8 or dir_delta == -8)
                    horizontal = (dir_delta == 1 or dir_delta == -1)
                    if vertical:
                        if (target & 7) != (sq & 7):
                            break
                    elif horizontal:
                        if (target // 8) != (sq // 8):
                            break
                    else:
                        if abs(dfile) != abs(drank):
                            break

                    target_bb = sq_to_bit(target)
                    if (target_bb & blockers) != 0:
                        if (target_bb & opp_occ) != 0:
                            self.moves[self.move_count].fr_sq = <uint8_t>sq
                            self.moves[self.move_count].to_sq = <uint8_t>target
                            self.move_count += 1
                        break
                    else:
                        self.moves[self.move_count].fr_sq = <uint8_t>sq
                        self.moves[self.move_count].to_sq = <uint8_t>target
                        self.move_count += 1
                    target += dir_delta
        return self.move_count

    cdef int _generate_queen_moves(self, uint64_t queens_bb, uint64_t own_occ, uint64_t opp_occ) nogil:
        cdef uint64_t blockers = own_occ | opp_occ
        cdef int sq, dir_idx, target, dir_delta
        cdef uint64_t target_bb
        cdef int dfile, drank
        cdef bint vertical, horizontal

        for sq in range(64):
            if (sq_to_bit(sq) & queens_bb) == 0:
                continue
            for dir_idx in range(8):
                if self.move_count >= 256:
                    return self.move_count
                dir_delta = QUEEN_DIRECTIONS[dir_idx]
                target = sq + dir_delta
                while 0 <= target < 64:
                    dfile = (target & 7) - (sq & 7)
                    drank = (target // 8) - (sq // 8)
                    vertical = (dir_delta == 8 or dir_delta == -8)
                    horizontal = (dir_delta == 1 or dir_delta == -1)
                    if vertical:
                        if (target & 7) != (sq & 7):
                            break
                    elif horizontal:
                        if (target // 8) != (sq // 8):
                            break
                    else:
                        if abs(dfile) != abs(drank):
                            break

                    target_bb = sq_to_bit(target)
                    if (target_bb & blockers) != 0:
                        if (target_bb & opp_occ) != 0:
                            self.moves[self.move_count].fr_sq = <uint8_t>sq
                            self.moves[self.move_count].to_sq = <uint8_t>target
                            self.move_count += 1
                        break
                    else:
                        self.moves[self.move_count].fr_sq = <uint8_t>sq
                        self.moves[self.move_count].to_sq = <uint8_t>target
                        self.move_count += 1
                    target += dir_delta
        return self.move_count

    cdef int generate_pseudo_legal_moves(self) nogil:
        self.move_count = 0
        cdef bint is_white = self.white_to_move
        cdef uint64_t own_occ = self.occupancy[0 if is_white else 1]
        cdef uint64_t opp_occ = self.occupancy[1 if is_white else 0]
        cdef uint8_t side = 0 if is_white else 1

        # Pawns
        cdef uint64_t pawns_bb = self.pieces[PIECE_WP - 1 if is_white else PIECE_BP - 1]
        self._generate_pawn_moves(pawns_bb, own_occ, opp_occ, side)

        # Knights
        cdef uint64_t knights_bb = self.pieces[PIECE_WN - 1 if is_white else PIECE_BN - 1]
        self._generate_knight_moves(knights_bb, own_occ, opp_occ)

        # Kings
        cdef uint64_t kings_bb = self.pieces[PIECE_WK - 1 if is_white else PIECE_BK - 1]
        self._generate_king_moves(kings_bb, own_occ, opp_occ)

        # Rooks
        cdef uint64_t rooks_bb = self.pieces[PIECE_WR - 1 if is_white else PIECE_BR - 1]
        self._generate_rook_moves(rooks_bb, own_occ, opp_occ)

        # Bishops
        cdef uint64_t bishops_bb = self.pieces[PIECE_WB - 1 if is_white else PIECE_BB - 1]
        self._generate_bishop_moves(bishops_bb, own_occ, opp_occ)

        # Queens
        cdef uint64_t queens_bb = self.pieces[PIECE_WQ - 1 if is_white else PIECE_BQ - 1]
        self._generate_queen_moves(queens_bb, own_occ, opp_occ)

        # TODO: sliders (rooks/bishops/queens), castling, EP, promotions
        return self.move_count

    cpdef int generate_moves(self):
        return self.generate_pseudo_legal_moves()

    cpdef list get_moves_list(self):
        self.generate_pseudo_legal_moves()
        cdef list result = []
        cdef int i
        for i in range(self.move_count):
            result.append((self.moves[i].fr_sq, self.moves[i].to_sq))
        return result