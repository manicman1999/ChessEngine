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

# Promo
cdef uint8_t PROMO_Q = 1, PROMO_N = 2, PROMO_B = 3, PROMO_R = 4
cdef uint8_t PROMO_PIECES[2][4]  # side 0=white,1=black; idx 0=Q,1=N,...
PROMO_PIECES[0][0] = PIECE_WQ; PROMO_PIECES[0][1] = PIECE_WN; PROMO_PIECES[0][2] = PIECE_WB; PROMO_PIECES[0][3] = PIECE_WR
PROMO_PIECES[1][0] = PIECE_BQ; PROMO_PIECES[1][1] = PIECE_BN; PROMO_PIECES[1][2] = PIECE_BB; PROMO_PIECES[1][3] = PIECE_BR

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

cpdef str square_to_alg(int sq):
    """0-63 → 'a1' alg notation."""
    if sq < 0 or sq >= 64:
        return "??"
    cdef int file = sq % 8
    cdef int rank = sq // 8
    return chr(ord('a') + file) + chr(ord('1') + rank)

cpdef int alg_to_square(str alg):
    """'a1' → 0-63 sq, or -1 invalid."""
    if len(alg) != 2:
        return -1
    cdef int file = ord(alg[0]) - ord('a')
    cdef int rank = ord(alg[1]) - ord('1')
    if 0 <= file <= 7 and 0 <= rank <= 7:
        return rank * 8 + file
    return -1

cdef inline bint is_promo_rank(int sq, uint8_t side) nogil:
    return (side == 0 and (sq // 8) == 7) or (side == 1 and (sq // 8) == 0)

cdef struct Move:
    uint8_t fr_sq
    uint8_t to_sq
    uint8_t promo

cdef struct UndoState:
    uint8_t fr_sq
    uint8_t to_sq
    uint8_t mover_type
    uint8_t cap_type
    uint8_t castling
    uint8_t promo
    int8_t ep_square
    int8_t ep_captured_sq
    int8_t castle_rook_fr
    uint16_t halfmove
    uint16_t fullmove

cdef class Board:
    # Board state
    cdef uint64_t pieces[12]        # 12 bitboards
    cdef uint64_t occupancy[3]      # 0=white, 1=black, 2=all
    cdef bint white_to_move
    cdef uint8_t castling
    cdef int8_t ep_square
    cdef uint16_t halfmove, fullmove

    # Moves state
    cdef Move moves[256]     # Fixed buffer for pseudo-legal moves (plenty for chess max ~218)
    cdef int move_count

    cdef UndoState undo_stack[1024]
    cdef int undo_index  # In __cinit__: self.undo_index = 0

    cdef Move move_cache[50][256]
    cdef int move_count_cache[50]

    def __cinit__(self):
        self._clear()
        self.undo_index = 0

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

    cdef inline int find_king_sq(self, bint is_white) nogil:
        cdef uint64_t kings_bb = self.pieces[PIECE_WK - 1 if is_white else PIECE_BK - 1]
        cdef int sq
        for sq in range(64):
            if sq_to_bit(sq) & kings_bb:
                return sq
        return -1  # Error state

    cdef inline void add_move(self, int fr, int to, uint8_t promo = 0) nogil:
        if self.move_count >= 256:
            return
        self.moves[self.move_count].fr_sq = <uint8_t>fr
        self.moves[self.move_count].to_sq = <uint8_t>to
        self.moves[self.move_count].promo = promo
        self.move_count += 1

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
                            self.add_move(sq, target)
        return self.move_count

    cdef int _generate_pawn_moves(self, uint64_t pawns_bb, uint64_t own_occ, uint64_t opp_occ, uint8_t side) nogil:
        cdef int sq, target, double_target
        cdef uint64_t target_bb
        cdef int push_delta = PAWN_PUSH[side]
        cdef int double_delta = PAWN_DOUBLE[side]
        cdef bint start_rank = False
        cdef bint attacker_rank_ok, is_promo
        cdef int p

        for sq in range(64):
            if (sq_to_bit(sq) & pawns_bb) == 0:
                continue
            start_rank = (side == 0 and (sq // 8) == 1) or (side == 1 and (sq // 8) == 6)

            # Single push
            target = sq + push_delta
            if 0 <= target < 64:
                target_bb = sq_to_bit(target)
                if (target_bb & (own_occ | opp_occ)) == 0:
                    is_promo = is_promo_rank(target, side)
                    if is_promo:
                        for p in range(4):
                            self.add_move(sq, target, p + 1)
                    else:
                        self.add_move(sq, target, 0)

                    # Double push
                    if start_rank:
                        double_target = sq + double_delta
                        if 0 <= double_target < 64:
                            target_bb = sq_to_bit(double_target)
                            if (target_bb & (own_occ | opp_occ)) == 0:
                                self.add_move(sq, double_target, 0)  # Never promo

            # West cap / EP
            target = sq + PAWN_CAP_WEST[side]
            if 0 <= target < 64:
                target_bb = sq_to_bit(target)
                attacker_rank_ok = (side == 0 and (sq // 8) == 4) or (side == 1 and (sq // 8) == 3)
                if ((target_bb & opp_occ) != 0) or (target == self.ep_square and attacker_rank_ok):
                    is_promo = is_promo_rank(target, side)
                    if is_promo:
                        for p in range(4):
                            self.add_move(sq, target, p + 1)
                    else:
                        self.add_move(sq, target, 0)

            # East cap / EP
            target = sq + PAWN_CAP_EAST[side]
            if 0 <= target < 64:
                target_bb = sq_to_bit(target)
                attacker_rank_ok = (side == 0 and (sq // 8) == 4) or (side == 1 and (sq // 8) == 3)
                if ((target_bb & opp_occ) != 0) or (target == self.ep_square and attacker_rank_ok):
                    is_promo = is_promo_rank(target, side)
                    if is_promo:
                        for p in range(4):
                            self.add_move(sq, target, p + 1)
                    else:
                        self.add_move(sq, target, 0)

        return self.move_count

    cdef int _generate_king_moves(self, uint64_t kings_bb, uint64_t own_occ, uint64_t opp_occ) nogil:
        cdef int sq, target
        cdef uint64_t target_bb, path_mask
        cdef bint is_white = self.white_to_move
        cdef uint8_t side = 0 if is_white else 1
        cdef int king_sq = -1

        # Find king (single)
        for sq in range(64):
            if sq_to_bit(sq) & kings_bb:
                king_sq = sq
                break

        # Normal king moves
        cdef int d
        for d in range(8):
            target = king_sq + KING_DELTAS[d]
            if 0 <= target < 64:
                target_bb = sq_to_bit(target)
                if (target_bb & own_occ) == 0:
                    self.add_move(king_sq, target, 0)

        # Castling (pseudo: rights + path empty)
        cdef uint8_t rights = self.castling
        if side == 0:  # White
            # KS
            if (rights & CASTLE_WK) and king_sq == 4 and (self.occupancy[2] & (sq_to_bit(5) | sq_to_bit(6))) == 0:
                self.add_move(4, 6, 0)
            # QS
            if (rights & CASTLE_WQ) and king_sq == 4 and (self.occupancy[2] & (sq_to_bit(1) | sq_to_bit(2))) == 0:
                self.add_move(4, 2, 0)
        else:  # Black
            # KS
            if (rights & CASTLE_BK) and king_sq == 60 and (self.occupancy[2] & (sq_to_bit(61) | sq_to_bit(62))) == 0:
                self.add_move(60, 62, 0)
            # QS
            if (rights & CASTLE_BQ) and king_sq == 60 and (self.occupancy[2] & (sq_to_bit(58) | sq_to_bit(59))) == 0:
                self.add_move(60, 58, 0)

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
                            self.add_move(sq, target)
                        break
                    else:
                        self.add_move(sq, target)
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
                            self.add_move(sq, target)
                        break
                    else:
                        self.add_move(sq, target)
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
                            self.add_move(sq, target)
                        break
                    else:
                        self.add_move(sq, target)
                    target += dir_delta
        return self.move_count

    cdef bint square_attacked(self, int sq, bint attacker_white) nogil:
        cdef uint64_t attackers = self.occupancy[0 if attacker_white else 1]
        cdef uint64_t all_occ = self.occupancy[2]
        cdef uint64_t attackers_bb, probe_bb
        cdef int probe_sq, delta, rev_delta
        cdef uint8_t att_side = 0 if attacker_white else 1

        # Pawns (rev caps)
        cdef int pawn_rev_west = -PAWN_CAP_WEST[att_side]
        cdef int pawn_rev_east = -PAWN_CAP_EAST[att_side]
        probe_sq = sq + pawn_rev_west
        if 0 <= probe_sq < 64:
            probe_bb = sq_to_bit(probe_sq)
            if probe_bb & self.pieces[PIECE_WP - 1 if attacker_white else PIECE_BP - 1]:
                return True
        probe_sq = sq + pawn_rev_east
        if 0 <= probe_sq < 64:
            probe_bb = sq_to_bit(probe_sq)
            if probe_bb & self.pieces[PIECE_WP - 1 if attacker_white else PIECE_BP - 1]:
                return True

        # Knights
        for delta in range(8):
            probe_sq = sq + KNIGHT_DELTAS[delta]
            if 0 <= probe_sq < 64:
                probe_bb = sq_to_bit(probe_sq)
                if probe_bb & self.pieces[PIECE_WN - 1 if attacker_white else PIECE_BN - 1]:
                    return True

        # King
        for delta in range(8):
            probe_sq = sq + KING_DELTAS[delta]
            if 0 <= probe_sq < 64:
                probe_bb = sq_to_bit(probe_sq)
                if probe_bb & self.pieces[PIECE_WK - 1 if attacker_white else PIECE_BK - 1]:
                    return True

        cdef int orig_file = sq % 8
        cdef int orig_rank = sq // 8
        cdef int pfile, prank, dfile, drank
        cdef uint64_t att_rook = (self.pieces[PIECE_WR - 1 if attacker_white else PIECE_BR - 1] |
                                self.pieces[PIECE_WQ - 1 if attacker_white else PIECE_BQ - 1])
        cdef int d

        # Rook ortho
        for d in range(4):
            rev_delta = -ROOK_DIRECTIONS[d]
            probe_sq = sq + rev_delta
            while 0 <= probe_sq < 64:
                pfile = probe_sq % 8
                prank = probe_sq // 8
                dfile = pfile - orig_file
                drank = prank - orig_rank
                if dfile != 0 and drank != 0:  # Not ortho
                    break
                probe_bb = sq_to_bit(probe_sq)
                if probe_bb & all_occ:
                    if probe_bb & att_rook:
                        return True
                    break
                probe_sq += rev_delta

        # Bishop diag
        cdef uint64_t att_bishop = (self.pieces[PIECE_WB - 1 if attacker_white else PIECE_BB - 1] |
                                    self.pieces[PIECE_WQ - 1 if attacker_white else PIECE_BQ - 1])
        for d in range(4):
            rev_delta = -BISHOP_DIRECTIONS[d]
            probe_sq = sq + rev_delta
            while 0 <= probe_sq < 64:
                pfile = probe_sq % 8
                prank = probe_sq // 8
                dfile = pfile - orig_file
                drank = prank - orig_rank
                if abs(dfile) != abs(drank):
                    break
                probe_bb = sq_to_bit(probe_sq)
                if probe_bb & all_occ:
                    if probe_bb & att_bishop:
                        return True
                    break
                probe_sq += rev_delta

        return False

    cdef bint _is_in_check(self, bint check_invalid = False) nogil:
        cdef bint defender_white = self.white_to_move ^ check_invalid
        cdef int ksq = self.find_king_sq(defender_white)
        cdef bint attacker_white = not defender_white
        return self.square_attacked(ksq, attacker_white)

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

        return self.move_count

    cdef int generate_legal_moves(self) nogil:
        cdef int pseudo_count = self.generate_pseudo_legal_moves()
        cdef Move temp_moves[256]
        cdef int legal_count = 0
        cdef int i
        for i in range(pseudo_count):
            temp_moves[i] = self.moves[i]
        for i in range(pseudo_count):
            if self._make_move(temp_moves[i].fr_sq, temp_moves[i].to_sq, temp_moves[i].promo):
                if not self._is_in_check(check_invalid=True):
                    self.moves[legal_count] = temp_moves[i]
                    legal_count += 1
                self._undo_move()
        self.move_count = legal_count
        return legal_count

    cpdef int generate_moves(self):
        return self.generate_legal_moves()

    cpdef list get_moves_list(self):
        self.generate_legal_moves()
        cdef list result = []
        cdef int i
        for i in range(self.move_count):
            result.append((self.moves[i].fr_sq, self.moves[i].to_sq, self.moves[i].promo))
        return result

    cdef void _set_piece(self, int sq, uint8_t piece_type) nogil:
        if sq < 0 or sq >= 64:
            return
        cdef uint64_t bit = sq_to_bit(sq)
        cdef int i
        # Clear sq from EVERYTHING
        for i in range(12):
            self.pieces[i] &= ~bit
        self.occupancy[0] &= ~bit
        self.occupancy[1] &= ~bit
        self.occupancy[2] &= ~bit
        # Set new
        if piece_type != PIECE_NONE:
            self.pieces[piece_type - 1] |= bit
            if piece_type <= 6:  # White
                self.occupancy[0] |= bit
            else:  # Black
                self.occupancy[1] |= bit
            self.occupancy[2] |= bit

    cpdef set_piece(self, int sq, uint8_t piece_type):
        return self._set_piece(sq, piece_type)

    cdef bint _make_move(self, uint8_t fr_sq_, uint8_t to_sq_, uint8_t promo_) nogil:
        cdef int fr_sq = fr_sq_, to_sq = to_sq_, promo = promo_
        if fr_sq < 0 or fr_sq >= 64 or to_sq < 0 or to_sq >= 64:
            return False
        if self.undo_index >= 1024:
            return False

        cdef uint64_t fr_bit = sq_to_bit(fr_sq)
        cdef uint64_t to_bit = sq_to_bit(to_sq)

        # Mover/cap types
        cdef uint8_t mover_type = PIECE_NONE
        cdef int i
        for i in range(12):
            if (self.pieces[i] & fr_bit):
                mover_type = <uint8_t>(i + 1)
                break
        if mover_type == PIECE_NONE:
            return False

        cdef uint8_t cap_type = PIECE_NONE
        for i in range(12):
            if (self.pieces[i] & to_bit):
                cap_type = <uint8_t>(i + 1)
                break

        cdef uint8_t mover_side = 0 if mover_type <= 6 else 1

        # Detect specials BEFORE changes
        cdef int8_t ep_captured_sq = -1
        cdef int8_t castle_rook_fr = -1
        cdef bint is_castle = False
        cdef bint is_ks = False
        if mover_type == PIECE_WP or mover_type == PIECE_BP:
            if self.ep_square == to_sq:
                ep_captured_sq = to_sq + (-8 if self.white_to_move else 8)
                if 0 <= ep_captured_sq < 64:
                    # Assume valid (gen ensures opp pawn there)
                    pass
        elif mover_type == PIECE_WK or mover_type == PIECE_BK:
            if abs(fr_sq - to_sq) == 2 and (fr_sq // 8) == (to_sq // 8):
                is_castle = True
                is_ks = to_sq > fr_sq
                if mover_side == 0:
                    castle_rook_fr = 7 if is_ks else 0
                else:
                    castle_rook_fr = 63 if is_ks else 56

        # Record undo (pre-change)
        cdef UndoState *undo = &self.undo_stack[self.undo_index]
        undo.fr_sq = fr_sq_
        undo.to_sq = to_sq_
        undo.mover_type = mover_type
        undo.cap_type = cap_type
        undo.promo = promo
        undo.castling = self.castling
        undo.ep_square = self.ep_square
        undo.ep_captured_sq = ep_captured_sq
        undo.castle_rook_fr = castle_rook_fr
        undo.halfmove = self.halfmove
        undo.fullmove = self.fullmove
        self.undo_index += 1

        # Clear fr
        self.pieces[mover_type - 1] &= ~fr_bit
        self.occupancy[mover_side] &= ~fr_bit
        self.occupancy[2] &= ~fr_bit

        # Clear to cap (if any)
        cdef uint8_t cap_side
        if cap_type != PIECE_NONE:
            self.pieces[cap_type - 1] &= ~to_bit
            cap_side = 0 if cap_type <= 6 else 1
            self.occupancy[cap_side] &= ~to_bit
            self.occupancy[2] &= ~to_bit

        # EP extra clear (empty to already)
        cdef uint64_t ep_bit
        cdef uint8_t ep_cap_type
        cdef uint8_t ep_side
        if ep_captured_sq != -1:
            ep_bit = sq_to_bit(ep_captured_sq)
            ep_cap_type = PIECE_BP if self.white_to_move else PIECE_WP
            self.pieces[ep_cap_type - 1] &= ~ep_bit
            ep_side = 1 - mover_side
            self.occupancy[ep_side] &= ~ep_bit
            self.occupancy[2] &= ~ep_bit

        # Castle rook move
        cdef int rook_to
        cdef uint64_t rfr_bit, rto_bit
        cdef uint8_t rook_type
        if castle_rook_fr != -1:
            rook_to = to_sq - (1 if to_sq > fr_sq else -1)
            rfr_bit = sq_to_bit(castle_rook_fr)
            rto_bit = sq_to_bit(rook_to)
            rook_type = PIECE_WR if mover_side == 0 else PIECE_BR
            self.pieces[rook_type - 1] &= ~rfr_bit
            self.occupancy[mover_side] &= ~rfr_bit
            self.occupancy[2] &= ~rfr_bit
            self.pieces[rook_type - 1] |= rto_bit
            self.occupancy[mover_side] |= rto_bit
            self.occupancy[2] |= rto_bit

        # Place (pawn) at to
        self.pieces[mover_type - 1] |= to_bit
        self.occupancy[mover_side] |= to_bit
        self.occupancy[2] |= to_bit

        # Promo override
        cdef uint8_t new_type
        if promo != 0:
            new_type = PROMO_PIECES[mover_side][promo - 1]
            self._set_piece(to_sq, new_type)

        # State updates
        self.white_to_move = not self.white_to_move

        # Halfmove (reset on pawn/cap)
        if mover_type == PIECE_WP or mover_type == PIECE_BP or cap_type != PIECE_NONE:
            self.halfmove = 0
        else:
            self.halfmove += 1

        # Fullmove
        if not self.white_to_move:
            self.fullmove += 1

        # EP (double push only)
        self.ep_square = -1
        if (mover_type == PIECE_WP or mover_type == PIECE_BP) and abs(to_sq - fr_sq) == 16:
            self.ep_square = fr_sq + PAWN_PUSH[mover_side]

        # Castling rights (recompute from positions)
        self.castling = 0
        if (self.pieces[PIECE_WK - 1] & sq_to_bit(4)) and (self.pieces[PIECE_WR - 1] & sq_to_bit(7)):
            self.castling |= CASTLE_WK
        if (self.pieces[PIECE_WK - 1] & sq_to_bit(4)) and (self.pieces[PIECE_WR - 1] & sq_to_bit(0)):
            self.castling |= CASTLE_WQ
        if (self.pieces[PIECE_BK - 1] & sq_to_bit(60)) and (self.pieces[PIECE_BR - 1] & sq_to_bit(63)):
            self.castling |= CASTLE_BK
        if (self.pieces[PIECE_BK - 1] & sq_to_bit(60)) and (self.pieces[PIECE_BR - 1] & sq_to_bit(56)):
            self.castling |= CASTLE_BQ

        self._update_occupancy()
        return True

    cpdef bint make_move(self, int fr_sq, int to_sq, uint8_t promo=0):
        return self._make_move(<uint8_t>fr_sq, <uint8_t>to_sq, promo)

    cdef void _undo_move(self) nogil:
        if self.undo_index <= 0:
            return
        self.undo_index -= 1
        cdef UndoState undo = self.undo_stack[self.undo_index]

        cdef uint64_t fr_bit = sq_to_bit(undo.fr_sq)
        cdef uint64_t to_bit = sq_to_bit(undo.to_sq)

        cdef uint8_t mover_side = 0 if undo.mover_type <= 6 else 1

        # Clear to
        self.pieces[undo.mover_type - 1] &= ~to_bit  # Undo promo/mover
        self.occupancy[mover_side] &= ~to_bit
        self.occupancy[2] &= ~to_bit

        # Restore cap at to
        cdef uint8_t cap_side
        if undo.cap_type != PIECE_NONE:
            self.pieces[undo.cap_type - 1] |= to_bit
            cap_side = 0 if undo.cap_type <= 6 else 1
            self.occupancy[cap_side] |= to_bit
            self.occupancy[2] |= to_bit

        # Restore EP captured
        cdef uint64_t ep_bit
        cdef uint8_t ep_cap_type
        cdef uint8_t ep_side
        if undo.ep_captured_sq != -1:
            ep_bit = sq_to_bit(undo.ep_captured_sq)
            ep_cap_type = (PIECE_BP if undo.mover_type <= 6 else PIECE_WP)
            self.pieces[ep_cap_type - 1] |= ep_bit
            ep_side = 1 - mover_side
            self.occupancy[ep_side] |= ep_bit
            self.occupancy[2] |= ep_bit

        # Restore castle rook
        cdef bint is_ks
        cdef int rook_to
        cdef uint64_t rfr_bit, rto_bit
        cdef uint8_t rook_type
        if undo.castle_rook_fr != -1:
            is_ks = undo.to_sq > undo.fr_sq
            rook_to = undo.to_sq - (1 if is_ks else -1)
            rfr_bit = sq_to_bit(undo.castle_rook_fr)
            rto_bit = sq_to_bit(rook_to)
            if undo.mover_type == PIECE_WK:
                rook_type = PIECE_WR
            elif undo.mover_type == PIECE_BK:
                rook_type = PIECE_BK
            else:
                rook_type = 0

            # Clear rook_to
            self.pieces[rook_type - 1] &= ~rto_bit
            self.occupancy[mover_side] &= ~rto_bit
            self.occupancy[2] &= ~rto_bit
            # Place at rook_fr
            self.pieces[rook_type - 1] |= rfr_bit
            self.occupancy[mover_side] |= rfr_bit
            self.occupancy[2] |= rfr_bit

        # Restore mover at fr
        self.pieces[undo.mover_type - 1] |= fr_bit
        self.occupancy[mover_side] |= fr_bit
        self.occupancy[2] |= fr_bit

        # Restore state
        self.castling = undo.castling
        self.ep_square = undo.ep_square
        self.halfmove = undo.halfmove
        self.fullmove = undo.fullmove
        self.white_to_move = not self.white_to_move

        self._update_occupancy()

    cpdef void undo_move(self):
        self._undo_move()

    cpdef long long perft(self, int depth):
        if depth == 0:
            return 1
        self.generate_legal_moves()
        
        cdef int j
        for j in range(self.move_count):
            self.move_cache[depth][j] = self.moves[j]
        self.move_count_cache[depth] = self.move_count
        cdef long long nodes = 0
        cdef int i
        for i in range(self.move_count_cache[depth]):
            if self._make_move(self.move_cache[depth][i].fr_sq, self.move_cache[depth][i].to_sq, self.move_cache[depth][i].promo):
                nodes += self.perft(depth - 1)
                self._undo_move()
        return nodes