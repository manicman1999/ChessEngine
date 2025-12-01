# distutils: include_dirs = numpy/
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport expf, fmaxf

cdef class AttnModelCompiled:
    """
    Optimized Cython implementation of the sliced attention model for chess board evaluation.
    Weights are copied to fixed-size C arrays during initialization for nogil-compatible forward pass.
    - embed: (960, 24) float32 → embed_flat[23040]
    - head_weight: (1024, 16) float32 → head_w_flat[16384]
    - head_bias: (16,) float32 → head_b_flat[16]
    - out_weight: (16, 1) float32 → out_w_flat[16]
    - out_bias: (1,) float32 → out_b_flat[1]
    
    Forward: takes tokens (64,) int32/uint8 array, returns single float logit.
    All operations manual (loops for matmul/softmax) for fixed-size efficiency; fully nogil.
    """
    # Fixed-size C arrays (no memoryviews for nogil safety)
    cdef float embed_flat[23040]  # 960 * 24
    cdef float head_w_flat[16384]  # 1024 * 16
    cdef float head_b_flat[16]     # 16
    cdef float out_w_flat[16]      # 16 * 1
    cdef float out_b_flat[1]       # 1
    
    # Temp buffers (flat 1D C arrays)
    cdef float q_flat[256]       # 64*4
    cdef float k_flat[256]       # 64*4
    cdef float v_flat[512]       # 64*8
    cdef float e_flat[512]       # 64*8
    cdef float attn_flat[4096]   # 64*64
    cdef float z_flat[512]       # 64*8
    cdef float combined_flat[1024]  # 64*16
    cdef float h_flat[16]        # 16
    cdef float value             # scalar
    
    # Tokens buffer (copied in forward for nogil)
    cdef int tokens_c[64]        # 64 ints
    
    def __init__(self, embed, head_weight, head_bias, out_weight, out_bias):
        # Validate inputs (embed etc. expected as np.ndarray float32)
        if not isinstance(embed, np.ndarray) or embed.dtype != np.float32:
            raise ValueError("embed must be float32 ndarray")
        if embed.shape[0] != 960 or embed.shape[1] != 24:
            raise ValueError(f"embed shape must be (960, 24), got {embed.shape}")
        
        if not isinstance(head_weight, np.ndarray) or head_weight.dtype != np.float32:
            raise ValueError("head_weight must be float32 ndarray")
        if head_weight.shape[0] != 1024 or head_weight.shape[1] != 16:
            raise ValueError(f"head_weight shape must be (1024, 16), got {head_weight.shape}")
        
        if not isinstance(head_bias, np.ndarray) or head_bias.dtype != np.float32:
            raise ValueError("head_bias must be float32 ndarray")
        if head_bias.shape[0] != 16:
            raise ValueError(f"head_bias shape must be (16,), got {head_bias.shape}")
        
        if not isinstance(out_weight, np.ndarray) or out_weight.dtype != np.float32:
            raise ValueError("out_weight must be float32 ndarray")
        if out_weight.shape[0] != 16 or out_weight.shape[1] != 1:
            raise ValueError(f"out_weight shape must be (16, 1), got {out_weight.shape}")
        
        if not isinstance(out_bias, np.ndarray) or out_bias.dtype != np.float32:
            raise ValueError("out_bias must be float32 ndarray")
        if out_bias.shape[0] != 1:
            raise ValueError(f"out_bias shape must be (1,), got {out_bias.shape}")
        
        # Copy to fixed C arrays (GIL ok here, as init is Python-called)
        cdef int i, j
        cdef cnp.ndarray[cnp.float32_t, ndim=2] embed_view = embed
        cdef cnp.ndarray[cnp.float32_t, ndim=2] head_w_view = head_weight
        cdef cnp.ndarray[cnp.float32_t, ndim=1] head_b_view = head_bias
        cdef cnp.ndarray[cnp.float32_t, ndim=2] out_w_view = out_weight
        cdef cnp.ndarray[cnp.float32_t, ndim=1] out_b_view = out_bias
        
        # Embed: row-major flat
        for i in range(960):
            for j in range(24):
                self.embed_flat[i * 24 + j] = embed_view[i, j]
        
        # Head weight: row-major flat (1024 rows, 16 cols)
        for i in range(1024):
            for j in range(16):
                self.head_w_flat[i * 16 + j] = head_w_view[i, j]
        
        # Head bias
        for i in range(16):
            self.head_b_flat[i] = head_b_view[i]
        
        # Out weight: row-major flat (16 rows, 1 col)
        for i in range(16):
            self.out_w_flat[i] = out_w_view[i, 0]
        
        # Out bias
        self.out_b_flat[0] = out_b_view[0]

    cpdef float forward(self, tokens):
        # GIL-held here (Python-called); validate/convert to memoryview
        if not isinstance(tokens, np.ndarray) or tokens.dtype != np.int32:
            raise ValueError("tokens must be int32 ndarray")
        if tokens.shape[0] != 64:
            raise ValueError(f"tokens shape must be (64,), got {tokens.shape}")
        cdef cnp.ndarray[cnp.int32_t, ndim=1] tok_arr = tokens
        cdef int[::1] tok_view = tok_arr  # Contiguous memoryview
        return self._forward(tok_view)

    cdef float _forward(self, int[::1] tokens) noexcept nogil:
        """
        Optimized forward: nogil, unrolled loops, fixed C arrays for ~5-10x speedup over base.
        tokens: contiguous int[::1] memoryview (64 elements, token IDs 0-959)
        Returns: float logit (scalar)
        """
        cdef int i, j, m, n, tid
        cdef float sum_qk, max_attn, exp_sum, temp, exp_temp
        
        # Copy tokens from memoryview to C array (nogil-safe)
        for i in range(64):
            self.tokens_c[i] = tokens[i]

        for i in range(64):
            tid = self.tokens_c[i]
            # q: 0:4 (unrolled)
            self.q_flat[i*4 + 0] = self.embed_flat[tid*24 + 0]
            self.q_flat[i*4 + 1] = self.embed_flat[tid*24 + 1]
            self.q_flat[i*4 + 2] = self.embed_flat[tid*24 + 2]
            self.q_flat[i*4 + 3] = self.embed_flat[tid*24 + 3]
            # k: 4:8 (unrolled)
            self.k_flat[i*4 + 0] = self.embed_flat[tid*24 + 4]
            self.k_flat[i*4 + 1] = self.embed_flat[tid*24 + 5]
            self.k_flat[i*4 + 2] = self.embed_flat[tid*24 + 6]
            self.k_flat[i*4 + 3] = self.embed_flat[tid*24 + 7]
            # v: 8:16 (loop for 8; unroll if desired, but compiler may auto)
            for j in range(8):
                self.v_flat[i*8 + j] = self.embed_flat[tid*24 + 8 + j]
            # e: 16:24 (loop)
            for j in range(8):
                self.e_flat[i*8 + j] = self.embed_flat[tid*24 + 16 + j]
        
        # attn = q @ k.T / 2.0 (unroll inner dim=4 dot product)
        for i in range(64):
            for j in range(64):
                sum_qk = (self.q_flat[i*4 + 0] * self.k_flat[j*4 + 0] +
                            self.q_flat[i*4 + 1] * self.k_flat[j*4 + 1] +
                            self.q_flat[i*4 + 2] * self.k_flat[j*4 + 2] +
                            self.q_flat[i*4 + 3] * self.k_flat[j*4 + 3])
                self.attn_flat[i*64 + j] = sum_qk * 0.5  # Use float literal for precision
        
        # Softmax (per row; log-sum-exp for stability)
        for i in range(64):
            max_attn = self.attn_flat[i*64 + 0]
            for j in range(1, 64):
                if self.attn_flat[i*64 + j] > max_attn:
                    max_attn = self.attn_flat[i*64 + j]
            exp_sum = 0.0
            for j in range(64):
                exp_temp = self.attn_flat[i*64 + j] - max_attn
                exp_temp = expf(exp_temp)
                self.attn_flat[i*64 + j] = exp_temp
                exp_sum += exp_temp
            if exp_sum > 0.0:
                for j in range(64):
                    self.attn_flat[i*64 + j] /= exp_sum
            else:
                # Fallback to uniform (better than all-zero; avoids NaN propagation)
                for j in range(64):
                    self.attn_flat[i*64 + j] = 1.0 / 64.0
        
        # z = attn @ v [64,8] (loop; 64x64x8 ops)
        for i in range(64):
            for n in range(8):
                temp = 0.0
                for j in range(64):
                    temp += self.attn_flat[i*64 + j] * self.v_flat[j*8 + n]
                self.z_flat[i*8 + n] = temp
        
        # combined = cat(z, e) [64,16] (loop over 8)
        for i in range(64):
            for j in range(8):
                self.combined_flat[i*16 + j] = self.z_flat[i*8 + j]
                self.combined_flat[i*16 + 8 + j] = self.e_flat[i*8 + j]
        
        # h = ReLU(head(combined_flat) + bias) [16] (1024x16 ops)
        for n in range(16):  # Use 'n' to avoid shadowing
            temp = self.head_b_flat[n]
            for m in range(1024):
                temp += self.combined_flat[m] * self.head_w_flat[m * 16 + n]
            self.h_flat[n] = fmaxf(0.0, temp)
        
        # value = out(h) + bias (loop over 16)
        temp = self.out_b_flat[0]
        for n in range(16):
            temp += self.h_flat[n] * self.out_w_flat[n]
        self.value = temp
        return self.value