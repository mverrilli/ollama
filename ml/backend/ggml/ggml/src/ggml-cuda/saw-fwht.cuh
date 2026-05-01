#pragma once

// Randomized Walsh-Hadamard Transform (FWHT) for SAW-INT KV cache quantization.
//
// Forward  F(x)   = normalize(WHT(sign_flip(x)))    // sign → butterfly → normalize
// Inverse  F⁻¹(y) = sign_flip(normalize(WHT(y)))    // butterfly → normalize → sign flip
//
// F is orthogonal: F(Q)·F(K)^T = Q·K^T.
// Self-inverse property: H_norm² = I  (normalized WHT is involutory).
//
// signLo / signHi: 128 sign bits as two uint64_t.
//   Bit i of signLo (0≤i<64): sign for element i
//   Bit j of signHi (0≤j<64): sign for element j+64
//   0 → +1,  1 → −1
//
// All functions operate on an array s[] in shared memory of length headDim.
// tid is the thread's flat index (0 .. nthreads-1, where nthreads >= headDim/2).
// Caller is responsible for initial __syncthreads() before calling and after
// the function returns (though each function ends with __syncthreads() internally).

// In-place sign flip: element tid gets ±1 factor from the sign bits.
__device__ static __forceinline__ void saw_sign_flip(
        float * __restrict__ s, int headDim,
        uint64_t signLo, uint64_t signHi, int tid) {
    if (tid < headDim) {
        uint64_t word = (tid < 64) ? signLo : signHi;
        int bit = (int)((word >> (tid & 63)) & 1ULL);
        if (bit) s[tid] = -s[tid];
    }
}

// In-place WHT butterfly.  Requires all threads to be present (no divergent
// __syncthreads).  Threads with tid >= headDim/2 are idle during each stage.
__device__ static __forceinline__ void saw_wht_butterfly(
        float * __restrict__ s, int headDim, int tid) {
    for (int stride = 1; stride < headDim; stride <<= 1) {
        if (tid < headDim / 2) {
            const int lo = (tid / stride) * 2 * stride + (tid % stride);
            const int hi = lo + stride;
            const float a = s[lo], b = s[hi];
            s[lo] = a + b;
            s[hi] = a - b;
        }
        __syncthreads();
    }
}

// Forward FWHT: sign_flip → butterfly → normalize.
// On return, s[] contains F(original_s).
__device__ static __forceinline__ void saw_fwht_forward(
        float * __restrict__ s, int headDim,
        uint64_t signLo, uint64_t signHi, int tid) {
    saw_sign_flip(s, headDim, signLo, signHi, tid);
    __syncthreads();
    saw_wht_butterfly(s, headDim, tid);
    // Normalize by 1/sqrt(headDim); all threads active
    if (tid < headDim) {
        s[tid] *= rsqrtf((float)headDim);
    }
    __syncthreads();
}

// Multi-vector forward FWHT: applies saw_fwht_forward to nvec consecutive
// D-element vectors packed as s[0..nvec*D), sharing butterfly barriers across
// all vectors instead of serialising them.
//
// Sync count: nvec*(log2(D)+2) → log2(D)+2.
// For D=128, nvec=4 (prefill ncols): 36 syncs → 9.
// For nvec=1 (decode): identical to calling saw_fwht_forward once.
//
// Correctness: each butterfly stage accesses only s[j*D .. (j+1)*D) for its
// vector j; the inner-loop accesses are disjoint across j. The shared sync
// after each stage ensures all threads finish stage k for all vectors before
// any thread starts stage k+1.
template<int nvec>
__device__ static __forceinline__ void saw_fwht_forward_multi(
        float * __restrict__ s, int D,
        uint64_t signLo, uint64_t signHi, int tid) {
    // Sign flip all vectors (no sync needed before butterfly).
    #pragma unroll
    for (int j = 0; j < nvec; ++j) {
        saw_sign_flip(s + j * D, D, signLo, signHi, tid);
    }
    __syncthreads();

    // Butterfly: one sync per stage, shared across all nvec vectors.
    for (int stride = 1; stride < D; stride <<= 1) {
        if (tid < D / 2) {
            const int lo = (tid / stride) * 2 * stride + (tid % stride);
            const int hi = lo + stride;
            #pragma unroll
            for (int j = 0; j < nvec; ++j) {
                const float a = s[j * D + lo], b = s[j * D + hi];
                s[j * D + lo] = a + b;
                s[j * D + hi] = a - b;
            }
        }
        __syncthreads();
    }

    // Normalize all vectors; one shared sync at the end.
    const float norm = rsqrtf((float)D);
    #pragma unroll
    for (int j = 0; j < nvec; ++j) {
        if (tid < D) s[j * D + tid] *= norm;
    }
    __syncthreads();
}

// Inverse FWHT: butterfly → normalize → sign_flip.
// On return, s[] contains F⁻¹(original_s).
__device__ static __forceinline__ void saw_fwht_inverse(
        float * __restrict__ s, int headDim,
        uint64_t signLo, uint64_t signHi, int tid) {
    saw_wht_butterfly(s, headDim, tid);
    if (tid < headDim) {
        s[tid] *= rsqrtf((float)headDim);
    }
    __syncthreads();
    saw_sign_flip(s, headDim, signLo, signHi, tid);
    __syncthreads();
}
