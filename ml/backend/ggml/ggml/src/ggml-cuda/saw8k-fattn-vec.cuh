#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "saw-fwht.cuh"

// Fused flash-attention for saw8k: int8 K (per-group asymmetric) + f16 V.
// Q is rotated forward by FWHT before the Q·K dot product, preserving the
// exact attention scores that would result from unrotated f16 K values:
//   F(Q) · F(K)^T = Q · K^T   (F orthogonal)
//
// Decode:  k[d] = packed[cell*nKVH*D + head*D + d] * scale[d/G] + min[d/G]  (G=32)
// Buffer layout (matches saw8k-encode.cu / q8k-encode.cu):
//   packed  [(cell*nKVH+head)*D + d]                   u8
//   scales  [cell*nG*nKVH + head*nG + g]               f16
//   mins    [same]                                      f16
//
// op_params: [scale(f32), softcap(f32), firstCell, nKVHeads, nCells,
//             signLo_lo32, signLo_hi32, signHi_lo32, signHi_hi32]
//
// Grid:  (ceil(nTokensQ/ncols), 1, nHeadsQ * nSeq)
// Block: (WARP_SIZE, 4) = 128 threads
// D=128 only.

#define SAW8K_FA_GROUP_SIZE 32

#if __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)

template<int D, int ncols, bool use_logit_softcap>
__launch_bounds__(128, 2)
static __global__ void saw8k_flash_attn_ext_vec(
    const char     * __restrict__ Q,
    const uint8_t  * __restrict__ K_packed,
    const char     * __restrict__ V,
    const char     * __restrict__ mask,
    float          * __restrict__ dst,
    const uint16_t * __restrict__ k_scales,
    const uint16_t * __restrict__ k_mins,
    float   scale,
    float   logit_softcap,
    int     firstCell,
    int     nCells,
    int     nKVHeads,
    uint64_t signLo,
    uint64_t signHi,
    int32_t ne00,
    uint3   ne01,
    int32_t ne02,
    int32_t ne03,
    int32_t nb01,
    int32_t nb02,
    int64_t nb03,
    int32_t nb21,
    int32_t nb22,
    int64_t nb23,
    int32_t ne31,
    int32_t nb31
)
{
#ifdef FLASH_ATTN_AVAILABLE
    if (use_logit_softcap && D != 128) { NO_DEVICE_CODE; return; }

    constexpr int nthreads    = 128;
    constexpr int cpy_nb      = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne      = cpy_nb / 4;
    constexpr int nthreads_KQ = nthreads / cpy_nb;
    constexpr int nthreads_V  = nthreads_KQ;
    constexpr int V_rows_per_thread = 2 * cpy_ne;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;
    constexpr int nwarps      = nthreads / WARP_SIZE;
    constexpr int numGroups   = D / SAW8K_FA_GROUP_SIZE;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_KQ");
    static_assert(D % (2 * WARP_SIZE) == 0, "D not divisible by 64");

    constexpr dequantize_V_t dequantize_V =
        get_dequantize_V<GGML_TYPE_F16, float, V_rows_per_thread>();

    const int ic0      = blockIdx.x * ncols;
    const int sequence = blockIdx.z / ne02;
    const int head     = blockIdx.z % ne02;
    const int gqa_ratio = ne02 / nKVHeads;
    const int head_kv   = head / gqa_ratio;

    Q += (int64_t)nb03 * sequence + (int64_t)nb02 * head + (int64_t)nb01 * ic0;
    V += (int64_t)nb23 * sequence + (int64_t)nb22 * head_kv;

    K_packed += (int64_t)firstCell * nKVHeads * D + (int64_t)head_kv * D;
    k_scales += (int64_t)firstCell * numGroups * nKVHeads + (int64_t)head_kv * numGroups;
    k_mins   += (int64_t)firstCell * numGroups * nKVHeads + (int64_t)head_kv * numGroups;

    const half * maskh = mask ? (const half *)(mask + (int64_t)nb31 * ic0) : nullptr;

    constexpr int ne_KQ      = ncols * nthreads;
    constexpr int ne_combine = nwarps * V_cols_per_iter * D;

    float2 VKQ[ncols][(D/2) / nthreads_V] = {{{0.0f, 0.0f}}};

    extern __shared__ float s_mem_all[];
    float * KQ        = s_mem_all;
    float * s_Q_fixed = s_mem_all + (ne_KQ > ne_combine ? ne_KQ : ne_combine);

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;

    // Load Q into shared memory, pre-scaled.
    for (int i = tid; i < ncols * D; i += nthreads) {
        const int head_q = i / D;
        const int elem   = i % D;
        if (head_q < ncols) {
            const float * Q_ptr = (const float *)(Q + (int64_t)head_q * nb01);
            s_Q_fixed[i] = Q_ptr[elem] * scale;
        }
    }
    __syncthreads();

    // Apply forward FWHT to all ncols Q vectors simultaneously, sharing the
    // per-stage butterfly barriers.  Saves (ncols-1)*(log2(D)+2) syncs vs
    // calling saw_fwht_forward in a loop (27 fewer barriers for ncols=4, D=128).
    saw_fwht_forward_multi<ncols>(s_Q_fixed, D, signLo, signHi, tid);
    // Ends with __syncthreads(); s_Q_fixed now holds F(Q)*scale.

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX / 2.0f;
        KQ_sum[j] = 0.0f;
    }

    const int tid_kq = threadIdx.x % nthreads_KQ;

    float2 Q_reg[ncols][(D/2) / nthreads_KQ];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int k = 0; k < (D/2) / nthreads_KQ; ++k) {
            const int i = k * nthreads_KQ + tid_kq;
            Q_reg[j][k].x = s_Q_fixed[j * D + 2*i];
            Q_reg[j][k].y = s_Q_fixed[j * D + 2*i + 1];
        }
    }

    for (int k_VKQ_0 = 0; k_VKQ_0 < nCells;
             k_VKQ_0 += nthreads,
             V += (int64_t)nthreads * nb21) {

        float KQ_reg[ncols];
        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y * WARP_SIZE
                           + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ - 1)))
                           + i_KQ_0;
            const int cell_rel = k_VKQ_0 + i_KQ;
            const bool in_range = (cell_rel < nCells);

            const uint8_t * packed_row = in_range
                ? K_packed + (int64_t)cell_rel * nKVHeads * D : nullptr;

            float k_s[numGroups];
            float k_m[numGroups];
            if (in_range) {
                const int meta_stride = numGroups * nKVHeads;
#pragma unroll
                for (int g = 0; g < numGroups; ++g) {
                    const int meta_idx = cell_rel * meta_stride + g;
                    k_s[g] = __half2float(__ushort_as_half(k_scales[meta_idx]));
                    k_m[g] = __half2float(__ushort_as_half(k_mins[meta_idx]));
                }
            }

            // Q-tile decode amortization: hoist K decode out of the `j` loop.
            float2 k_lane[(D/2) / nthreads_KQ];
            if (in_range) {
#pragma unroll
                for (int k = 0; k < (D/2) / nthreads_KQ; ++k) {
                    const int d0 = 2 * (k * nthreads_KQ + tid_kq);
                    const int g0 = d0 / SAW8K_FA_GROUP_SIZE;
                    const int g1 = (d0 + 1) / SAW8K_FA_GROUP_SIZE;
                    k_lane[k].x = (float)packed_row[d0]     * k_s[g0] + k_m[g0];
                    k_lane[k].y = (float)packed_row[d0 + 1] * k_s[g1] + k_m[g1];
                }
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float mask_val = 0.0f;
                if (maskh && in_range) {
                    mask_val = (cell_rel < ne31)
                        ? __half2float(maskh[(int64_t)j * ne31 + cell_rel]) : -FLT_MAX / 2.0f;
                }
                float sum = 0.0f;
                if (in_range) {
#pragma unroll
                    for (int k = 0; k < (D/2) / nthreads_KQ; ++k) {
                        sum += Q_reg[j][k].x * k_lane[k].x + Q_reg[j][k].y * k_lane[k].y;
                    }
                }
                for (int offset = nthreads_KQ / 2; offset > 0; offset >>= 1) {
                    sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
                }
                sum += mask_val;
                if (use_logit_softcap) {
                    sum = logit_softcap * tanhf(sum / logit_softcap);
                }
                if (!in_range) sum = -FLT_MAX / 2.0f;
                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);
                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)
                        == (uint32_t)i_KQ_0) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j],
                    __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];
            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j] * KQ_max_scale + KQ_reg[j];
            KQ[j * nthreads + tid] = KQ_reg[j];
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0 / nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0 / nthreads_V].y *= KQ_max_scale;
            }
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y * WARP_SIZE + k0
                        + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j * nthreads + k];
            }

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V * V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + (int64_t)k * nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)
                              * V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x * KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y * KQ_k[j];
                    }
                }
            }
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX / 2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= (int)ne01.z) { break; }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

        float2 * VKQ_tmp = (float2 *)KQ + threadIdx.y * (V_cols_per_iter * D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V) * (D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V * V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0
                + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)
                  * (V_rows_per_thread/2);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,
                                &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4,
                                &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }
        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w * V_cols_per_iter * D + v * D + i0 + tid]);
                    }
                }
                dst_val /= KQ_sum[j_VKQ];
                dst[(((int64_t)sequence * (int)ne01.z + ic0 + j_VKQ) * ne02 + head) * D + i0 + tid] = dst_val;
            }
        }
        if (j_VKQ < ncols - 1) { __syncthreads(); }
    }

#else
    GGML_UNUSED_VARS(Q, K_packed, V, mask, dst, k_scales, k_mins,
        scale, logit_softcap, firstCell, nCells, nKVHeads, signLo, signHi,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        nb21, nb22, nb23, ne31, nb31);
    NO_DEVICE_CODE;
#endif
}

#else

template<int D, int ncols, bool use_logit_softcap>
static __global__ void saw8k_flash_attn_ext_vec(
    const char *, const uint8_t *, const char *, const char *, float *,
    const uint16_t *, const uint16_t *,
    float, float, int, int, int, uint64_t, uint64_t,
    int32_t, uint3, int32_t, int32_t, int32_t, int32_t, int64_t,
    int32_t, int32_t, int64_t, int32_t, int32_t) {}

#endif
