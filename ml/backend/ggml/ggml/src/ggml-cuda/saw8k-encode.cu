#include "saw8k-encode.cuh"
#include "saw-fwht.cuh"
#include <math.h>

// Per-group asymmetric int8 encode with FWHT rotation preprocessing (saw8k preset).
//
// Pipeline:
//   1. Load element into shared float array s_k[headDim]
//   2. Apply forward randomized FWHT to s_k (sign flip → butterfly → normalize)
//   3. Warp-level min/max reduction over each group of 32 rotated elements
//   4. scale = (max-min)/255, quantize to [0,255], store byte
//
// Buffer layout (identical to q8k-encode.cu):
//   packed  [(cell*numKVHeads+head)*headDim + d]            u8
//   scales  [cell*numGroups*numKVHeads + head*numGroups + g] f16
//   mins    [same]                                           f16
//
// Grid: (batchSize, numKVHeads).  Block: headDim threads.
// op_params: [firstCell, signLo_lo32, signLo_hi32, signHi_lo32, signHi_hi32]

#define SAW8K_GROUP_SIZE 32

#if __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)
__global__ void saw8k_encode_kernel(
    const void     * __restrict__ k,
    uint8_t        * __restrict__ packed,
    uint16_t       * __restrict__ scales,
    uint16_t       * __restrict__ mins,
    int   firstCell,
    int   headDim,
    int   numKVHeads,
    int   kIsF32,
    uint64_t signLo,
    uint64_t signHi
) {
    const int batch = blockIdx.x;
    const int head  = blockIdx.y;
    const int cell  = firstCell + batch;
    const int t     = threadIdx.x;
    const int g     = t / SAW8K_GROUP_SIZE;
    const int j     = t % SAW8K_GROUP_SIZE;
    const int numGroups = headDim / SAW8K_GROUP_SIZE;

    extern __shared__ float s_k[];  // headDim floats

    // Load element from K buffer into shared float array.
    const int kidx = batch * numKVHeads * headDim + head * headDim + t;
    s_k[t] = kIsF32
        ? ((const float   *)k)[kidx]
        : __half2float(__ushort_as_half(((const uint16_t *)k)[kidx]));
    __syncthreads();

    // Apply forward FWHT to s_k (operates on all headDim elements).
    saw_fwht_forward(s_k, headDim, signLo, signHi, t);
    // saw_fwht_forward ends with __syncthreads(), so s_k is ready.

    // Warp-level min/max over this thread's group (32 lanes = one warp).
    float x = s_k[t];
    float min_val = x, max_val = x;
    for (int offset = SAW8K_GROUP_SIZE >> 1; offset > 0; offset >>= 1) {
        min_val = fminf(min_val, __shfl_xor_sync(0xffffffff, min_val, offset, SAW8K_GROUP_SIZE));
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset, SAW8K_GROUP_SIZE));
    }

    if (j == 0) {
        const float scale = (max_val - min_val) / 255.0f;
        const int meta_base = cell * numGroups * numKVHeads + head * numGroups;
        scales[meta_base + g] = __half_as_ushort(__float2half(scale));
        mins  [meta_base + g] = __half_as_ushort(__float2half(min_val));
    }

    const float scale     = (max_val - min_val) / 255.0f;
    const float inv_scale = (scale > 1e-8f) ? (1.0f / scale) : 0.0f;
    const int q = max(0, min(255, (int)roundf((x - min_val) * inv_scale)));

    const int packed_base = (cell * numKVHeads + head) * headDim;
    packed[packed_base + t] = (uint8_t)q;
}
#else
__global__ void saw8k_encode_kernel(
    const void *, uint8_t *, uint16_t *, uint16_t *,
    int, int, int, int, uint64_t, uint64_t) {}
#endif

void ggml_cuda_saw8k_encode(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * k      = dst->src[0];
    const struct ggml_tensor * scales = dst->src[1];
    const struct ggml_tensor * mins_t = dst->src[2];

    const int headDim    = (int)k->ne[0];
    const int numKVHeads = (int)k->ne[1];
    const int batchSize  = (int)k->ne[2];
    const int firstCell  = ggml_get_op_params_i32(dst, 0);
    const uint64_t signLo = ((uint64_t)(uint32_t)ggml_get_op_params_i32(dst, 1)) |
                            ((uint64_t)(uint32_t)ggml_get_op_params_i32(dst, 2) << 32);
    const uint64_t signHi = ((uint64_t)(uint32_t)ggml_get_op_params_i32(dst, 3)) |
                            ((uint64_t)(uint32_t)ggml_get_op_params_i32(dst, 4) << 32);
    const int kIsF32 = (k->type == GGML_TYPE_F32) ? 1 : 0;

    GGML_ASSERT(headDim % SAW8K_GROUP_SIZE == 0);
    GGML_ASSERT(headDim <= 512);
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(k->data != nullptr && dst->data != nullptr);
    GGML_ASSERT(scales->data != nullptr && mins_t->data != nullptr);

    const dim3 grid(batchSize, numKVHeads);
    const dim3 block(headDim);
    const size_t smem = (size_t)headDim * sizeof(float);

    cudaStream_t stream = ctx.stream();
    saw8k_encode_kernel<<<grid, block, smem, stream>>>(
        k->data,
        (uint8_t *)dst->data,
        (uint16_t *)scales->data,
        (uint16_t *)mins_t->data,
        firstCell, headDim, numKVHeads, kIsF32, signLo, signHi
    );
}
