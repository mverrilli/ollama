#include "q4k-encode.cuh"
#include <math.h>

// Per-group asymmetric int4 encode: nibble packing, no rotation, no codebook.
//
// Grid: (batchSize, numKVHeads).  Block: headDim threads.
// Each thread block encodes one (token, head) pair.
// Groups of 32 elements: each warp handles one group.
// Per group: find min/max via warp shuffle → scale = (max-min)/15 → quantize [0,15].
//
// Encoded format:
//   packed  [(cell*numKVHeads+head) * D/2 + d/2]  u8  (low nibble = even d, high = odd d)
//   scales  [(cell*numKVHeads*numGroups + head*numGroups + g)]  f16
//   mins    [same index as scales]  f16

#define Q4K_GROUP_SIZE 32

#if __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)
__global__ void q4k_encode_kernel(
    const void     * __restrict__ k,      // [headDim, numKVHeads, batchSize] f16 or f32
    uint8_t        * __restrict__ packed, // [(cell*numKVHeads+head) * D/2]  u8
    uint16_t       * __restrict__ scales, // [(headDim/32) * numKVHeads, capacity] f16
    uint16_t       * __restrict__ mins,   // same index as scales  f16
    int firstCell,
    int headDim,
    int numKVHeads,
    int kIsF32
) {
    const int batch = blockIdx.x;
    const int head  = blockIdx.y;
    const int cell  = firstCell + batch;
    const int t     = threadIdx.x;    // 0 .. headDim-1
    const int g     = t / Q4K_GROUP_SIZE;
    const int j     = t % Q4K_GROUP_SIZE;
    const int numGroups = headDim / Q4K_GROUP_SIZE;

    extern __shared__ uint8_t nibbles[];  // headDim bytes, one nibble per thread

    const int kidx = batch * numKVHeads * headDim + head * headDim + t;
    const float x = kIsF32
        ? ((const float    *)k)[kidx]
        : __half2float(__ushort_as_half(((const uint16_t *)k)[kidx]));

    // Warp-level min/max reduction over the group (32 lanes = one warp).
    float min_val = x;
    float max_val = x;
    for (int offset = Q4K_GROUP_SIZE >> 1; offset > 0; offset >>= 1) {
        min_val = fminf(min_val, __shfl_xor_sync(0xffffffff, min_val, offset, Q4K_GROUP_SIZE));
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset, Q4K_GROUP_SIZE));
    }

    if (j == 0) {
        const float scale = (max_val - min_val) / 15.0f;
        const int meta_base = cell * numGroups * numKVHeads + head * numGroups;
        scales[meta_base + g] = __half_as_ushort(__float2half(scale));
        mins  [meta_base + g] = __half_as_ushort(__float2half(min_val));
    }

    const float scale = (max_val - min_val) / 15.0f;
    const float inv_scale = (scale > 1e-8f) ? (1.0f / scale) : 0.0f;
    const int q = max(0, min(15, (int)roundf((x - min_val) * inv_scale)));

    // Write nibble to shared memory — one slot per thread.
    nibbles[t] = (uint8_t)q;
    __syncthreads();

    // Even-indexed threads pack pairs: nibbles[t] → low nibble, nibbles[t+1] → high nibble.
    if (t % 2 == 0) {
        const int packed_base = (cell * numKVHeads + head) * (headDim / 2);
        packed[packed_base + t / 2] = nibbles[t] | (nibbles[t + 1] << 4);
    }
}
#else
__global__ void q4k_encode_kernel(
    const void *, uint8_t *, uint16_t *, uint16_t *,
    int, int, int, int) {}
#endif

void ggml_cuda_q4k_encode(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * k      = dst->src[0];
    const struct ggml_tensor * scales = dst->src[1];
    const struct ggml_tensor * mins_t = dst->src[2];

    const int headDim    = (int)k->ne[0];
    const int numKVHeads = (int)k->ne[1];
    const int batchSize  = (int)k->ne[2];
    const int firstCell  = ggml_get_op_params_i32(dst, 0);
    const int kIsF32     = (k->type == GGML_TYPE_F32) ? 1 : 0;

    GGML_ASSERT(headDim % Q4K_GROUP_SIZE == 0);
    GGML_ASSERT(headDim % 2 == 0);
    GGML_ASSERT(headDim <= 512);
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(k->data      != nullptr);
    GGML_ASSERT(dst->data    != nullptr);
    GGML_ASSERT(scales->data != nullptr);
    GGML_ASSERT(mins_t->data != nullptr);

    const dim3 grid(batchSize, numKVHeads);
    const dim3 block(headDim);
    const size_t smem = (size_t)headDim * sizeof(uint8_t);

    cudaStream_t stream = ctx.stream();
    q4k_encode_kernel<<<grid, block, smem, stream>>>(
        k->data,
        (uint8_t *)dst->data,
        (uint16_t *)scales->data,
        (uint16_t *)mins_t->data,
        firstCell, headDim, numKVHeads, kIsF32
    );
}
