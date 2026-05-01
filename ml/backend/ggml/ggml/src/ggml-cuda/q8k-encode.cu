#include "q8k-encode.cuh"
#include <math.h>

// Per-group asymmetric int8 encode: no rotation, no codebook.
//
// Grid: (batchSize, numKVHeads).  Block: headDim threads.
// Each thread block encodes one (token, head) pair.
// Groups of 32 elements: each warp handles one group.
// Per group: find min/max via warp shuffle → scale = (max-min)/255 → quantize.
// One byte per element in output buffer.
//
// Encoded format:
//   packed  [(cell*numKVHeads+head) * headDim + elem]  u8
//   scales  [(cell*numKVHeads*numGroups + head*numGroups + g)]  f16
//   mins    [same index as scales]  f16

#define Q8K_GROUP_SIZE 32

#if __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)
__global__ void q8k_encode_kernel(
    const void     * __restrict__ k,      // [headDim, numKVHeads, batchSize] f16 or f32
    uint8_t        * __restrict__ packed, // [headDim/2 * numKVHeads, capacity] i8
    uint16_t       * __restrict__ scales, // [(headDim/32) * numKVHeads, capacity] f16
    uint16_t       * __restrict__ mins,   // [(headDim/32) * numKVHeads, capacity] f16
    int firstCell,
    int headDim,
    int numKVHeads,
    int kIsF32
) {
    const int batch = blockIdx.x;
    const int head  = blockIdx.y;
    const int cell  = firstCell + batch;
    const int t     = threadIdx.x;    // 0 .. headDim-1
    const int g     = t / Q8K_GROUP_SIZE;   // group index
    const int j     = t % Q8K_GROUP_SIZE;   // within-group index (= warp lane)
    const int numGroups = headDim / Q8K_GROUP_SIZE;

    // Load element: K layout [headDim, numKVHeads, batchSize] — element [t, head, batch]
    // Memory address: batch * numKVHeads * headDim + head * headDim + t
    const int kidx = batch * numKVHeads * headDim + head * headDim + t;
    const float x = kIsF32
        ? ((const float    *)k)[kidx]
        : __half2float(__ushort_as_half(((const uint16_t *)k)[kidx]));

    // Warp-level min/max reduction (each warp = one group of 32 elements).
    float min_val = x;
    float max_val = x;
    for (int offset = Q8K_GROUP_SIZE >> 1; offset > 0; offset >>= 1) {
        min_val = fminf(min_val, __shfl_xor_sync(0xffffffff, min_val, offset, Q8K_GROUP_SIZE));
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset, Q8K_GROUP_SIZE));
    }

    // Write scale and min — one write per group, performed by lane 0 of each warp
    if (j == 0) {
        const float scale = (max_val - min_val) / 255.0f;
        const int meta_base = cell * numGroups * numKVHeads + head * numGroups;
        scales[meta_base + g] = __half_as_ushort(__float2half(scale));
        mins  [meta_base + g] = __half_as_ushort(__float2half(min_val));
    }

    // Quantize element to byte [0, 255]
    const float scale = (max_val - min_val) / 255.0f;
    const float inv_scale = (scale > 1e-8f) ? (1.0f / scale) : 0.0f;
    const int q = max(0, min(255, (int)roundf((x - min_val) * inv_scale)));

    // Each thread writes its own byte — no shared memory needed
    const int packed_base = (cell * numKVHeads + head) * headDim;
    packed[packed_base + t] = (uint8_t)q;
}
#else
__global__ void q8k_encode_kernel(
    const void *, uint8_t *, uint16_t *, uint16_t *,
    int, int, int, int) {}
#endif

void ggml_cuda_q8k_encode(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // dst = view of packed buffer
    // src[0] = k   [headDim, numKVHeads, batchSize] f16
    // src[1] = scales (f16 output, written as side-effect)
    // src[2] = mins   (f16 output, written as side-effect)
    // op_params[0] = firstCell

    const struct ggml_tensor * k      = dst->src[0];
    const struct ggml_tensor * scales = dst->src[1];
    const struct ggml_tensor * mins_t = dst->src[2];

    const int headDim    = (int)k->ne[0];
    const int numKVHeads = (int)k->ne[1];
    const int batchSize  = (int)k->ne[2];
    const int firstCell  = ggml_get_op_params_i32(dst, 0);
    const int kIsF32     = (k->type == GGML_TYPE_F32) ? 1 : 0;

    GGML_ASSERT(headDim % Q8K_GROUP_SIZE == 0);
    GGML_ASSERT(headDim <= 512);  // shared mem guard
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(k->data      != nullptr);
    GGML_ASSERT(dst->data    != nullptr);
    GGML_ASSERT(scales->data != nullptr);
    GGML_ASSERT(mins_t->data != nullptr);

    const dim3 grid(batchSize, numKVHeads);
    const dim3 block(headDim);

    cudaStream_t stream = ctx.stream();
    q8k_encode_kernel<<<grid, block, 0, stream>>>(
        k->data,
        (uint8_t *)dst->data,
        (uint16_t *)scales->data,
        (uint16_t *)mins_t->data,
        firstCell, headDim, numKVHeads, kIsF32
    );
}
