#include "q4k-dequant.cuh"

// Per-group asymmetric int4 decode: nibble → val = nibble * scale + min.
//
// Grid: (nCells, numKVHeads).  Block: headDim threads (one thread per element).
// Each thread decodes one element from a nibble-packed byte.
//
// Encoded format (must match q4k-encode.cu):
//   packed  [(firstCell+c)*numKVHeads*(headDim/2) + h*(headDim/2) + elem/2]  u8
//     low nibble  = element at even index (elem & ~1)
//     high nibble = element at odd index  (elem |  1)
//   scales  [(firstCell+c)*numGroups*numKVHeads + h*numGroups + g]  f16
//   mins    [same]  f16
//
// Output: [headDim, numKVHeads, nCells] f16
//   element [d, h, c] at: c * numKVHeads * headDim + h * headDim + d

#define Q4K_GROUP_SIZE 32

#if __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)
__global__ void q4k_dequant_kernel(
    const uint8_t  * __restrict__ packed,  // [(cell*nKVH+h)*headDim/2 + d/2] u8
    const uint16_t * __restrict__ scales,  // [(cell*nG*nKVH + h*nG + g)] f16
    const uint16_t * __restrict__ mins,    // same
    uint16_t       * __restrict__ output,  // [headDim, numKVHeads, nCells] f16
    int headDim,
    int numKVHeads,
    int firstCell
) {
    const int c    = blockIdx.x;   // relative cell index [0, nCells)
    const int h    = blockIdx.y;   // KV head index
    const int d    = threadIdx.x;  // element index [0, headDim)
    const int cell = firstCell + c;

    const int numGroups = headDim / Q4K_GROUP_SIZE;
    const int g         = d / Q4K_GROUP_SIZE;

    // Load scale and min for this element's group.
    const int meta_idx = cell * numGroups * numKVHeads + h * numGroups + g;
    const float scale  = __half2float(__ushort_as_half(scales[meta_idx]));
    const float min_v  = __half2float(__ushort_as_half(mins[meta_idx]));

    // Extract nibble: low nibble for even d, high nibble for odd d.
    const int packed_base = (cell * numKVHeads + h) * (headDim / 2);
    const uint8_t byte    = packed[packed_base + d / 2];
    const int nibble      = (d & 1) ? (byte >> 4) : (byte & 0xF);

    // Decode and write f16.
    const float val   = (float)nibble * scale + min_v;
    const int out_idx = c * numKVHeads * headDim + h * headDim + d;
    output[out_idx]   = __half_as_ushort(__float2half_rn(val));
}
#else
__global__ void q4k_dequant_kernel(
    const uint8_t *, const uint16_t *, const uint16_t *, uint16_t *,
    int, int, int) {}
#endif

void ggml_cuda_q4k_dequant(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // dst = [headDim, numKVHeads, nCells] f16
    // src[0] = encode_result (packed buffer view)
    // src[1] = scales f16
    // src[2] = mins f16
    // op_params: [headDim, numKVHeads, nCells, firstCell]

    const struct ggml_tensor * packed  = dst->src[0];
    const struct ggml_tensor * scales  = dst->src[1];
    const struct ggml_tensor * mins_t  = dst->src[2];

    const int headDim    = ggml_get_op_params_i32(dst, 0);
    const int numKVHeads = ggml_get_op_params_i32(dst, 1);
    const int nCells     = ggml_get_op_params_i32(dst, 2);
    const int firstCell  = ggml_get_op_params_i32(dst, 3);

    GGML_ASSERT(headDim % Q4K_GROUP_SIZE == 0);
    GGML_ASSERT(packed->data != nullptr);
    GGML_ASSERT(scales->data != nullptr);
    GGML_ASSERT(mins_t->data != nullptr);

    const dim3 grid(nCells, numKVHeads);
    const dim3 block(headDim);

    cudaStream_t stream = ctx.stream();
    q4k_dequant_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t  *)packed->data,
        (const uint16_t *)scales->data,
        (const uint16_t *)mins_t->data,
        (uint16_t *)dst->data,
        headDim, numKVHeads, firstCell
    );
}
