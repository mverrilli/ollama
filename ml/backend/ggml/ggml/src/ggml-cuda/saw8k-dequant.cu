#include "saw8k-dequant.cuh"

// Per-group asymmetric int8 decode for saw8k.
// Dequanted values remain in rotated space; the fused flash-attention kernel
// applies inverse FWHT to Q before the dot product, preserving Q·K identity.
// Buffer layout and decode math identical to q8k-dequant.cu.

#define SAW8K_DEQUANT_GROUP_SIZE 32

#if __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)
__global__ void saw8k_dequant_kernel(
    const uint8_t  * __restrict__ packed,
    const uint16_t * __restrict__ scales,
    const uint16_t * __restrict__ mins,
    uint16_t       * __restrict__ output,
    int headDim,
    int numKVHeads,
    int firstCell
) {
    const int c    = blockIdx.x;
    const int h    = blockIdx.y;
    const int t    = threadIdx.x;
    const int cell = firstCell + c;

    const int g         = t / SAW8K_DEQUANT_GROUP_SIZE;
    const int numGroups = headDim / SAW8K_DEQUANT_GROUP_SIZE;

    const int meta_idx = cell * numGroups * numKVHeads + h * numGroups + g;
    const float scale  = __half2float(__ushort_as_half(scales[meta_idx]));
    const float min_v  = __half2float(__ushort_as_half(mins[meta_idx]));

    const int packed_base = (cell * numKVHeads + h) * headDim;
    const int nibble      = (int)packed[packed_base + t];

    const float val = (float)nibble * scale + min_v;
    const int out_idx = c * numKVHeads * headDim + h * headDim + t;
    output[out_idx] = __half_as_ushort(__float2half_rn(val));
}
#else
__global__ void saw8k_dequant_kernel(
    const uint8_t *, const uint16_t *, const uint16_t *, uint16_t *,
    int, int, int) {}
#endif

void ggml_cuda_saw8k_dequant(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * packed  = dst->src[0];
    const struct ggml_tensor * scales  = dst->src[1];
    const struct ggml_tensor * mins_t  = dst->src[2];

    const int headDim    = ggml_get_op_params_i32(dst, 0);
    const int numKVHeads = ggml_get_op_params_i32(dst, 1);
    const int nCells     = ggml_get_op_params_i32(dst, 2);
    const int firstCell  = ggml_get_op_params_i32(dst, 3);

    GGML_ASSERT(headDim % SAW8K_DEQUANT_GROUP_SIZE == 0);
    GGML_ASSERT(packed->data && scales->data && mins_t->data);

    const dim3 grid(nCells, numKVHeads);
    const dim3 block(headDim);

    cudaStream_t stream = ctx.stream();
    saw8k_dequant_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t  *)packed->data,
        (const uint16_t *)scales->data,
        (const uint16_t *)mins_t->data,
        (uint16_t *)dst->data,
        headDim, numKVHeads, firstCell
    );
}
