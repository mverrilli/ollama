#include "q4k-fattn.cuh"
#include "q4k-fattn-vec.cuh"
#include "ggml-cuda.h"

// Shared memory budget matches q8k — same tile sizes, same accumulator layout.
// K is packed at D/2 bytes per (cell,head) instead of D bytes, but shared
// memory usage is dominated by Q/V tiles and the online-softmax accumulators.
constexpr size_t Q4K_FATTN_SMEM_FLOATS = 2560;

template<int D, int ncols>
static void q4k_fattn_launch(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    float scale, float logit_softcap,
    int firstCell, int nCells, int nKVHeads)
{
    const ggml_tensor * Q        = dst->src[0];
    const ggml_tensor * K_p      = dst->src[1];
    const ggml_tensor * V        = dst->src[2];
    const ggml_tensor * mask     = dst->src[3];
    const ggml_tensor * k_scales = dst->src[4];
    const ggml_tensor * k_mins   = dst->src[5];

    GGML_ASSERT(Q->ne[0] == D);
    GGML_ASSERT(Q->type  == GGML_TYPE_F32);
    GGML_ASSERT(V->type  == GGML_TYPE_F16);

    const int nTokensQ = (int)Q->ne[1];
    const int nHeadsQ  = (int)Q->ne[2];
    const int nSeq     = (int)Q->ne[3];

    const uint3 ne01 = init_fastdiv_values((uint64_t)nTokensQ);

    const int ntiles_x = (nTokensQ + ncols - 1) / ncols;
    const dim3 blocks(ntiles_x, 1, nHeadsQ * nSeq);
    const dim3 threads(WARP_SIZE, 4);

    if (logit_softcap != 0.0f) {
        q4k_flash_attn_ext_vec<D, ncols, true><<<blocks, threads,
            Q4K_FATTN_SMEM_FLOATS * sizeof(float), ctx.stream()>>>(
            (const char *)Q->data,
            (const uint8_t *)K_p->data,
            (const char *)V->data,
            mask ? (const char *)mask->data : nullptr,
            (float *)dst->data,
            (const uint16_t *)k_scales->data,
            (const uint16_t *)k_mins->data,
            scale, logit_softcap, firstCell, nCells, nKVHeads,
            (int32_t)Q->ne[0], ne01, (int32_t)Q->ne[2], (int32_t)Q->ne[3],
            (int32_t)Q->nb[1], (int32_t)Q->nb[2], (int64_t)Q->nb[3],
            (int32_t)V->nb[1], (int32_t)V->nb[2], (int64_t)V->nb[3],
            mask ? (int32_t)mask->ne[0] : 0,
            mask ? (int32_t)mask->nb[1] : 0
        );
    } else {
        q4k_flash_attn_ext_vec<D, ncols, false><<<blocks, threads,
            Q4K_FATTN_SMEM_FLOATS * sizeof(float), ctx.stream()>>>(
            (const char *)Q->data,
            (const uint8_t *)K_p->data,
            (const char *)V->data,
            mask ? (const char *)mask->data : nullptr,
            (float *)dst->data,
            (const uint16_t *)k_scales->data,
            (const uint16_t *)k_mins->data,
            scale, logit_softcap, firstCell, nCells, nKVHeads,
            (int32_t)Q->ne[0], ne01, (int32_t)Q->ne[2], (int32_t)Q->ne[3],
            (int32_t)Q->nb[1], (int32_t)Q->nb[2], (int64_t)Q->nb[3],
            (int32_t)V->nb[1], (int32_t)V->nb[2], (int64_t)V->nb[3],
            mask ? (int32_t)mask->ne[0] : 0,
            mask ? (int32_t)mask->nb[1] : 0
        );
    }
}

void ggml_cuda_q4k_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst)
{
    GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc >= 600 &&
                "q4k fused flash attention requires compute capability 6.0+");

    float scale, logit_softcap;
    int32_t firstCell, nKVHeads, nCells;
    memcpy(&scale,         (const float *)dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float *)dst->op_params + 1, sizeof(float));
    memcpy(&firstCell,     (const int32_t *)dst->op_params + 2, sizeof(int32_t));
    memcpy(&nKVHeads,      (const int32_t *)dst->op_params + 3, sizeof(int32_t));
    memcpy(&nCells,        (const int32_t *)dst->op_params + 4, sizeof(int32_t));

    const int headDim  = (int)dst->src[0]->ne[0];
    const int nTokensQ = (int)dst->src[0]->ne[1];

    GGML_ASSERT(headDim == 128);

    if (nTokensQ >= 4) {
        q4k_fattn_launch<128, 4>(ctx, dst, scale, logit_softcap, firstCell, nCells, nKVHeads);
    } else if (nTokensQ >= 2) {
        q4k_fattn_launch<128, 2>(ctx, dst, scale, logit_softcap, firstCell, nCells, nKVHeads);
    } else {
        q4k_fattn_launch<128, 1>(ctx, dst, scale, logit_softcap, firstCell, nCells, nKVHeads);
    }
}
