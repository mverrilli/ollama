#include "saw8k-fattn.cuh"
#include "saw8k-fattn-vec.cuh"
#include "ggml-cuda.h"

// Shared memory matches q8k layout (same tile sizes, same accumulator structure).
constexpr size_t SAW8K_FATTN_SMEM_FLOATS = 2560;

template<int D, int ncols>
static void saw8k_fattn_launch(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    float scale, float logit_softcap,
    int firstCell, int nCells, int nKVHeads,
    uint64_t signLo, uint64_t signHi)
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
        saw8k_flash_attn_ext_vec<D, ncols, true><<<blocks, threads,
            SAW8K_FATTN_SMEM_FLOATS * sizeof(float), ctx.stream()>>>(
            (const char *)Q->data,
            (const uint8_t *)K_p->data,
            (const char *)V->data,
            mask ? (const char *)mask->data : nullptr,
            (float *)dst->data,
            (const uint16_t *)k_scales->data,
            (const uint16_t *)k_mins->data,
            scale, logit_softcap, firstCell, nCells, nKVHeads,
            signLo, signHi,
            (int32_t)Q->ne[0], ne01, (int32_t)Q->ne[2], (int32_t)Q->ne[3],
            (int32_t)Q->nb[1], (int32_t)Q->nb[2], (int64_t)Q->nb[3],
            (int32_t)V->nb[1], (int32_t)V->nb[2], (int64_t)V->nb[3],
            mask ? (int32_t)mask->ne[0] : 0,
            mask ? (int32_t)mask->nb[1] : 0
        );
    } else {
        saw8k_flash_attn_ext_vec<D, ncols, false><<<blocks, threads,
            SAW8K_FATTN_SMEM_FLOATS * sizeof(float), ctx.stream()>>>(
            (const char *)Q->data,
            (const uint8_t *)K_p->data,
            (const char *)V->data,
            mask ? (const char *)mask->data : nullptr,
            (float *)dst->data,
            (const uint16_t *)k_scales->data,
            (const uint16_t *)k_mins->data,
            scale, logit_softcap, firstCell, nCells, nKVHeads,
            signLo, signHi,
            (int32_t)Q->ne[0], ne01, (int32_t)Q->ne[2], (int32_t)Q->ne[3],
            (int32_t)Q->nb[1], (int32_t)Q->nb[2], (int64_t)Q->nb[3],
            (int32_t)V->nb[1], (int32_t)V->nb[2], (int64_t)V->nb[3],
            mask ? (int32_t)mask->ne[0] : 0,
            mask ? (int32_t)mask->nb[1] : 0
        );
    }
}

void ggml_cuda_saw8k_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst)
{
    GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc >= 600 &&
                "saw8k fused flash attention requires compute capability 6.0+");

    float scale, logit_softcap;
    int32_t firstCell, nKVHeads, nCells;
    memcpy(&scale,         (const float  *)dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float  *)dst->op_params + 1, sizeof(float));
    memcpy(&firstCell,     (const int32_t*)dst->op_params + 2, sizeof(int32_t));
    memcpy(&nKVHeads,      (const int32_t*)dst->op_params + 3, sizeof(int32_t));
    memcpy(&nCells,        (const int32_t*)dst->op_params + 4, sizeof(int32_t));

    const uint64_t signLo = ((uint64_t)(uint32_t)ggml_get_op_params_i32(dst, 5)) |
                            ((uint64_t)(uint32_t)ggml_get_op_params_i32(dst, 6) << 32);
    const uint64_t signHi = ((uint64_t)(uint32_t)ggml_get_op_params_i32(dst, 7)) |
                            ((uint64_t)(uint32_t)ggml_get_op_params_i32(dst, 8) << 32);

    const int headDim  = (int)dst->src[0]->ne[0];
    const int nTokensQ = (int)dst->src[0]->ne[1];

    GGML_ASSERT(headDim == 128);

    if (nTokensQ >= 4) {
        saw8k_fattn_launch<128, 4>(ctx, dst, scale, logit_softcap, firstCell, nCells, nKVHeads, signLo, signHi);
    } else if (nTokensQ >= 2) {
        saw8k_fattn_launch<128, 2>(ctx, dst, scale, logit_softcap, firstCell, nCells, nKVHeads, signLo, signHi);
    } else {
        saw8k_fattn_launch<128, 1>(ctx, dst, scale, logit_softcap, firstCell, nCells, nKVHeads, signLo, signHi);
    }
}
