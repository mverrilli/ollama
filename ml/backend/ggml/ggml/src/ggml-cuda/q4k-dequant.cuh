#pragma once

#include "common.cuh"

void ggml_cuda_q4k_dequant(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
