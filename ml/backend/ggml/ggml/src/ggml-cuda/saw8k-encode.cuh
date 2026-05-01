#pragma once

#include "common.cuh"

void ggml_cuda_saw8k_encode(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
