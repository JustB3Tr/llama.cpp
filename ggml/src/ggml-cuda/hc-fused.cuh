#pragma once
#include "common.cuh"

// GGML_CUDA_HC_MIX_FUSE: fuse the hyper-connection mix run starting at node i (see hc-fused.cu).
// Returns the number of nodes to skip after i, or 0 when the run does not match.
int ggml_cuda_hc_mix_fuse(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
