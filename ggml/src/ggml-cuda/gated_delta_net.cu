#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdint>

constexpr int gdn_cols_per_warp = 4;

// LLAMA_GDN_REPLAY: per kept token k [S_v, H_k], v [S_v, H], g (log gate, after softplus * A) [H], beta (after sigmoid) [H]
struct gdn_replay_args {
    const int32_t * info  = nullptr; // { S row, x row to replay, x row to write, tokens to replay }
    float *         x     = nullptr; // [n_x * k_max, rows]
    int64_t         n_x   = 0;
    int64_t         H_k   = 0;
    int             k_max = 0;
};

template <int S_v, bool KDA, bool keep_rs_t, bool replay_t = false, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     const int32_t * state_ids,
                                     const float * gate_dt,
                                     const float * gate_a,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const gdn_replay_args rp) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
    if constexpr (replay_t) {
        curr_state += (int64_t) rp.info[0] * H * S_v * S_v;
    } else if (state_ids != nullptr) {
        // the input is the whole cache and sequence i starts at row state_ids[i]
        curr_state += ((int64_t) state_ids[sequence] - (int64_t) sequence) * H * S_v * S_v;
    }
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    // the state update of one token, shared by the replay and the main loop (same operations, so a replayed token gives
    // the state the original pass gave); returns S^T q when q_reg is given
    auto update = [&](const float * k_reg, const float * q_reg, float v_col, float g_log, float beta_val) -> float {
        const float g_val = expf(g_log);
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);
        float delta_col = (v_col - g_val * kv_col) * beta_val;
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
            if (q_reg) {
                attn_partial += s_shard[r] * q_reg[r];
            }
        }
        return q_reg ? warp_reduce_sum<lanes_per_col>(attn_partial) : 0.0f;
    };

    float * x_w = nullptr;
    if constexpr (replay_t && !KDA) {
        const int64_t row_len = rp.n_x * rp.k_max;
        const float * x_r = rp.x + (int64_t) rp.info[1] * row_len;
        const int n_rep = rp.info[3];
        for (int t = 0; t < n_rep; t++) {
            const float * xt = x_r + t * rp.n_x;
            float k_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = xt[iq1 * S_v + r * lanes_per_col + lane_in_col];
            }
            const float * xv = xt + rp.H_k * S_v;
            const float * xg = xv + H * S_v;
            update(k_reg, nullptr, xv[h_idx * S_v + col], xg[h_idx], xg[H + h_idx]);
        }
        if (n_tokens <= rp.k_max) {
            // the next S plane is this replayed state; the tokens below are kept for the next ubatch
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i          = r * lanes_per_col + lane_in_col;
                state[col * S_v + i] = s_shard[r];
            }
            x_w = rp.x + (int64_t) rp.info[2] * row_len;
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        // raw gates: the same float operations as add -> softplus -> mul and sigmoid
        const float beta_val = gate_dt ? 1.0f / (1.0f + expf(-*beta_t)) : *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            float g_log = *g_t;
            if (gate_dt) {
                const float x  = __fadd_rn(g_log, gate_dt[h_idx]);
                const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
                g_log = __fmul_rn(sp, gate_a[h_idx]);
            }

            if constexpr (replay_t) {
                if (x_w != nullptr) {
                    // keep this token's inputs: each k head by the CTA of the same index, v by its column's lane 0
                    float * xt = x_w + t * rp.n_x;
                    if (h_idx < rp.H_k && blockIdx.z == 0 && threadIdx.y == 0 && col_in_warp == 0) {
#pragma unroll
                        for (int r = 0; r < rows_per_lane; r++) {
                            xt[h_idx * S_v + r * lanes_per_col + lane_in_col] = k_t[r * lanes_per_col + lane_in_col];
                        }
                    }
                    float * xv = xt + rp.H_k * S_v;
                    if (lane_in_col == 0) {
                        xv[h_idx * S_v + col] = v_t[col];
                    }
                    if (blockIdx.z == 0 && threadIdx.y == 0 && lane == 0) {
                        xv[H * S_v + h_idx]     = g_log;
                        xv[H * S_v + H + h_idx] = beta_val;
                    }
                }
            }

            // kv = S^T k, delta = (v - g kv) beta, S = g S + k delta, attn = S^T q
            float attn_col = update(k_reg, q_reg, v_t[col], g_log, beta_val);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
        if (!replay_t || x_w == nullptr) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i          = r * lanes_per_col + lane_in_col;
                state[col * S_v + i] = s_shard[r];
            }
        }
    }
}

template <bool KDA, bool keep_rs_t, bool replay_t = false>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d, const int32_t * ids_d,
        const float * dt_d, const float * a_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream, const gdn_replay_args rp = {}) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, replay_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, ids_d, dt_d, a_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rp);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, replay_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, ids_d, dt_d, a_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rp);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, replay_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, ids_d, dt_d, a_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rp);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, replay_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, ids_d, dt_d, a_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rp);
            break;
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    // optional row index per sequence into a whole-cache state (ggml_gated_delta_net_ids)
    const int32_t * ids_d = dst->src[6] ? (const int32_t *) dst->src[6]->data : nullptr;
    // optional raw gates (ggml_gated_delta_net_ext): softplus/sigmoid applied in the kernel
    const float * dt_d = dst->src[7] ? (const float *) dst->src[7]->data : nullptr;
    const float * a_d  = dst->src[8] ? (const float *) dst->src[8]->data : nullptr;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(dt_d != nullptr || ggml_is_contiguous(src_g));
    GGML_ASSERT(dt_d != nullptr || ggml_is_contiguous(src_beta));
    GGML_ASSERT(dt_d == nullptr || (!kda && src_g->nb[2] == src_beta->nb[2] && src_g->nb[3] == src_beta->nb[3]));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // LLAMA_GDN_REPLAY (ggml_gated_delta_net_replay): x in src[9], info in src[6], k_max in op param 1
    if (dst->src[9] != nullptr) {
        GGML_ASSERT(!kda && !keep_rs && n_seqs == 1);
        gdn_replay_args rp;
        rp.info  = (const int32_t *) dst->src[6]->data;
        rp.x     = (float *) dst->src[9]->data;
        rp.H_k   = neqk1;
        rp.n_x   = S_v * neqk1 + S_v * H + 2 * H;
        rp.k_max = ggml_get_op_params_i32(dst, 1);
        GGML_ASSERT(dst->src[9]->ne[0] == rp.n_x * rp.k_max);
        if (n_tokens > rp.k_max) {
            // a long batch may take the chunked kernel: replay into a scratch state first, then run it from there
            ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
            args.state_out = state_d;
            ggml_cuda_pool_alloc<float> s_rep(ctx.pool(), S_v * S_v * H);
            launch_gated_delta_net<false, false, true>(q_d, k_d, v_d, g_d, b_d, s_d, nullptr, dt_d, a_d, dst_d, s_rep.get(),
                S_v, H, 0, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, rp);
            args.state = s_rep.get();
            if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
                return;
            }
            const gdn_replay_args none = {};
            // the recurrent kernel from the replayed state: no replay, nothing kept (one row at offset 0 of s_rep)
            launch_gated_delta_net<false, false, false>(q_d, k_d, v_d, g_d, b_d, s_rep.get(), nullptr, dt_d, a_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, none);
            return;
        }
        launch_gated_delta_net<false, false, true>(q_d, k_d, v_d, g_d, b_d, s_d, nullptr, dt_d, a_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, rp);
        return;
    }

    {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        // the chunked kernel reads the state from its base: with a row index (LLAMA_GDN_DIRECT, e.g. a rollback plane after
        // a speculative verify) gather that row into a scratch state first (before this, a prompt batch after a rollback
        // started from the state that still had the rejected draft tokens)
        ggml_cuda_pool_alloc<float> s_row(ctx.pool());
        if (ids_d != nullptr && args.eligible && n_tokens >= 64 && !kda) {
            s_row.alloc(S_v * S_v * H * n_seqs);
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, ids_d, dt_d, a_d, dst_d, s_row.get(),
                S_v, H, 0, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, S_v * S_v * H * n_seqs, 1, stream);
            args.state = s_row.get();
        }
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }
    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, ids_d, dt_d, a_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, ids_d, dt_d, a_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, ids_d, dt_d, a_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, ids_d, dt_d, a_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
