// Fused hyper-connection mix (qwen4exp, GGML_CUDA_HC_MIX_FUSE): replaces the node run
//   RMS_NORM -> MUL(norm weight) -> [views] -> MUL_MAT(down, BF16) -> SILU -> MUL_MAT(up, BF16) -> [views] -> DSV4_HC_PRE(gated)
// with two kernels for small batches:
//   A: per-stream RMS norm, the down projection over the normed input and SiLU (block 0 also stores the per-stream
//      norm scales); when the next node is the inject projection of the same normed input (MUL_MAT, BF16), extra
//      blocks compute it too
//   B: the normed input (written out as hc_norm), the up projection, the sigmoid gate and the scaled sum over the streams
// GGML_CUDA_HC_MIX_POST=1: when the run is preceded by the DSV4_HC_POST that produces its input, A also computes that
// post on the fly (every block for its own reads, block 0 writes it), with the same float operations as the post kernel.
// The sums run in a different order than the separate kernels, so results match to rounding, not bit for bit;
// the folded post and the normed input are the same values as the separate kernels give.
#include "hc-fused.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda_bf16.h> // HIP/MUSA: vendors/*.h (via common.cuh) map the bf16 types
#endif

static constexpr int HCF_HC        = 4;   // streams
static constexpr int HCF_MAX_TOK   = 8;   // tokens per launch (decode, speculative verify)
static constexpr int HCF_A_THREADS = 256;
static constexpr int HCF_B_E       = 8;   // outputs per block (one warp each)

// the DSV4_HC_POST folded into kernel A: x[i0, s, t] = (bo[i0, t] + add[i0, t]*sigmoid(gate[t]))*pv(s, t) + res[i0, s, t]
struct hcf_post {
    const float * bo;   // [n_embd, NT] block output, null when no post is folded
    const float * res;  // [n_embd, hc, NT] residual in
    const float * pl;   // post logits, (s, t) at s*sp0 + t*sp1
    const float * add;  // [n_embd, NT] or null
    const float * gate; // [1, NT] (t at t*sg1)
    int64_t sbo_t, sres_t, sp0, sp1, sadd_t, sg1;
    int   sig;
    float p_in, p_out;
};

// threads are split by stream (HCF_A_THREADS/HC per stream), so each thread keeps one sum of squares and one partial
// dot per row and token; RPB rows per block: more rows per block means fewer blocks re-reading x, which matters as the
// token count grows (x traffic = blocks * D * NT)
template <int NT, int RPB>
static __global__ void __launch_bounds__(HCF_A_THREADS) hc_mix_a(
        const hcf_post pp,
        float * __restrict__ x,                // [n_embd, hc, NT] residual (written by block 0 when a post is folded)
        const float * __restrict__ w_norm,     // [n_embd*hc]
        const nv_bfloat16 * __restrict__ w_dn, // [D, R] row-major: row r at r*D
        const nv_bfloat16 * __restrict__ w_inj,// [D, n_inj] or null: rows R.. R+n_inj-1
        float * __restrict__ rs_out,           // [NT][HC] norm scale per token and stream
        float * __restrict__ lo_out,           // [R, NT] silu(down)
        float * __restrict__ inj_out,          // [n_inj, NT] inject (no activation)
        const int n_embd, const int R, const int n_rows, const float eps,
        const int64_t sx_t, const int64_t slo_t, const int64_t sinj_t) {
    constexpr int TPS = HCF_A_THREADS/HCF_HC;   // threads per stream
    constexpr int WPS = TPS/32;                 // warps per stream
    constexpr int NV  = (RPB + 1)*NT;           // values per thread to reduce: dots and the sum of squares
    const int D   = n_embd*HCF_HC;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int s   = tid / TPS, u = tid % TPS;   // this thread's stream
    const int r0  = blockIdx.x*RPB;

    __shared__ float s_red[HCF_A_THREADS/32][NV];
    __shared__ float s_pv[NT][HCF_HC], s_g[NT];

    const nv_bfloat16 * w_row[RPB];
#pragma unroll
    for (int j = 0; j < RPB; ++j) {
        const int r = r0 + j;
        w_row[j] = r < R ? w_dn + (int64_t) r*D : (r < n_rows ? w_inj + (int64_t) (r - R)*D : nullptr);
    }

    float ss[NT], p[RPB][NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        ss[t] = 0.0f;
#pragma unroll
        for (int j = 0; j < RPB; ++j) { p[j][t] = 0.0f; }
    }
    if (pp.bo) {
        if (tid < NT*HCF_HC) {
            const int t = tid / HCF_HC, q = tid % HCF_HC;
            float pv = pp.pl[q*pp.sp0 + t*pp.sp1];
            if (pp.sig) {
                // same operations as the post kernel
                pv = pp.p_out * (1.0f / (1.0f + expf(-(pp.p_in * pv))));
            }
            s_pv[t][q] = pv;
        }
        if (tid < NT && pp.add) {
            s_g[tid] = 1.0f / (1.0f + expf(-pp.gate[tid*pp.sg1]));
        }
        __syncthreads();
    }
    const float4 * x4  = (const float4 *) x;
    const float4 * wn4 = (const float4 *) w_norm;
    const int E8 = n_embd/8;
    // 16-byte weight loads: 8 elements per row and iteration, as two float4 halves (n_embd % 8 == 0)
    for (int k8 = s*E8 + u; k8 < (s + 1)*E8; k8 += TPS) {
        uint4 raw[RPB];
#pragma unroll
        for (int j = 0; j < RPB; ++j) {
            raw[j] = w_row[j] ? ((const uint4 *) w_row[j])[k8] : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int    k4 = 2*k8 + h;
            const float4 wn = wn4[k4];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                float4 v;
                if (pp.bo) {
                    const int i0 = 4*k4 - s*n_embd;
                    float4 xb = *(const float4 *) (pp.bo + i0 + t*pp.sbo_t);
                    if (pp.add) {
                        const float4 a = *(const float4 *) (pp.add + i0 + t*pp.sadd_t);
                        const float  g = s_g[t];
                        xb = make_float4(__fadd_rn(xb.x, __fmul_rn(a.x, g)), __fadd_rn(xb.y, __fmul_rn(a.y, g)),
                                         __fadd_rn(xb.z, __fmul_rn(a.z, g)), __fadd_rn(xb.w, __fmul_rn(a.w, g)));
                    }
                    const float  pv = s_pv[t][s];
                    const float4 r  = *(const float4 *) (pp.res + 4*k4 + t*pp.sres_t);
                    // the post kernel's x*pv + residual compiles to one fma
                    v = make_float4(__fmaf_rn(xb.x, pv, r.x), __fmaf_rn(xb.y, pv, r.y), __fmaf_rn(xb.z, pv, r.z), __fmaf_rn(xb.w, pv, r.w));
                    if (blockIdx.x == 0) {
                        ((float4 *) x)[k4 + t*(sx_t/4)] = v;
                    }
                } else {
                    v = x4[k4 + t*(sx_t/4)];
                }
                ss[t] += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
                const float4 xw = make_float4(v.x*wn.x, v.y*wn.y, v.z*wn.z, v.w*wn.w);
#pragma unroll
                for (int j = 0; j < RPB; ++j) {
                    const unsigned int u0 = h ? raw[j].z : raw[j].x, u1 = h ? raw[j].w : raw[j].y;
                    const float2 lo2 = __bfloat1622float2(*(const nv_bfloat162 *) &u0);
                    const float2 hi2 = __bfloat1622float2(*(const nv_bfloat162 *) &u1);
                    p[j][t] += lo2.x*xw.x + lo2.y*xw.y + hi2.x*xw.z + hi2.y*xw.w;
                }
            }
        }
    }
    // reduce within each warp, then over the warps of a stream
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        float v = i < RPB*NT ? p[i / NT][i % NT] : ss[i - RPB*NT];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffff, v, o, 32); }
        if (lane == 0) { s_red[warp][i] = v; }
    }
    __syncthreads();
    // per stream totals: stream q is warps q*WPS .. q*WPS + WPS-1
    auto tot = [&](int q, int i) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < WPS; ++w) { v += s_red[q*WPS + w][i]; }
        return v;
    };
    if (tid < RPB*NT) {
        const int j = tid / NT, t = tid % NT;
        const int r = r0 + j;
        if (r < n_rows) {
            float v = 0.0f;
#pragma unroll
            for (int q = 0; q < HCF_HC; ++q) {
                v += rsqrtf(tot(q, RPB*NT + t) / (float) n_embd + eps) * tot(q, j*NT + t);
            }
            if (r >= R) {
                inj_out[(r - R) + t*sinj_t] = v;
            } else {
                lo_out[r + t*slo_t] = v / (1.0f + expf(-v));
            }
        }
    }
    if (blockIdx.x == 0 && tid < NT*HCF_HC) {
        const int t = tid / HCF_HC, q = tid % HCF_HC;
        rs_out[tid] = rsqrtf(tot(q, RPB*NT + t) / (float) n_embd + eps);
    }
}

template <int NT>
static __global__ void hc_mix_b(
        const float * __restrict__ x,          // [D, NT] residual
        const float * __restrict__ w_norm,     // [D]
        const float * __restrict__ rs,         // [NT][HC] norm scales from A
        const float * __restrict__ lo,         // [R, NT]
        const nv_bfloat16 * __restrict__ w_up, // [R, D] row-major: row d at d*R
        float * __restrict__ xn,               // [D, NT] normed input (hc_norm), written here
        float * __restrict__ mixed,            // [n_embd, NT]
        const float * __restrict__ inj_src,    // inject from A's scratch, copied to inj_dst (or null)
        float * __restrict__ inj_dst, const int n_inj_vals,
        const int n_embd, const int R, const float scale,
        const int64_t sx_t, const int64_t sxn_t, const int64_t slo_t, const int64_t smx_t) {
    extern __shared__ float s_lo[]; // [NT][R]
    for (int i = threadIdx.x; i < R*NT; i += blockDim.x) {
        s_lo[i] = lo[(i % R) + (i / R)*slo_t];
    }
    if (inj_src && blockIdx.x == 0) {
        for (int i = threadIdx.x; i < n_inj_vals; i += blockDim.x) { inj_dst[i] = inj_src[i]; }
    }
    __syncthreads();

    // one warp per output e: 8 lanes per gate row (lane/8 = stream s), each lane one partial sum per token over a
    // 16-byte column stride of 8; 3 shuffles per token sum a row, 2 more the gated terms of the 4 streams
    // (llama-moe: 7900 XTX 15.4 -> 14.4 us at 4 tokens vs 32 lanes over all 4 rows; the second pass of that layout ran
    // 8 of 32 lanes with R = 320)
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int e = blockIdx.x*HCF_B_E + warp;
    if (e >= n_embd) {
        return;
    }
    static_assert(HCF_HC*8 == 32, "8 lanes per stream row");
    const int s = lane / 8, q = lane % 8;
    const uint4 * row = (const uint4 *) (w_up + (int64_t) (s*n_embd + e)*R);
    float g[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) { g[t] = 0.0f; }
    for (int j8 = q; j8 < R/8; j8 += 8) {
        const uint4 raw = row[j8];
        const unsigned int u[4] = { raw.x, raw.y, raw.z, raw.w };
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float2 wv = __bfloat1622float2(*(const nv_bfloat162 *) &u[c]);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float2 l = *(const float2 *) &s_lo[t*R + 8*j8 + 2*c];
                g[t] += wv.x*l.x + wv.y*l.y;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) { g[t] += __shfl_xor_sync(0xffffffff, g[t], o, 32); }
    }
    float term = 0.0f;
    if (q < NT) {
        const int t = q;
        float gv = 0.0f;
#pragma unroll
        for (int k = 0; k < NT; ++k) { gv = k == t ? g[k] : gv; }
        const int k = s*n_embd + e;
        // same operations as rms_norm -> mul
        const float xnv = (x[k + t*sx_t]*rs[t*HCF_HC + s])*w_norm[k];
        xn[k + t*sxn_t] = xnv;
        term = xnv * (1.0f / (1.0f + expf(-gv)));
    }
    term += __shfl_xor_sync(0xffffffff, term, 8, 32);
    term += __shfl_xor_sync(0xffffffff, term, 16, 32);
    if (lane < NT) {
        mixed[e + lane*smx_t] = scale*term;
    }
}

static bool hcf_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

// the tensor whose memory a view chain ends in
static const ggml_tensor * hcf_base(const ggml_tensor * t) {
    while (t->view_src) { t = t->view_src; }
    return t;
}

static bool hcf_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data, * a1 = a0 + ggml_nbytes(a);
    const char * b0 = (const char *) b->data, * b1 = b0 + ggml_nbytes(b);
    return a0 < b1 && b0 < a1;
}

// does any node after `after` read tensor t (or a view of it)
static bool hcf_read_after(const ggml_cgraph * cgraph, int after, const ggml_tensor * t) {
    for (int k = after + 1; k < cgraph->n_nodes; ++k) {
        const ggml_tensor * nk = cgraph->nodes[k];
        for (int s = 0; s < GGML_MAX_SRC && nk->src[s]; ++s) {
            if (hcf_base(nk->src[s]) == t) {
                return true;
            }
        }
    }
    return false;
}

static int hc_mix_fuse_at(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, const ggml_tensor * post);

int ggml_cuda_hc_mix_fuse(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = getenv("GGML_CUDA_HC_MIX_FUSE") != nullptr && atoi(getenv("GGML_CUDA_HC_MIX_FUSE")) != 0;
    if (!enabled) {
        return 0;
    }
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op == GGML_OP_DSV4_HC_POST) {
        // opt-in: every A block then reads the post inputs instead of x (+50% bytes), which costs more than the
        // saved kernel once the speculative verify windows grow (MTP smoke: -4%), and is about even at one token
        static const bool post_on = getenv("GGML_CUDA_HC_MIX_POST") != nullptr && atoi(getenv("GGML_CUDA_HC_MIX_POST")) != 0;
        if (!post_on) {
            return 0;
        }
        int j = i + 1;
        while (j < cgraph->n_nodes && hcf_is_view(cgraph->nodes[j])) { ++j; }
        if (j < cgraph->n_nodes && cgraph->nodes[j]->op == GGML_OP_RMS_NORM && hcf_base(cgraph->nodes[j]->src[0]) == node) {
            const int r = hc_mix_fuse_at(ctx, cgraph, j, node);
            if (r > 0) {
                return r + (j - i);
            }
        }
        return 0; // the post runs alone, the run after it may still fuse
    }
    return hc_mix_fuse_at(ctx, cgraph, i, nullptr);
}

static int hc_mix_fuse_at(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, const ggml_tensor * post) {
    const int n = cgraph->n_nodes;
    ggml_tensor * nd[6];
    int idx[6];
    const ggml_op want[6] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL_MAT, GGML_OP_DSV4_HC_PRE };
    int j = i;
    for (int k = 0; k < 6; ++k) {
        while (k > 0 && j < n && hcf_is_view(cgraph->nodes[j])) { ++j; }
        if (j >= n || cgraph->nodes[j]->op != want[k]) {
            return 0;
        }
        nd[k] = cgraph->nodes[j]; idx[k] = j; ++j;
    }
    ggml_tensor * rn = nd[0], * mul = nd[1], * dn = nd[2], * act = nd[3], * up = nd[4], * pre = nd[5];
    const ggml_tensor * x = rn->src[0];
    const int64_t n_embd = x->ne[0], nt = x->ne[2];

    // shapes, types, wiring
    if (x->type != GGML_TYPE_F32 || x->ne[1] != HCF_HC || nt < 1 || nt > HCF_MAX_TOK || !ggml_is_contiguous(x)) return 0;
    if (mul->src[0] != rn || mul->src[1]->type != GGML_TYPE_F32 || ggml_nelements(mul->src[1]) != n_embd*HCF_HC || !ggml_is_contiguous(mul->src[1])) return 0;
    if (!ggml_is_contiguous(mul)) return 0;
    const ggml_tensor * wd = dn->src[0], * wu = up->src[0];
    const int64_t D = n_embd*HCF_HC, R = wd->ne[1];
    if (wd->type != GGML_TYPE_BF16 || wd->ne[0] != D || !ggml_is_contiguous(wd) || R % 8 != 0 || n_embd % 8 != 0) return 0;
    if (hcf_base(dn->src[1]) != hcf_base(mul) || dn->src[1]->data != mul->data || dn->ne[1] != nt || !ggml_is_contiguous(dn)) return 0;
    if (ggml_get_unary_op(act) != GGML_UNARY_OP_SILU || act->src[0] != dn || !ggml_is_contiguous(act)) return 0;
    if (wu->type != GGML_TYPE_BF16 || wu->ne[0] != R || wu->ne[1] != D || !ggml_is_contiguous(wu)) return 0;
    if (up->src[1] != act || !ggml_is_contiguous(up)) return 0;
    if (ggml_get_op_params_i32(pre, 1) == 0) return 0; // gated only
    if (pre->src[0]->data != mul->data || pre->src[1]->data != up->data || !ggml_is_contiguous(pre)) return 0;
    // written: mul (hc_norm), act (lo), pre (hc_mixed). Not written: rn, dn, up, which must have no reader past the run
    if (hcf_read_after(cgraph, idx[5], rn) || hcf_read_after(cgraph, idx[5], dn) || hcf_read_after(cgraph, idx[5], up)) return 0;
    // A reads x and writes lo; B reads x and lo while it writes mul and pre (each B warp reads only the x it normalizes)
    if (hcf_overlap(pre, mul) || hcf_overlap(pre, x) || hcf_overlap(mul, x) || hcf_overlap(act, x) || hcf_overlap(act, mul)) return 0;

    // optional: the inject projection of the same normed input, next in the node order (see build_hc_mix)
    int last = idx[5];
    const ggml_tensor * inj = nullptr;
    static const bool inj_off = getenv("GGML_CUDA_HC_MIX_NOINJ") != nullptr; // A/B switch
    if (!inj_off) {
        int j2 = idx[5] + 1;
        while (j2 < n && hcf_is_view(cgraph->nodes[j2])) { ++j2; }
        if (j2 < n) {
            const ggml_tensor * c = cgraph->nodes[j2];
            const ggml_tensor * wi = c->src[0];
            if (c->op == GGML_OP_MUL_MAT && wi->type == GGML_TYPE_BF16 && wi->ne[0] == D && ggml_is_contiguous(wi) &&
                    wi->ne[2] == 1 && wi->ne[3] == 1 && c->src[1]->data == mul->data &&
                    hcf_base(c->src[1]) == hcf_base(mul) && c->type == GGML_TYPE_F32 && c->ne[1] == nt && ggml_is_contiguous(c) &&
                    !hcf_overlap(c, x) && !hcf_overlap(c, mul) && !hcf_overlap(c, act) && !hcf_overlap(c, pre)) {
                inj  = c;
                last = j2;
            }
        }
    }

    // the folded post: its inputs die at the post, so the allocator may hand their memory to what this run writes.
    // x must not overlap them (block 0 writes it while the other blocks read them); lo and inject go to scratch
    // buffers when they would, and B (which runs after every read of A) copies inject into place
    hcf_post pp = {};
    bool lo_scratch  = hcf_overlap(pre, act); // the allocator often places pre over act (dead after up)
    bool inj_scratch = false;
    if (post) {
        const ggml_tensor * bo = post->src[0], * res = post->src[1], * pl = post->src[2];
        const ggml_tensor * add = post->src[4], * gate = post->src[5];
        auto al16 = [](const ggml_tensor * t, int64_t s) { return ((uintptr_t) t->data % 16) == 0 && s % 4 == 0; };
        if (post->src[3] != nullptr || x->data != post->data || !ggml_is_contiguous(post) || ggml_nelements(post) != ggml_nelements(x) ||
                bo->type != GGML_TYPE_F32 || bo->ne[0] != n_embd || bo->ne[1] != nt || bo->nb[0] != sizeof(float) ||
                res->type != GGML_TYPE_F32 || !ggml_is_contiguous(res) || ggml_nelements(res) != D*nt ||
                pl->type != GGML_TYPE_F32 || (add && (add->type != GGML_TYPE_F32 || add->nb[0] != sizeof(float) || gate == nullptr)) ||
                !al16(bo, bo->nb[1]/sizeof(float)) || !al16(res, D) || (add && !al16(add, add->nb[1]/sizeof(float)))) {
            return 0;
        }
        const ggml_tensor * ins[5] = { bo, res, pl, add, gate };
        for (const ggml_tensor * t : ins) {
            if (t == nullptr) {
                continue;
            }
            if (hcf_overlap(post, t)) {
                return 0;
            }
            lo_scratch  = lo_scratch  || hcf_overlap(act, t);
            inj_scratch = inj_scratch || (inj && hcf_overlap(inj, t));
        }
        pp.bo = (const float *) bo->data; pp.res = (const float *) res->data; pp.pl = (const float *) pl->data;
        pp.add = add ? (const float *) add->data : nullptr; pp.gate = add ? (const float *) gate->data : nullptr;
        pp.sbo_t = bo->nb[1]/sizeof(float); pp.sres_t = D; // res is contiguous
        pp.sp0 = pl->nb[0]/sizeof(float); pp.sp1 = pl->nb[1]/sizeof(float);
        pp.sadd_t = add ? add->nb[1]/sizeof(float) : 0; pp.sg1 = add ? gate->nb[1]/sizeof(float) : 0;
        pp.sig = ggml_get_op_params_i32(post, 0); pp.p_in = ggml_get_op_params_f32(post, 1); pp.p_out = ggml_get_op_params_f32(post, 2);
    }
    // lo in scratch: act is then never written, so nothing past the run may read it
    if (lo_scratch && hcf_read_after(cgraph, idx[5], act)) {
        return 0;
    }

    const int64_t n_inj = inj ? inj->ne[0] : 0;
    ggml_cuda_pool_alloc<float> rs_tmp(ctx.pool(), nt*HCF_HC);
    ggml_cuda_pool_alloc<float> lo_tmp(ctx.pool());
    ggml_cuda_pool_alloc<float> inj_tmp(ctx.pool());
    float * lo_d  = lo_scratch  ? lo_tmp.alloc(R*nt) : (float *) act->data;
    float * inj_d = inj ? (inj_scratch ? inj_tmp.alloc(n_inj*nt) : (float *) inj->data) : nullptr;

    const float eps   = ggml_get_op_params_f32(rn, 0);
    const float scale = ggml_get_op_params_f32(pre, 0);
    cudaStream_t st = ctx.stream();
    const int64_t sx_t = x->nb[2]/sizeof(float), sxn_t = D; // mul is contiguous [n_embd, hc, nt]
    const int64_t slo_t = R, smx_t = pre->nb[1]/sizeof(float);
    const int64_t sinj_t = inj ? (inj_scratch ? n_inj : (int64_t) (inj->nb[1]/sizeof(float))) : 0;
    // rows per block: GGML_CUDA_HC_MIX_RPB1 (one token, default 2) and GGML_CUDA_HC_MIX_RPBN (more, default 4; HIP 2)
    static const int rpb1 = getenv("GGML_CUDA_HC_MIX_RPB1") ? atoi(getenv("GGML_CUDA_HC_MIX_RPB1")) : 2;
    // HIP default 2 (llama-moe, 7900 XTX kernel A at 4 tokens: 12.4 us vs 13.4 at 4 rows); CUDA keeps 4
#if defined(GGML_USE_HIP)
    static const int rpbn = getenv("GGML_CUDA_HC_MIX_RPBN") ? atoi(getenv("GGML_CUDA_HC_MIX_RPBN")) : 2;
#else
    static const int rpbn = getenv("GGML_CUDA_HC_MIX_RPBN") ? atoi(getenv("GGML_CUDA_HC_MIX_RPBN")) : 4;
#endif
    const int rpb = nt == 1 ? rpb1 : rpbn;
    const int n_rows_a = (int) (R + n_inj);
    const dim3 ga((n_rows_a + rpb - 1)/rpb), gb((n_embd + HCF_B_E - 1)/HCF_B_E);
    const nv_bfloat16 * wi_d = inj ? (const nv_bfloat16 *) inj->src[0]->data : nullptr;
    // scratch inject is packed [n_inj, nt]; the copy needs the destination packed the same way
    if (inj_scratch && sinj_t != (int64_t) (inj->nb[1]/sizeof(float))) {
        return 0;
    }
    const size_t shb = R*nt*sizeof(float);
#define HCF_LAUNCH_A(NT, RPB) \
    hc_mix_a<NT, RPB><<<ga, HCF_A_THREADS, 0, st>>>(pp, (float *) x->data, (const float *) mul->src[1]->data, \
        (const nv_bfloat16 *) wd->data, wi_d, rs_tmp.get(), lo_d, inj_d, (int) n_embd, (int) R, n_rows_a, eps, sx_t, slo_t, sinj_t)
#define HCF_LAUNCH(NT) \
    switch (rpb) { \
        case 4:  HCF_LAUNCH_A(NT, 4); break; \
        case 8:  HCF_LAUNCH_A(NT, 8); break; \
        default: HCF_LAUNCH_A(NT, 2); break; \
    } \
    hc_mix_b<NT><<<gb, HCF_B_E*32, shb, st>>>((const float *) x->data, (const float *) mul->src[1]->data, rs_tmp.get(), \
        (const float *) lo_d, (const nv_bfloat16 *) wu->data, (float *) mul->data, (float *) pre->data, \
        inj_scratch ? inj_d : nullptr, inj ? (float *) inj->data : nullptr, (int) (n_inj*nt), \
        (int) n_embd, (int) R, scale, sx_t, sxn_t, slo_t, smx_t)
    switch (nt) {
        case 1: HCF_LAUNCH(1); break;
        case 2: HCF_LAUNCH(2); break;
        case 3: HCF_LAUNCH(3); break;
        case 4: HCF_LAUNCH(4); break;
        case 5: HCF_LAUNCH(5); break;
        case 6: HCF_LAUNCH(6); break;
        case 7: HCF_LAUNCH(7); break;
        case 8: HCF_LAUNCH(8); break;
    }
#undef HCF_LAUNCH
#undef HCF_LAUNCH_A
    CUDA_CHECK(cudaGetLastError());
    return last - i; // nodes to skip
}
