#pragma once
// hybrid mode, i-quant experts (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S): one weight row against several tokens' Q8_K
// activations. ggml's AVX2 dot products for these formats take one activation: every token re-does the codebook lookups
// and the sign expansion. Here each 32-value chunk is decoded once (grid magnitudes, a +-1 sign vector, the 16-bit scale of
// each half) and every token applies it with a load, a sign, a maddubs, a madd and an add. The integer sums are ggml's
// (ggml-cpu/arch/x86/quants.c) lane for lane, and the float steps are too (the same fmadd per 256-value block, the same
// horizontal sum and final factor), so the results are bit-identical to ggml's vec_dot.
//
// llama_moe_iq::select(type) returns the kernel for a type (nullptr: not covered, or the CPU has no AVX2/FMA/F16C).

#include "ggml.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "../ggml/src/ggml-common.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define LLAMA_MOE_IQ_X86 1
#endif

namespace llama_moe_iq {

// out[t] = row . y[t] for t < nt (1 <= nt <= 8); n = row length (a multiple of 256), y[t] = Q8_K rows of n values
using dot_mt_t = void (*)(int64_t n, const void * row, const void * const * y, int nt, float * out);

#ifdef LLAMA_MOE_IQ_X86

// 128 x 8 bytes: the 8 signs (+1 / -1) of each 7-bit sign index; the 8th sign makes the count of negatives even
// (ksigns_iq2xs, ggml's keven_signs_q2xs)
struct even_signs_t {
    uint64_t v[128];
    constexpr even_signs_t() : v() {
        for (int k = 0; k < 128; ++k) {
            int par = 0;
            for (int b = 0; b < 7; ++b) {
                par ^= (k >> b) & 1;
            }
            const int m = k | (par << 7);
            uint64_t w = 0;
            for (int b = 0; b < 8; ++b) {
                w |= (uint64_t) ((m >> b) & 1 ? 0xFF : 0x01) << (8*b);
            }
            v[k] = w;
        }
    }
};
static constexpr even_signs_t k_even_signs;
static_assert(k_even_signs.v[0] == 0x0101010101010101ull && k_even_signs.v[1] == 0xFF010101010101FFull, "even signs");
inline const uint64_t * even_signs() {
    return k_even_signs.v;
}

#if defined(__clang__)
#pragma clang attribute push (__attribute__((target("avx2,fma,f16c"))), apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")
#endif

struct chunk {
    __m256i q;   // the 32 magnitudes (unsigned bytes, maddubs' first operand)
    __m256i s;   // their signs, +1 / -1 bytes
    __m256i sc;  // the scale of each int16 lane of the maddubs result: lanes 0-7 values 0-15, lanes 8-15 values 16-31
};

// 4 sign bytes (bit set = negative) -> 32 bytes of +1 / -1
static inline __m256i signs_from_bits(const uint8_t * p) {
    static const uint8_t k_mask1[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
    static const uint8_t k_mask2[32] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
                                        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint32_t w;
    memcpy(&w, p, 4);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i *) k_mask2);
    __m256i a = _mm256_set1_epi32((int) w);
    a = _mm256_and_si256(_mm256_shuffle_epi8(a, _mm256_loadu_si256((const __m256i *) k_mask1)), mask2);
    return _mm256_or_si256(_mm256_cmpeq_epi8(a, mask2), _mm256_set1_epi8(1));
}

static inline __m256i scale_halves(int lo, int hi) {
    return _mm256_setr_m128i(_mm_set1_epi16((short) lo), _mm_set1_epi16((short) hi));
}

// the four 7-bit sign indices of a 32-value chunk (bits 0-6, 7-13, 14-20, 21-27)
static inline __m128i sign_idx4(uint32_t a) {
    return _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int) a), _mm_setr_epi32(0, 7, 14, 21)), _mm_set1_epi32(127));
}

struct f_iq2_xxs {
    using block = block_iq2_xxs;
    static constexpr float factor = 0.125f;
    static inline void decode(const block & x, int j, chunk & c) {
        const uint64_t * s64 = even_signs();
        uint32_t a[2];
        memcpy(a, x.qs + 4*j, 8);
        const uint8_t * g = (const uint8_t *) a;
        c.q  = _mm256_set_epi64x(iq2xxs_grid[g[3]], iq2xxs_grid[g[2]], iq2xxs_grid[g[1]], iq2xxs_grid[g[0]]);
        c.s  = _mm256_set_epi64x(s64[(a[1] >> 21) & 127], s64[(a[1] >> 14) & 127], s64[(a[1] >> 7) & 127], s64[a[1] & 127]);
        c.sc = _mm256_set1_epi16((short) (2*(a[1] >> 28) + 1));
    }
    static inline void decode_g(const block & x, int j, chunk & c) {
        uint32_t a[2];
        memcpy(a, x.qs + 4*j, 8);
        c.q  = _mm256_i32gather_epi64((const long long *) iq2xxs_grid, _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) a[0])), 8);
        c.s  = _mm256_i32gather_epi64((const long long *) even_signs(), sign_idx4(a[1]), 8);
        c.sc = _mm256_set1_epi16((short) (2*(a[1] >> 28) + 1));
    }
};

struct f_iq2_xs {
    using block = block_iq2_xs;
    static constexpr float factor = 0.125f;
    static inline void decode(const block & x, int j, chunk & c) {
        const uint64_t * s64 = even_signs();
        const uint16_t * q = x.qs + 4*j;
        c.q  = _mm256_set_epi64x(iq2xs_grid[q[3] & 511], iq2xs_grid[q[2] & 511], iq2xs_grid[q[1] & 511], iq2xs_grid[q[0] & 511]);
        c.s  = _mm256_set_epi64x(s64[q[3] >> 9], s64[q[2] >> 9], s64[q[1] >> 9], s64[q[0] >> 9]);
        c.sc = scale_halves(2*(x.scales[j] & 15) + 1, 2*(x.scales[j] >> 4) + 1);
    }
    static inline void decode_g(const block & x, int j, chunk & c) {
        const __m128i q = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *) (x.qs + 4*j)));
        c.q  = _mm256_i32gather_epi64((const long long *) iq2xs_grid, _mm_and_si128(q, _mm_set1_epi32(511)), 8);
        c.s  = _mm256_i32gather_epi64((const long long *) even_signs(), _mm_srli_epi32(q, 9), 8);
        c.sc = scale_halves(2*(x.scales[j] & 15) + 1, 2*(x.scales[j] >> 4) + 1);
    }
};

struct f_iq2_s {
    using block = block_iq2_s;
    static constexpr float factor = 0.125f;
    static inline void decode(const block & x, int j, chunk & c) {
        const uint8_t * qs = x.qs + 4*j;
        const int qh = x.qh[j];
        c.q  = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh << 2) & 0x300)], iq2s_grid[qs[2] | ((qh << 4) & 0x300)],
                                 iq2s_grid[qs[1] | ((qh << 6) & 0x300)], iq2s_grid[qs[0] | ((qh << 8) & 0x300)]);
        c.s  = signs_from_bits(x.qs + QK_K/8 + 4*j);
        c.sc = scale_halves(2*(x.scales[j] & 15) + 1, 2*(x.scales[j] >> 4) + 1);
    }
    static inline void decode_g(const block & x, int j, chunk & c) {
        uint32_t w;
        memcpy(&w, x.qs + 4*j, 4);
        const __m128i hi = _mm_and_si128(_mm_sllv_epi32(_mm_set1_epi32(x.qh[j]), _mm_setr_epi32(8, 6, 4, 2)), _mm_set1_epi32(0x300));
        c.q  = _mm256_i32gather_epi64((const long long *) iq2s_grid, _mm_or_si128(_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) w)), hi), 8);
        c.s  = signs_from_bits(x.qs + QK_K/8 + 4*j);
        c.sc = scale_halves(2*(x.scales[j] & 15) + 1, 2*(x.scales[j] >> 4) + 1);
    }
};

struct f_iq3_xxs {
    using block = block_iq3_xxs;
    static constexpr float factor = 0.25f;
    static inline void decode(const block & x, int j, chunk & c) {
        const uint64_t * s64 = even_signs();
        const uint8_t * q3 = x.qs + 8*j;
        uint32_t a;
        memcpy(&a, x.qs + QK_K/4 + 4*j, 4);
        c.q  = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
        c.s  = _mm256_set_epi64x(s64[(a >> 21) & 127], s64[(a >> 14) & 127], s64[(a >> 7) & 127], s64[a & 127]);
        c.sc = _mm256_set1_epi16((short) (2*(a >> 28) + 1));
    }
    static inline void decode_g(const block & x, int j, chunk & c) {
        uint32_t a;
        memcpy(&a, x.qs + QK_K/4 + 4*j, 4);
        c.q  = _mm256_i32gather_epi32((const int *) iq3xxs_grid, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) (x.qs + 8*j))), 4);
        c.s  = _mm256_i32gather_epi64((const long long *) even_signs(), sign_idx4(a), 8);
        c.sc = _mm256_set1_epi16((short) (2*(a >> 28) + 1));
    }
};

struct f_iq3_s {
    using block = block_iq3_s;
    static constexpr float factor = 1.0f;
    static inline void decode(const block & x, int j, chunk & c) {
        const uint8_t * qs = x.qs + 8*j;
        const int qh = x.qh[j];
        uint32_t g[8];
        for (int k = 0; k < 8; ++k) {
            g[k] = iq3s_grid[qs[k] | (((qh >> k) & 1) << 8)];
        }
        c.q  = _mm256_loadu_si256((const __m256i *) g);
        c.s  = signs_from_bits(x.signs + 4*j);
        c.sc = _mm256_set1_epi16((short) (2*((x.scales[j/2] >> (4*(j & 1))) & 15) + 1));
    }
    static inline void decode_g(const block & x, int j, chunk & c) {
        const __m256i hi  = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(x.qh[j]), _mm256_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1)),
                                             _mm256_set1_epi32(256));
        const __m256i idx = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) (x.qs + 8*j))), hi);
        c.q  = _mm256_i32gather_epi32((const int *) iq3s_grid, idx, 4);
        c.s  = signs_from_bits(x.signs + 4*j);
        c.sc = _mm256_set1_epi16((short) (2*((x.scales[j/2] >> (4*(j & 1))) & 15) + 1));
    }
};

// ggml's hsum_float_8
static inline float hsum8(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

template <class F, int NT, bool G>
static inline void dot_nt(int64_t n, const typename F::block * x, const block_q8_K * const * y, float * out) {
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) {
        acc[t] = _mm256_setzero_ps();
    }
    for (int64_t i = 0; i < n/QK_K; ++i) {
        __m256i isum[NT];
        for (int t = 0; t < NT; ++t) {
            isum[t] = _mm256_setzero_si256();
        }
        for (int j = 0; j < QK_K/32; ++j) {
            chunk c;
            if (G) {
                F::decode_g(x[i], j, c);
            } else {
                F::decode(x[i], j, c);
            }
            for (int t = 0; t < NT; ++t) {
                const __m256i a = _mm256_loadu_si256((const __m256i *) (y[t][i].qs + 32*j));
                isum[t] = _mm256_add_epi32(isum[t], _mm256_madd_epi16(_mm256_maddubs_epi16(c.q, _mm256_sign_epi8(a, c.s)), c.sc));
            }
        }
        const float dx = _cvtsh_ss(x[i].d);
        for (int t = 0; t < NT; ++t) {
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx*y[t][i].d), _mm256_cvtepi32_ps(isum[t]), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        out[t] = F::factor*hsum8(acc[t]);
    }
}

// all tokens in one pass for nt <= 8 (decode once per row). G: the grid and sign lookups by vpgather (faster on some cores)
template <class F, bool G>
static void dot_mt(int64_t n, const void * row, const void * const * y, int nt, float * out) {
    const auto * x  = (const typename F::block *) row;
    const auto * yq = (const block_q8_K * const *) y;
    switch (nt) {
        case 8: dot_nt<F, 8, G>(n, x, yq, out); break;
        case 7: dot_nt<F, 7, G>(n, x, yq, out); break;
        case 6: dot_nt<F, 6, G>(n, x, yq, out); break;
        case 5: dot_nt<F, 5, G>(n, x, yq, out); break;
        case 4: dot_nt<F, 4, G>(n, x, yq, out); break;
        case 3: dot_nt<F, 3, G>(n, x, yq, out); break;
        case 2: dot_nt<F, 2, G>(n, x, yq, out); break;
        case 1: dot_nt<F, 1, G>(n, x, yq, out); break;
        default: break;
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif

#endif // LLAMA_MOE_IQ_X86

// the kernel for a weight type, nullptr if it is not covered or the CPU lacks AVX2/FMA/F16C (LLAMA_MOE_CPU_IQ=0: always nullptr)
inline dot_mt_t select(ggml_type type, int gather = -1) {
#ifdef LLAMA_MOE_IQ_X86
    if (const char * e = getenv("LLAMA_MOE_CPU_IQ"); e && atoi(e) == 0) {
        return nullptr;
    }
    __builtin_cpu_init();
    if (!(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c"))) {
        return nullptr;
    }
    if (gather < 0) {   // LLAMA_MOE_CPU_IQ_GATHER=1: the vpgather decode
        const char * e = getenv("LLAMA_MOE_CPU_IQ_GATHER");
        gather = e ? atoi(e) : 0;
    }
#define LLAMA_MOE_IQ_PICK(F) (gather ? dot_mt<F, true> : dot_mt<F, false>)
    switch (type) {
        case GGML_TYPE_IQ2_XXS: return LLAMA_MOE_IQ_PICK(f_iq2_xxs);
        case GGML_TYPE_IQ2_XS:  return LLAMA_MOE_IQ_PICK(f_iq2_xs);
        case GGML_TYPE_IQ2_S:   return LLAMA_MOE_IQ_PICK(f_iq2_s);
        case GGML_TYPE_IQ3_XXS: return LLAMA_MOE_IQ_PICK(f_iq3_xxs);
        case GGML_TYPE_IQ3_S:   return LLAMA_MOE_IQ_PICK(f_iq3_s);
        default: break;
    }
#undef LLAMA_MOE_IQ_PICK
#else
    (void) type;
#endif
    return nullptr;
}

} // namespace llama_moe_iq
