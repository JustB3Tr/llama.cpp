#include "llama-moe-cache.h"
#include "llama-moe-iq.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <deque>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <ctime>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <string>
#ifdef __linux__
#include <pthread.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define LLAMA_MOE_Q2_X86 1
#endif

// physical cores among the CPUs in this process's affinity mask: distinct core_cpus_list (SMT siblings) on Linux;
// hardware threads / 2 elsewhere or if the topology cannot be read
static int moe_physical_cores() {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        std::set<std::string> cores;
        bool ok = true;
        for (int c = 0; c < CPU_SETSIZE && ok; ++c) {
            if (!CPU_ISSET(c, &set)) {
                continue;
            }
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/core_cpus_list");
            if (!f) {
                f.open("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
            }
            std::string l;
            ok = f && std::getline(f, l) && !l.empty();
            cores.insert(l);
        }
        if (ok && !cores.empty()) {
            return (int) cores.size();
        }
    }
#endif
    return std::max(1, (int) std::thread::hardware_concurrency()/2);
}

// hybrid mode, Q2_0 experts: a faster dot product than ggml's for one Q2_0 row against Q8_0 activations. The activations are
// permuted once (per token, or per pair for the down projection) into the weights' bit-plane order: byte b of a 64-weight block
// holds weights 4b+k at bits 2k, so plane k of the 16 bytes is (q >> 2k) & 3 and needs no shuffle. Same integer products and the
// same scales as ggml's kernel, only the float summation order differs.
namespace {
struct q2_block { ggml_fp16_t d; uint8_t qs[16]; };
struct q8_block { ggml_fp16_t d; int8_t  qs[32]; };
static_assert(sizeof(q2_block) == 18 && sizeof(q8_block) == 34, "unexpected Q2_0 / Q8_0 block size");

// one 64-element block of prepared activations: planes [0|1] and [2|3] (each 16 elements of 2 Q8_0 halves), the scale of each
// int32 lane after the dot products (dwords 0,1,4,5 come from the first Q8_0 block, 2,3,6,7 from the second), and the offset
// c = sum over the block of y (the weights are v - 1)
struct q2_act { int8_t q[64]; float dyv[8]; float c; };

void q2_prepare(const q8_block * y, int64_t n, q2_act * a) {
    for (int64_t i = 0; i < n/64; ++i) {
        const q8_block & y0 = y[2*i];
        const q8_block & y1 = y[2*i + 1];
        const float d0 = ggml_fp16_to_fp32(y0.d), d1 = ggml_fp16_to_fp32(y1.d);
        int s0 = 0, s1 = 0;
        for (int b = 0; b < 16; ++b) {
            for (int k = 0; k < 4; ++k) {
                const int e = 4*b + k;
                a[i].q[k*16 + b] = e < 32 ? y0.qs[e] : y1.qs[e - 32];
            }
        }
        for (int k = 0; k < 32; ++k) {
            s0 += y0.qs[k];
            s1 += y1.qs[k];
        }
        const float dyv[8] = { d0, d0, d1, d1, d0, d0, d1, d1 };
        memcpy(a[i].dyv, dyv, sizeof(dyv));
        a[i].c = d0*s0 + d1*s1;
    }
}

#ifdef LLAMA_MOE_Q2_X86
// the same body for each target (an always_inline template cannot be compiled for a target its callers add)
#define LLAMA_MOE_Q2_DOT_BODY(DOT)                                                                                         \
    __m256 acc = _mm256_setzero_ps();                                                                                      \
    float corr = 0.0f;                                                                                                     \
    const __m256i sh0 = _mm256_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2);                                                         \
    const __m256i sh1 = _mm256_setr_epi32(4, 4, 4, 4, 6, 6, 6, 6);                                                         \
    const __m256i m3  = _mm256_set1_epi8(3);                                                                               \
    for (int64_t i = 0; i < n/64; ++i) {                                                                                   \
        const __m256i q  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) x[i].qs));                        \
        const __m256i vA = _mm256_and_si256(_mm256_srlv_epi32(q, sh0), m3);                                                \
        const __m256i vB = _mm256_and_si256(_mm256_srlv_epi32(q, sh1), m3);                                                \
        const __m256i yA = _mm256_loadu_si256((const __m256i *) (a[i].q));                                                 \
        const __m256i yB = _mm256_loadu_si256((const __m256i *) (a[i].q + 32));                                            \
        const __m256i s  = DOT;                                                                                            \
        const float dx = _cvtsh_ss(x[i].d);                                                                                \
        acc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dx), _mm256_loadu_ps(a[i].dyv)), _mm256_cvtepi32_ps(s), acc);  \
        corr += dx*a[i].c;                                                                                                 \
    }                                                                                                                      \
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));                                     \
    r = _mm_hadd_ps(r, r);                                                                                                 \
    r = _mm_hadd_ps(r, r);                                                                                                 \
    return _mm_cvtss_f32(r) - corr;

__attribute__((target("avx2,fma,f16c")))
float q2_dot_avx2(int64_t n, const q2_block * x, const q2_act * a) {
    LLAMA_MOE_Q2_DOT_BODY(_mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(vA, yA), _mm256_maddubs_epi16(vB, yB)), _mm256_set1_epi16(1)))
}

__attribute__((target("avx2,fma,f16c,avxvnni")))
float q2_dot_avxvnni(int64_t n, const q2_block * x, const q2_act * a) {
    LLAMA_MOE_Q2_DOT_BODY(_mm256_dpbusd_avx_epi32(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), vA, yA), vB, yB))
}
#undef LLAMA_MOE_Q2_DOT_BODY

// 4 rows against one activation: the activation loads are shared, each row's arithmetic is q2_dot's (same results)
#define LLAMA_MOE_Q2_DOT4_BODY(DOT)                                                                                        \
    __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };               \
    float corr[4] = { 0.0f, 0.0f, 0.0f, 0.0f };                                                                            \
    const __m256i sh0 = _mm256_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2);                                                         \
    const __m256i sh1 = _mm256_setr_epi32(4, 4, 4, 4, 6, 6, 6, 6);                                                         \
    const __m256i m3  = _mm256_set1_epi8(3);                                                                               \
    for (int64_t i = 0; i < n/64; ++i) {                                                                                   \
        const __m256i yA  = _mm256_loadu_si256((const __m256i *) (a[i].q));                                                \
        const __m256i yB  = _mm256_loadu_si256((const __m256i *) (a[i].q + 32));                                           \
        const __m256  dyv = _mm256_loadu_ps(a[i].dyv);                                                                     \
        const float   c   = a[i].c;                                                                                        \
        _Pragma("GCC unroll 4")                                                                                            \
        for (int r = 0; r < 4; ++r) {                                                                                      \
            const q2_block & xb = *(const q2_block *) ((const char *) x + r*row_stride + i*sizeof(q2_block));            \
            const __m256i q  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) xb.qs));                     \
            const __m256i vA = _mm256_and_si256(_mm256_srlv_epi32(q, sh0), m3);                                            \
            const __m256i vB = _mm256_and_si256(_mm256_srlv_epi32(q, sh1), m3);                                            \
            const __m256i s  = DOT;                                                                                        \
            const float dx = _cvtsh_ss(xb.d);                                                                              \
            acc[r] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dx), dyv), _mm256_cvtepi32_ps(s), acc[r]);               \
            corr[r] += dx*c;                                                                                               \
        }                                                                                                                  \
    }                                                                                                                      \
    for (int r = 0; r < 4; ++r) {                                                                                          \
        __m128 v = _mm_add_ps(_mm256_castps256_ps128(acc[r]), _mm256_extractf128_ps(acc[r], 1));                           \
        v = _mm_hadd_ps(v, v);                                                                                             \
        v = _mm_hadd_ps(v, v);                                                                                             \
        out[r] = _mm_cvtss_f32(v) - corr[r];                                                                               \
    }

__attribute__((target("avx2,fma,f16c")))
void q2_dot4_avx2(int64_t n, const q2_block * x, size_t row_stride, const q2_act * a, float * out) {
    LLAMA_MOE_Q2_DOT4_BODY(_mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(vA, yA), _mm256_maddubs_epi16(vB, yB)), _mm256_set1_epi16(1)))
}

__attribute__((target("avx2,fma,f16c,avxvnni")))
void q2_dot4_avxvnni(int64_t n, const q2_block * x, size_t row_stride, const q2_act * a, float * out) {
    LLAMA_MOE_Q2_DOT4_BODY(_mm256_dpbusd_avx_epi32(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), vA, yA), vB, yB))
}
#undef LLAMA_MOE_Q2_DOT4_BODY
#endif

using q2_dot_t = float (*)(int64_t, const q2_block *, const q2_act *);
// rows x, x + row_stride, x + 2*row_stride, x + 3*row_stride (bytes) against one activation, results in out[4]
using q2_dot4_t = void (*)(int64_t, const q2_block *, size_t, const q2_act *, float *);

q2_dot4_t q2_dot4_select() {
    if (const char * e = getenv("LLAMA_MOE_CPU_Q2"); e && atoi(e) == 0) {
        return nullptr;
    }
    if (const char * e = getenv("LLAMA_MOE_CPU_DOT4"); e && atoi(e) == 0) {
        return nullptr;
    }
#ifdef LLAMA_MOE_Q2_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) {
        if (__builtin_cpu_supports("avxvnni")) {
            return q2_dot4_avxvnni;
        }
        return q2_dot4_avx2;
    }
#endif
    return nullptr;
}
const char * q2_dot_name();

// the fast Q2_0 dot product this CPU supports, nullptr if none (LLAMA_MOE_CPU_Q2=0 turns it off)
q2_dot_t q2_dot_select() {
    if (const char * e = getenv("LLAMA_MOE_CPU_Q2"); e && atoi(e) == 0) {
        return nullptr;
    }
#ifdef LLAMA_MOE_Q2_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) {
        if (__builtin_cpu_supports("avxvnni")) {
            return q2_dot_avxvnni;
        }
        return q2_dot_avx2;
    }
#endif
    return nullptr;
}

// the multi-token i-quant kernel of a weight type (llama-moe-iq.h), nullptr if none, and the fewest tokens of an expert it
// takes (2: one token goes to ggml's vec_dot). Chosen once per type by timing on this CPU, single thread, 256 rows of 4096
// (in cache: the decode is compute-bound): the vpgather or the shuffle decode, whichever is faster at 2 tokens, and 1 token
// too when it beats ggml by 5% (cfl1 i7-8700, IQ3_S at 1 token: gather 112 us vs ggml 186 per 1536x4096 expert, 6 threads;
// Zen 3: gather 0.7x). LLAMA_MOE_CPU_IQ_GATHER=0/1 fixes the decode, LLAMA_MOE_CPU_IQ_MIN=1/2 the token floor
struct iq_pick { llama_moe_iq::dot_mt_t f = nullptr; int min_nt = 2; };

static iq_pick iq_calibrate(ggml_type t) {
    iq_pick r;
    llama_moe_iq::dot_mt_t f[2] = { llama_moe_iq::select(t, 0), llama_moe_iq::select(t, 1) };
    if (!f[0] || !f[1]) {
        return r;
    }
    const auto * tr = ggml_get_type_traits_cpu(t);
    const auto * ty = ggml_get_type_traits_cpu(tr->vec_dot_type);
    if (!tr->vec_dot || !ty->from_float) {
        r.f = f[0];
        return r;
    }
    const int64_t n = 4096, rows = 256;
    const size_t rs = ggml_row_size(t, n), bs = ggml_type_size(t);
    std::vector<uint8_t> w(rs*rows);
    uint32_t h = 12345;
    for (auto & b : w) { h = h*1664525u + 1013904223u; b = (uint8_t) (h >> 24); }
    const ggml_fp16_t one = ggml_fp32_to_fp16(0.01f);
    for (size_t o = 0; o < w.size(); o += bs) { memcpy(&w[o], &one, sizeof(one)); } // every i-quant block starts with its fp16 d
    std::vector<float> xf(n);
    for (auto & v : xf) { h = h*1664525u + 1013904223u; v = (float) (h >> 8)/(1 << 24) - 0.5f; }
    std::vector<uint8_t> y(ggml_row_size(tr->vec_dot_type, n)*2);
    ty->from_float(xf.data(), y.data(), n);
    ty->from_float(xf.data(), y.data() + y.size()/2, n);
    const void * ys[2] = { y.data(), y.data() + y.size()/2 };
    float out[2];
    static volatile float sink;
    auto best = [&](auto && fn) {
        double b = 1e30;
        for (int rep = 0; rep < 5; ++rep) {
            const int64_t t0 = ggml_time_us();
            for (int64_t i = 0; i < rows; ++i) { fn(w.data() + i*rs); sink = out[0]; }
            b = std::min(b, (double) (ggml_time_us() - t0));
        }
        return b;
    };
    const double tg1 = best([&](const void * row) { tr->vec_dot((int) n, out, 0, row, 0, ys[0], 0, 1); });
    double t1[2], t2[2];
    for (int g = 0; g < 2; ++g) {
        t1[g] = best([&](const void * row) { f[g](n, row, ys, 1, out); });
        t2[g] = best([&](const void * row) { f[g](n, row, ys, 2, out); });
    }
    int g = t2[1] < t2[0] ? 1 : 0;
    if (const char * e = getenv("LLAMA_MOE_CPU_IQ_GATHER")) {
        g = atoi(e) != 0;
    }
    r.f = f[g];
    r.min_nt = t1[g] < 0.95*tg1 ? 1 : 2;
    if (const char * e = getenv("LLAMA_MOE_CPU_IQ_MIN")) {
        r.min_nt = std::clamp(atoi(e), 1, 2);
    }
    LLAMA_LOG_INFO("llama_moe_cache: CPU %s kernel: %s decode, from %d token%s (per 256 rows: ggml %.0f us, shuffle %.0f/%.0f, gather %.0f/%.0f at 1/2 tokens)\n",
        ggml_type_name(t), g ? "gather" : "shuffle", r.min_nt, r.min_nt > 1 ? "s" : "", tg1, t1[0], t2[0], t1[1], t2[1]);
    return r;
}

static iq_pick iq_choose(ggml_type t) {
    static std::mutex mu;
    static std::vector<int8_t> done(GGML_TYPE_COUNT, 0);
    static std::vector<iq_pick> tab(GGML_TYPE_COUNT);
    if (t < 0 || t >= GGML_TYPE_COUNT) {
        return iq_pick();
    }
    std::lock_guard<std::mutex> lk(mu);
    if (!done[t]) {
        tab[t] = llama_moe_iq::select(t) ? iq_calibrate(t) : iq_pick();
        done[t] = 1;
    }
    return tab[t];
}

llama_moe_iq::dot_mt_t iq_mt(ggml_type t) {
    return iq_choose(t).f;
}

const char * q2_dot_name() {
#ifdef LLAMA_MOE_Q2_X86
    const q2_dot_t f = q2_dot_select();
    return f == q2_dot_avxvnni ? " (Q2_0 fast path: AVX-VNNI)" : f == q2_dot_avx2 ? " (Q2_0 fast path: AVX2)" : "";
#else
    return "";
#endif
}
} // namespace

namespace {

// LRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
// lookahead prefetch: the router dot products on the host (float weights and input)
typedef float (*la_dot_t)(const float *, const float *, int64_t);
static float la_dot_ref(const float * a, const float * b, int64_t n) {
    float r = 0.0f;
    for (int64_t i = 0; i < n; ++i) r += a[i]*b[i];
    return r;
}
#ifdef LLAMA_MOE_Q2_X86
__attribute__((target("avx2,fma")))
static float la_dot_avx2(const float * a, const float * b, int64_t n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    h = _mm_hadd_ps(h, h);
    h = _mm_hadd_ps(h, h);
    float r = _mm_cvtss_f32(h);
    for (; i < n; ++i) r += a[i]*b[i];
    return r;
}
#endif
static la_dot_t la_dot_select() {
#ifdef LLAMA_MOE_Q2_X86
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return la_dot_avx2;
    }
#endif
    return la_dot_ref;
}

struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;
    std::vector<uint64_t> * hot = nullptr; // LLAMA_MOE_HOT_SAVE: [il*n_expert + e] the calls that selected each expert

    void count_hot(int32_t il) {
        if (hot) {
            for (int32_t e : uniq) {
                (*hot)[(size_t) il*n_expert + e]++;
            }
        }
    }

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
        if (const char * fq = getenv("LLAMA_MOE_VRAM_FREQ")) {
            const double d = atof(fq);
            if (d > 0.0 && d < 1.0) {
                freq_ld = -std::log(d);
                freq_dt = 1.0/std::max(1, n_layer);
                fscore.assign((size_t) n_layer*n_expert, -INFINITY);
            }
        }
    }

    // LLAMA_MOE_VRAM_FREQ=<decay per decode step, e.g. 0.99> (hybrid decode, opt-in): the cache keeps the experts of the highest
    // decayed selection count instead of the most recent ones (bench/tier_sim.py: the same hit rate as LRU with 2-4x fewer
    // uploads). Each decode call counts its experts; a miss that the share admits gets a slot only if it now scores above the
    // coldest resident (which it evicts), otherwise the CPU computes it. Scores are logs normalized to time 0
    // (ln s(t) = fscore - t*ln(1/decay)), so their order never changes and a lazy min-heap of the residents works
    double freq_ld = 0.0, freq_t = 0.0, freq_dt = 0.0;
    std::vector<double> fscore;                         // [il*n_expert + e]
    std::vector<std::pair<double, int32_t>> fheap;      // (score, slot) of the residents, lazy min-heap
    std::vector<std::pair<double, int32_t>> fskip;
    size_t n_freq_rejected = 0;

    double freq_bump(int32_t key) {
        const double x = freq_t*freq_ld;
        double & o = fscore[key];
        o = std::isinf(o) ? x : std::max(o, x) + std::log1p(std::exp(-std::fabs(o - x)));
        return o;
    }
    void freq_rebuild() {
        fheap.clear();
        for (int32_t s = 0; s < n_slots; ++s) {
            if (key_of[s] >= 0) {
                fheap.push_back({ fscore[key_of[s]], s });
            }
        }
        std::make_heap(fheap.begin(), fheap.end(), std::greater<>());
    }
    // the slot for a miss of layer il scoring sc: an empty one, or the coldest resident not used by this call if it scores
    // below sc; -1: keep the cache (the CPU computes the miss)
    int32_t freq_victim(int32_t il, double sc) {
        if (key_of[head] < 0) {
            return head;
        }
        if (fheap.empty() || fheap.size() > 4*(size_t) n_slots) {
            freq_rebuild();
        }
        fskip.clear();
        int32_t res = -1;
        while (!fheap.empty()) {
            const auto top = fheap.front();
            const int32_t k = key_of[top.second];
            std::pop_heap(fheap.begin(), fheap.end(), std::greater<>());
            fheap.pop_back();
            if (k < 0) {
                res = top.second;
                break;
            }
            if (fscore[k] != top.first) {
                continue; // stale
            }
            if (k / n_expert == il && seen[k % n_expert] == seen_gen) {
                fskip.push_back(top); // used by this call
                continue;
            }
            if (top.first < sc) {
                res = top.second;
            } else {
                fskip.push_back(top);
            }
            break;
        }
        for (const auto & t : fskip) {
            fheap.push_back(t);
            std::push_heap(fheap.begin(), fheap.end(), std::greater<>());
        }
        return res;
    }

    // move slot s to the tail (most recently used)
    void touch(int32_t s) {
        if (s == tail) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    // take slot s out of the list (lent: never a victim) / put it back first in line (free)
    void unlink(int32_t s) {
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail = prev[s];
        }
        prev[s] = next[s] = -1;
    }
    void push_head(int32_t s) {
        prev[s] = -1;
        next[s] = head;
        if (head >= 0) {
            prev[head] = s;
        } else {
            tail = s;
        }
        head = s;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
        int32_t victim = -1; // the key (il*n_expert + e) the slot held before, -1: none
    };

    // give a slot to each expert selected by ids in layer il, the misses evict the least recently used experts
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        count_hot(il);
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // hits go to the tail first, so the head can be evicted below
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = head;
            const int32_t vk = key_of[s];
            if (vk >= 0) {
                slot_map[vk / n_expert][vk % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            touch(s);
            fills.push_back({ e, s, vk });
        }
        return true;
    }

    // hybrid mode: like plan, but only n_admit of the misses get a slot (those selected by the most tokens first),
    // the others are returned in rest and stay uncached. tail_rank > 0 (LLAMA_MOE_TAIL_SKIP, approximate): a missed expert
    // that every token routes at rank >= tail_rank (ids are in router order, n_used per token) goes to skipped instead:
    // neither uploaded nor computed, its contribution is zero (the other weights are not renormalized)
    void plan_hybrid(int32_t il, const int32_t * ids, size_t n_ids, double share, std::vector<fill> & fills, std::vector<int32_t> & rest, size_t & n_hit,
            int32_t n_used = 0, int32_t tail_rank = 0, std::vector<int32_t> * skipped = nullptr) {
        fills.clear();
        rest.clear();
        if (skipped) {
            skipped->clear();
        }
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        count.resize(n_expert);
        if (tail_rank > 0) {
            min_rank.resize(n_expert);
        }
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                count[ids[i]] = 0;
                if (tail_rank > 0) {
                    min_rank[ids[i]] = INT32_MAX;
                }
                uniq.push_back(ids[i]);
            }
            count[ids[i]]++;
            if (tail_rank > 0) {
                min_rank[ids[i]] = std::min(min_rank[ids[i]], (int32_t) (i % n_used));
            }
        }

        count_hot(il);
        int32_t * slots = slot_map[il];
        const bool freq = freq_ld > 0.0 && share > 0.0;
        if (freq) {
            freq_t += freq_dt;
            for (int32_t e : uniq) {
                const double sc = freq_bump(il*n_expert + e);
                if (slots[e] >= 0) {
                    fheap.push_back({ sc, slots[e] });
                    std::push_heap(fheap.begin(), fheap.end(), std::greater<>());
                }
            }
        }

        std::vector<int32_t> & misses = rest;
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            } else if (tail_rank > 0 && skipped && min_rank[e] >= tail_rank) {
                skipped->push_back(e);
            } else {
                misses.push_back(e);
            }
        }
        const size_t n_admit = std::min(misses.size(), (size_t) std::lround(share*misses.size()));
        std::stable_sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) { return count[a] > count[b]; });
        if (freq) {
            // the admitted misses that beat the coldest resident get its slot, the others stay with the CPU
            fkeep.clear();
            for (size_t i = 0; i < misses.size(); ++i) {
                const int32_t e = misses[i];
                const int32_t key = il*n_expert + e;
                const int32_t s = i < n_admit ? freq_victim(il, fscore[key]) : -1;
                if (s < 0) {
                    n_freq_rejected += i < n_admit;
                    fkeep.push_back(e);
                    continue;
                }
                const int32_t vk = key_of[s];
                if (vk >= 0) {
                    slot_map[vk / n_expert][vk % n_expert] = -1;
                }
                key_of[s] = key;
                slots[e] = s;
                touch(s);
                fills.push_back({ e, s, vk });
                fheap.push_back({ fscore[key], s });
                std::push_heap(fheap.begin(), fheap.end(), std::greater<>());
            }
            misses.swap(fkeep);
            return;
        }
        std::sort(misses.begin(), misses.begin() + n_admit);
        for (size_t i = 0; i < n_admit; ++i) {
            const int32_t e = misses[i];
            const int32_t s = head;
            const int32_t vk = key_of[s];
            if (vk >= 0) {
                slot_map[vk / n_expert][vk % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            touch(s);
            fills.push_back({ e, s, vk });
        }
        misses.erase(misses.begin(), misses.begin() + n_admit);
    }

    std::vector<int32_t> fkeep;
    std::vector<int32_t> count; // [n_expert] tokens that select each expert, valid for the experts in uniq
    std::vector<int32_t> min_rank; // [n_expert] best router rank of each expert in uniq (tail skip)
};

// LLAMA_MOE_ARENA_MIB=N (hybrid decode, experts mapped from the model file, Linux): N MiB of host memory (pinned when the
// device has a host buffer type) holding whole experts, all their tensors. A layer's missed experts (the uploads and the
// CPU's) that are not in it are read from the file with O_DIRECT, all at once, in chunks, by a pool of reader threads and
// the caller; the uploads and the CPU job then read them from the arena. On a box whose RAM is far below the model, the
// page faults of a mapped file read one page range at a time (the drive at queue depth ~1); these reads keep it at depth
// 4+. Only decode plans fill it (a prefill never evicts). LRU; the slots used in this and the previous call are not evicted
// (the previous layer's CPU job may still read them). Lossless: an expert that does not fit or fails to read comes from
// the mapping as before.
// LLAMA_MOE_FAULT_THREADS=N (same conditions, no arena): the same pool only touches the missed experts' pages of the mapping,
// in parallel, before the uploads and the CPU job read them: the page cache keeps its capacity, the drive gets depth N+1
struct moe_arena {
    struct part { int fd = -1; uint64_t off = 0; size_t nb2 = 0; const uint8_t * map = nullptr; }; // per (layer, tensor): file, offset of expert 0, bytes, mapping
    struct chunk { int fd; uint64_t off; size_t n; uint8_t * dst; int32_t slot; const uint8_t * src; int32_t pi = -1; };
    bool touch_only = false;
    bool no_wait    = false;  // LLAMA_MOE_FAULT_NOWAIT=1: post the touches and return, the readers of the data fault on the rest
    bool bulk       = false;  // the posted jobs are a shadow pass's read-ahead (touch_bulk)
    const int how   = getenv("LLAMA_MOE_FAULT_HOW") ? atoi(getenv("LLAMA_MOE_FAULT_HOW")) : 2; // see work()
    std::unordered_map<int, int> fd_buf;     // O_DIRECT fd -> a buffered fd of the same file (LLAMA_MOE_FAULT_HOW=3)
    size_t n_bulk = 0, n_bulk_skip = 0;
    volatile uint64_t sink = 0;
    // LLAMA_MOE_FAULT_TIMING=1: each touch batch from posting to its last piece, with the drive's sectors read meanwhile
    // (/sys/dev/block/<dev>/stat of the model's partition): the read rate inside the bursts, which the 1 s means hide
    const bool timing = getenv("LLAMA_MOE_FAULT_TIMING") && atoi(getenv("LLAMA_MOE_FAULT_TIMING")) != 0;
    int stat_fd = -1;
    bool stat_tried = false;
    int64_t tb_t0 = 0;
    uint64_t tb_sec0 = 0;
    size_t tb_n = 0, tb_n_read = 0;
    double tb_us = 0.0, tb_us_read = 0.0, tb_bytes = 0.0, tb_bytes_by_rate[4] = {};   // rate < 1, 1-2, 2-3, >= 3 GB/s
    uint64_t sectors_read() const {
#ifndef __linux__
        return 0;
#else
        char buf[256];
        const ssize_t r = stat_fd >= 0 ? pread(stat_fd, buf, sizeof(buf) - 1, 0) : -1;
        if (r <= 0) {
            return 0;
        }
        buf[r] = 0;
        unsigned long long v[3] = {};
        sscanf(buf, "%llu %llu %llu", &v[0], &v[1], &v[2]);
        return v[2];
#endif
    }
    std::atomic<size_t> fin{0};
    void batch_begin() {
        fin.store(0, std::memory_order_relaxed);
        if (!timing) {
            return;
        }
#ifdef __linux__
        if (stat_fd < 0 && !stat_tried) {
            // the device of the first expert's mapping, from /proc/self/maps (touch mode holds no file descriptors)
            stat_tried = true;
            const uintptr_t x = [&] { for (const auto & pl : parts) { if (!pl.empty()) return (uintptr_t) pl[0].map; } return (uintptr_t) 0; }();
            std::ifstream f("/proc/self/maps");
            for (std::string line; x != 0 && std::getline(f, line); ) {
                unsigned long a, b, off;
                unsigned int ma, mi;
                char perms[8];
                if (sscanf(line.c_str(), "%lx-%lx %7s %lx %x:%x", &a, &b, perms, &off, &ma, &mi) == 6 && x >= a && x < b) {
                    char path[96];
                    snprintf(path, sizeof(path), "/sys/dev/block/%u:%u/stat", ma, mi);
                    stat_fd = open(path, O_RDONLY);
                    LLAMA_LOG_INFO("llama_moe_cache: fault batch timing: %s (fd %d), %llu sectors read so far\n", path, stat_fd, (unsigned long long) sectors_read());
                    break;
                }
            }
        }
#endif
        tb_t0   = ggml_time_us();
        tb_sec0 = sectors_read();
    }
    // by the reader that finishes the batch's last piece
    void batch_end() {
        const double dt = (double) (ggml_time_us() - tb_t0);
        const double b  = 512.0*(double) (sectors_read() - tb_sec0);
        tb_n++;
        tb_us += dt;
        if (b > 0) {
            tb_n_read++;
            tb_us_read += dt;
            tb_bytes   += b;
            const double gbs = b/std::max(dt, 1.0)/1e3;
            tb_bytes_by_rate[gbs < 1 ? 0 : gbs < 2 ? 1 : gbs < 3 ? 2 : 3] += b;
        }
    }

    static constexpr size_t align = 4096;
    int32_t n_expert = 0, n_parts = 0, n_slots = 0;
    size_t  slot_size = 0, chunk_size = 1 << 20;  // slot_size: the largest pool's
    // pools: the layers whose experts need the same slot size share one, with its own LRU list and a share of the memory
    // in proportion to its experts' bytes (mixed-quant models: DS4.1 8.7-11.5 MiB per expert, one size would waste ~20%)
    std::vector<int32_t> pool_of;                 // [il] -1: not in the arena
    std::vector<size_t>  pool_size;               // [pool] slot size
    std::vector<double>  pool_weight;             // [pool] bytes of all its experts
    std::vector<int32_t> phead, ptail;            // [pool] its LRU list, least recently used at head
    std::vector<int32_t> pool_of_slot;            // [slot]
    std::vector<size_t>  slot_addr;               // [slot] offset from base
    std::vector<std::vector<size_t>> part_off;    // [il][ip] offset of a part's buffer in the layer's slots
    uint8_t * base = nullptr;
    std::vector<std::vector<part>> parts;         // [il][ip], empty: the layer is not in the arena
    std::vector<std::vector<int32_t>> slot_of;    // [il][e], -1: not in the arena
    std::vector<int64_t>  key_of;                 // [slot] il*n_expert + e, -1: free
    std::vector<uint64_t> stamp;                  // [slot] the call that last used it
    std::vector<int32_t>  prv, nxt;               // [slot] its pool's LRU list
    std::vector<const uint8_t *> ptr;             // [slot*n_parts + ip] the expert's data in the slot
    std::vector<uint8_t>  bad;                    // [slot] a read of the current call failed
    uint64_t call = 0;
    std::vector<int> fds;
    std::unordered_map<int, uint64_t> fsize;      // O_DIRECT fd -> file size (a read may end at the end of the file)
    bool cold_now = false;                        // ensure_cold(): a prompt's reads, see there
    // store mode, decode: the reads are posted and ensure returns; the uploads and the CPU job wait for each expert's part
    // (get_ready) instead of the whole layer (LLAMA_MOE_ARENA_ASYNC=0: wait in ensure as before)
    bool async_reads = false;
    std::unique_ptr<std::atomic<int32_t>[]> pend;  // [slot*n_parts + ip] chunks still being read
    std::vector<int32_t> filled_prev;             // the last async call's new slots (failed reads cleared at the next call)
    mutable std::atomic<int64_t> t_ready_us{0};
    mutable std::atomic<size_t>  n_ready_waits{0};
    FILE * trace = getenv("LLAMA_MOE_ARENA_TRACE") ? fopen(getenv("LLAMA_MOE_ARENA_TRACE"), "wb") : nullptr;
    size_t n_cold_calls = 0, n_cold_reads = 0, n_cold_hits = 0;
    static constexpr size_t pad_after = 512;      // valid bytes read past each expert (the prompt copies' MMQ padding)

    // the reader pool: the caller posts the chunks of a call, the readers and the caller take them, the caller waits for all
    std::vector<chunk> jobs;
    std::atomic<size_t> next{0}, done{0};
    std::atomic<uint64_t> gen{0};
    std::atomic<bool> exit{false};
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::thread> readers;

    size_t n_calls = 0, n_hits = 0, n_reads = 0, n_full = 0, n_fail = 0;
    // LLAMA_MOE_FAULT_PRED: the next layer's predicted experts touched with this layer's, and how many it then missed
    int32_t pred_il = -1;
    std::vector<int32_t> pred_last;
    size_t n_pred = 0, n_pred_used = 0;
    double bytes = 0.0, t_us = 0.0;

    ~moe_arena() {
        if (trace) {
            fclose(trace);
        }
        exit.store(true);
        {
            std::lock_guard<std::mutex> lock(mtx);
            gen++;
        }
        cv.notify_all();
        for (auto & t : readers) {
            t.join();
        }
#ifdef __linux__
        for (int fd : fds) {
            close(fd);
        }
#endif
    }

    // the file and offset of each mapped expert tensor, from /proc/self/maps; false: some tensor is not file-mapped
    bool init(const std::vector<std::vector<ggml_tensor *>> & experts, int32_t n_exp) {
#ifdef __linux__
        if (touch_only) {
            // no file needed: only the mapping's addresses
            n_expert = n_exp;
            parts.assign(experts.size(), {});
            slot_of.assign(experts.size(), {});
            for (size_t il = 0; il < experts.size(); ++il) {
                for (const ggml_tensor * w : experts[il]) {
                    parts[il].push_back({ -1, 0, w->nb[2], (const uint8_t *) w->data });
                }
                if (!parts[il].empty()) {
                    n_parts = (int32_t) parts[il].size();
                    slot_of[il].assign(n_expert, -1);
                }
            }
            return n_parts > 0;
        }
        struct mapping { uintptr_t a, b; uint64_t off; std::string path; };
        std::vector<mapping> maps;
        std::ifstream f("/proc/self/maps");
        for (std::string line; std::getline(f, line); ) {
            unsigned long a, b, off;
            char perms[8], dev[16];
            unsigned long inode;
            int n = 0;
            if (sscanf(line.c_str(), "%lx-%lx %7s %lx %15s %lu %n", &a, &b, perms, &off, dev, &inode, &n) < 6 || inode == 0 || n <= 0) {
                continue;
            }
            const std::string path = line.substr(n);
            if (!path.empty() && path[0] == '/') {
                maps.push_back({ a, b, off, path });
            }
        }
        std::map<std::string, int> fd_of;
        n_expert = n_exp;
        parts.assign(experts.size(), {});
        slot_of.assign(experts.size(), {});
        std::vector<size_t> max_nb2;
        for (size_t il = 0; il < experts.size(); ++il) {
            if (experts[il].empty()) {
                continue;
            }
            std::vector<part> ps;
            for (const ggml_tensor * w : experts[il]) {
                const uintptr_t p = (uintptr_t) w->data;
                const mapping * m = nullptr;
                for (const mapping & mm : maps) {
                    if (p >= mm.a && p + ggml_nbytes(w) <= mm.b) {
                        m = &mm;
                        break;
                    }
                }
                if (m == nullptr) {
                    LLAMA_LOG_WARN("llama_moe_cache: arena: %s is not mapped from a file, no arena\n", w->name);
                    return false;
                }
                auto it = fd_of.find(m->path);
                if (it == fd_of.end()) {
                    const int fd = open(m->path.c_str(), O_RDONLY | O_DIRECT);
                    if (fd < 0) {
                        LLAMA_LOG_WARN("llama_moe_cache: arena: cannot open %s with O_DIRECT, no arena\n", m->path.c_str());
                        return false;
                    }
                    fds.push_back(fd);
                    it = fd_of.emplace(m->path, fd).first;
                    struct stat st_f;
                    fsize[fd] = fstat(fd, &st_f) == 0 ? (uint64_t) st_f.st_size : UINT64_MAX;
                    if (how == 3) {
                        const int fb = open(m->path.c_str(), O_RDONLY);
                        if (fb >= 0) {
                            fds.push_back(fb);
                            fd_buf[fd] = fb;
                        }
                    }
                }
                ps.push_back({ it->second, m->off + (p - m->a), w->nb[2], (const uint8_t *) w->data });
            }
            if (n_parts == 0) {
                n_parts = (int32_t) ps.size();
                max_nb2.assign(n_parts, 0);
            }
            GGML_ASSERT((int32_t) ps.size() == n_parts);
            for (int32_t ip = 0; ip < n_parts; ++ip) {
                max_nb2[ip] = std::max(max_nb2[ip], ps[ip].nb2);
            }
            parts[il] = std::move(ps);
            slot_of[il].assign(n_expert, -1);
        }
        // a part's buffer holds the aligned range around the expert: up to one block before and one after
        GGML_UNUSED(max_nb2);
        slot_size = 0;
        pool_of.assign(parts.size(), -1);
        part_off.assign(parts.size(), {});
        std::map<size_t, int32_t> pool_by_size;
        for (size_t il = 0; il < parts.size(); ++il) {
            if (parts[il].empty()) {
                continue;
            }
            size_t sz = 0;
            for (const part & p : parts[il]) {
                part_off[il].push_back(sz);
                sz += GGML_PAD(p.nb2, align) + 2*align;
            }
            auto it = pool_by_size.find(sz);
            if (it == pool_by_size.end()) {
                it = pool_by_size.emplace(sz, (int32_t) pool_size.size()).first;
                pool_size.push_back(sz);
                pool_weight.push_back(0.0);
            }
            pool_of[il] = it->second;
            pool_weight[it->second] += (double) n_expert*sz;
            slot_size = std::max(slot_size, sz);
        }
        return n_parts > 0;
#else
        GGML_UNUSED(experts); GGML_UNUSED(n_exp);
        return false;
#endif
    }

    // all the experts' bytes in slots
    size_t all_bytes() const {
        double w = 0.0;
        for (double x : pool_weight) {
            w += x;
        }
        return (size_t) w;
    }

    void start(uint8_t * mem, size_t size, int n_threads) {
        base    = mem;
        const double w_all = std::max(1.0, (double) all_bytes());
        n_slots = 0;
        size_t off = 0;
        phead.assign(pool_size.size(), -1);
        ptail.assign(pool_size.size(), -1);
        for (size_t ip = 0; ip < pool_size.size(); ++ip) {
            const int32_t n = (int32_t) std::min<double>(pool_weight[ip]/pool_size[ip], (double) size*pool_weight[ip]/w_all/pool_size[ip]);
            for (int32_t j = 0; j < n; ++j) {
                const int32_t s = n_slots++;
                slot_addr.push_back(off);
                pool_of_slot.push_back((int32_t) ip);
                prv.push_back(j > 0 ? s - 1 : -1);
                nxt.push_back(j + 1 < n ? s + 1 : -1);
                off += pool_size[ip];
            }
            if (n > 0) {
                phead[ip] = n_slots - n;
                ptail[ip] = n_slots - 1;
            }
        }
        GGML_ASSERT(off <= size);
        key_of.assign(n_slots, -1);
        stamp.assign(n_slots, 0);
        ptr.assign((size_t) n_slots*n_parts, nullptr);
        bad.assign(n_slots, 0);
        pend.reset(new std::atomic<int32_t>[(size_t) n_slots*n_parts]);
        for (size_t i = 0; i < (size_t) n_slots*n_parts; ++i) {
            pend[i].store(0, std::memory_order_relaxed);
        }
        for (int i = 0; i < n_threads; ++i) {
            readers.emplace_back([this] {
                uint64_t seen = 0;
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [&] { return gen.load() != seen; });
                        seen = gen.load();
                    }
                    if (exit.load()) {
                        return;
                    }
                    work();
                }
            });
        }
    }

    void work() {
#ifdef __linux__
        for (size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < jobs.size(); ) {
            const chunk & c = jobs[i];
            if (c.src != nullptr) {
                // LLAMA_MOE_FAULT_HOW: 2 (default) madvise(MADV_WILLNEED) then touch: the kernel reads the range in large
                // requests (pcbox 32 GB: +13%, 2.5k vs 8k major faults/s); 0 touch one byte per page only (the model mapping is
                // MADV_RANDOM, so each fault reads little), 1 madvise(MADV_POPULATE_READ) (-33%), 3 a buffered pread of the
                // range into a scratch buffer (no gain)
                if (how == 1) {
                    const uintptr_t a = (uintptr_t) c.src & ~(uintptr_t) 4095;
                    madvise((void *) a, (uintptr_t) c.src + c.n - a, 23 /* MADV_POPULATE_READ */);
                } else if (how == 3 && c.fd >= 0) {
                    thread_local std::vector<uint8_t> scratch;
                    scratch.resize(c.n);
                    const auto it = fd_buf.find(c.fd);
                    if (it != fd_buf.end()) {
                        size_t got = 0;
                        while (got < c.n) {
                            const ssize_t r = pread(it->second, scratch.data() + got, c.n - got, (off_t) (c.off + got));
                            if (r <= 0) {
                                break;
                            }
                            got += (size_t) r;
                        }
                    }
                } else {
                    if (how == 2) {
                        const uintptr_t a = (uintptr_t) c.src & ~(uintptr_t) 4095;
                        madvise((void *) a, (uintptr_t) c.src + c.n - a, MADV_WILLNEED);
                    }
                    uint64_t acc = 0;
                    for (size_t o = 0; o < c.n; o += align) {
                        acc += c.src[o];
                    }
                    sink += acc;
                }
                // the batch's end is taken before the last piece counts as done (the caller then reuses the batch fields)
                if (timing && fin.fetch_add(1, std::memory_order_acq_rel) + 1 == jobs.size()) {
                    batch_end();
                }
                done.fetch_add(1, std::memory_order_acq_rel);
                continue;
            }
            size_t got = 0;
            while (got < c.n) {
                const ssize_t r = pread(c.fd, c.dst + got, c.n - got, (off_t) (c.off + got));
                if (r <= 0) {
                    break;
                }
                got += (size_t) r;
            }
            // a short read at the end of the file is fine: the rest (padding past the last expert) is zeroed
            if (got < c.n) {
                const auto fz = fsize.find(c.fd);
                if (fz != fsize.end() && c.off + got >= fz->second) {
                    memset(c.dst + got, 0, c.n - got);
                } else {
                    bad[c.slot] = 1;
                }
            }
            if (c.pi >= 0) {
                pend[c.pi].fetch_sub(1, std::memory_order_acq_rel);
            }
            done.fetch_add(1, std::memory_order_acq_rel);
        }
#endif
    }

    // slot j becomes the most recently used
    void touch(int32_t j) {
        int32_t & head = phead[pool_of_slot[j]];
        int32_t & tail = ptail[pool_of_slot[j]];
        if (j == tail) {
            return;
        }
        if (prv[j] >= 0) nxt[prv[j]] = nxt[j]; else head = nxt[j];
        prv[nxt[j]] = prv[j];
        prv[j] = tail;
        nxt[j] = -1;
        nxt[tail] = j;
        tail = j;
    }

    // slot j becomes the least recently used (the first victim once this and the next call are over)
    void to_head(int32_t j) {
        int32_t & head = phead[pool_of_slot[j]];
        int32_t & tail = ptail[pool_of_slot[j]];
        if (j == head) {
            return;
        }
        if (nxt[j] >= 0) prv[nxt[j]] = prv[j]; else tail = prv[j];
        nxt[prv[j]] = nxt[j];
        nxt[j] = head;
        prv[j] = -1;
        prv[head] = j;
        head = j;
    }

    // a prompt's experts: read like a decode call's, but the new ones go to the cold end and the ones already here do not
    // move, so a long prompt passes through the coldest slots instead of pushing the decode's hot set out
    void ensure_cold(int32_t il, const int32_t * es, size_t n) {
        if (base == nullptr || touch_only || slot_of[il].empty() || n == 0) {
            return;
        }
        cold_now = true;
        n_cold_calls++;
        const size_t r0 = n_reads, h0 = n_hits;
        ensure(il, es, n);
        n_cold_reads += n_reads - r0;
        n_cold_hits  += n_hits - h0;
        cold_now = false;
    }

    // experts just uploaded to the VRAM cache: their slots to the cold end (exclusive caching: the GPU holds them now; the
    // stamps keep them for this and the next call, so the copies finish first). LLAMA_MOE_ARENA_DEMOTE=1 (off by default:
    // pcbox 32 GB Qwen 20.7 / 17.4 vs 22.5 / 23.6 without, reads +15%: the VRAM cache turns over, its experts come back)
    size_t n_demoted = 0;
    void demote(int32_t il, const int32_t * es, size_t n) {
        if (base == nullptr || touch_only || slot_of[il].empty()) {
            return;
        }
        for (size_t i = 0; i < n; ++i) {
            const int32_t s = slot_of[il][es[i]];
            if (s >= 0) {
                to_head(s);
                n_demoted++;
            }
        }
    }

    // exclusive mode (LLAMA_MOE_ARENA_EXCL, store mode): each expert is in VRAM or in the arena, not both, so their sum is the
    // RAM's capacity (bench/tier_sim.py: 3x fewer SSD reads than the inclusive arena on the 32 GB trace). release: an expert
    // just uploaded to VRAM leaves the arena (its slot unmapped, to the cold end; the stamp keeps it for this and the next
    // call, so the upload reads it first). claim_wb: a slot for an expert the VRAM cache evicts, which the caller copies back
    // into it from the device on the compute stream (done by the next layer call's sync, before anyone reads it); dst gets
    // each part's start. False when the expert is still here (moved to the hot end) or no slot is free of this call
    size_t n_released = 0, n_wb = 0, n_wb_here = 0, n_wb_full = 0;
    void release(int32_t il, int32_t e) {
        if (base == nullptr || touch_only || slot_of[il].empty()) {
            return;
        }
        int32_t & s = slot_of[il][e];
        if (s < 0) {
            return;
        }
        key_of[s] = -1;
        to_head(s);
        s = -1;
        n_released++;
    }
    // LLAMA_MOE_ARENA_EXCL_VERIFY=1 (test): a copied back expert compared with the model file (through the mapping)
    size_t n_verified = 0, n_verify_bad = 0;
    void verify(int32_t il, int32_t e) {
        const int32_t s = slot_of[il][e];
        if (s < 0) {
            return;
        }
        for (int32_t ip = 0; ip < n_parts; ++ip) {
            const part & p = parts[il][ip];
            if (memcmp(ptr[(size_t) s*n_parts + ip], p.map + (size_t) e*p.nb2, p.nb2) != 0) {
                n_verify_bad++;
                LLAMA_LOG_WARN("llama_moe_cache: arena verify: layer %d expert %d part %d differs from the file\n", il, e, ip);
            }
        }
        n_verified++;
    }
    bool claim_wb(int32_t il, int32_t e, uint8_t ** dst) {
        if (base == nullptr || touch_only || il < 0 || il >= (int32_t) slot_of.size() || slot_of[il].empty()) {
            return false;
        }
        int32_t & s = slot_of[il][e];
        if (s >= 0) {
            touch(s);
            n_wb_here++;
            return false;
        }
        int32_t v = phead[pool_of[il]];
        while (v >= 0 && stamp[v] + 1 >= call && stamp[v] != 0) {
            v = nxt[v];
        }
        if (v < 0) {
            n_wb_full++;
            return false;
        }
        if (key_of[v] >= 0) {
            slot_of[key_of[v]/n_expert][key_of[v] % n_expert] = -1;
        }
        key_of[v] = (int64_t) il*n_expert + e;
        stamp[v]  = call;
        bad[v]    = 0;
        touch(v);
        s = v;
        for (int32_t ip = 0; ip < n_parts; ++ip) {
            const part & p = parts[il][ip];
            const uint64_t a = p.off + (uint64_t) e*p.nb2;
            const uint64_t a0 = a & ~(uint64_t) (align - 1);
            uint8_t * d = base + slot_addr[v] + part_off[il][ip] + (a - a0);
            ptr[(size_t) v*n_parts + ip] = d;
            pend[(size_t) v*n_parts + ip].store(0, std::memory_order_relaxed);
            dst[ip] = d;
        }
        n_wb++;
        return true;
    }

    // store mode, at start: the hottest experts of a ranking (hottest first) read in, up to 90% of the slots, one call per
    // layer (all of a layer's reads in flight at once); the hottest end up most recently used
    size_t preload(const std::vector<std::pair<int32_t, int32_t>> & list) {
        if (base == nullptr || touch_only) {
            return 0;
        }
        const size_t k = std::min(list.size(), (size_t) n_slots*9/10);
        std::vector<std::vector<int32_t>> by_layer(parts.size());
        std::vector<int32_t> order;                 // layers by their coldest entry first
        for (size_t i = k; i-- > 0; ) {
            const auto & le = list[i];
            if (le.first < 0 || le.first >= (int32_t) parts.size() || parts[le.first].empty()) {
                continue;
            }
            if (by_layer[le.first].empty()) {
                order.push_back(le.first);
            }
            by_layer[le.first].push_back(le.second);
        }
        const size_t r0 = n_reads;
        for (int32_t il : order) {
            ensure(il, by_layer[il].data(), by_layer[il].size());
        }
        return n_reads - r0;
    }

    // the model file's pages of the experts out of the page cache: the arena is their only RAM copy (store mode)
    size_t drop_page_cache() const {
        size_t n = 0;
#ifdef __linux__
        const uintptr_t page = (uintptr_t) sysconf(_SC_PAGE_SIZE);
        for (size_t il = 0; il < parts.size(); ++il) {
            for (const part & p : parts[il]) {
                const size_t len = (size_t) n_expert*p.nb2;
                const uintptr_t a = (uintptr_t) p.map & ~(page - 1);
                const uintptr_t z = ((uintptr_t) p.map + len + page - 1) & ~(page - 1);
                madvise((void *) a, z - a, MADV_DONTNEED);
                if (p.fd >= 0) {
                    posix_fadvise(p.fd, (off_t) p.off, (off_t) len, POSIX_FADV_DONTNEED);
                }
                n += len;
            }
        }
#endif
        return n;
    }

    // the expert's tensor ip in the arena, nullptr: read it from the mapping
    const uint8_t * get(int32_t il, int32_t ip, int32_t e) const {
        if (base == nullptr || slot_of[il].empty()) {
            return nullptr;
        }
        const int32_t s = slot_of[il][e];
        return s >= 0 ? ptr[(size_t) s*n_parts + ip] : nullptr;
    }

    // get, after waiting for the part's reads (async decode reads); nullptr also when the read failed
    const uint8_t * get_ready(int32_t il, int32_t ip, int32_t e) const {
        const uint8_t * p = get(il, ip, e);
        if (p == nullptr) {
            return nullptr;
        }
        const int32_t s = slot_of[il][e];
        const std::atomic<int32_t> & c = pend[(size_t) s*n_parts + ip];
        if (c.load(std::memory_order_acquire) > 0) {
            const int64_t t0 = ggml_time_us();
            while (c.load(std::memory_order_acquire) > 0) {
            }
            t_ready_us.fetch_add(ggml_time_us() - t0, std::memory_order_relaxed);
            n_ready_waits.fetch_add(1, std::memory_order_relaxed);
        }
        return bad[s] ? nullptr : p;
    }

    // failed reads of the last async call: their slots emptied (after all its reads are done)
    void clear_failed(std::vector<int32_t> & fl) {
        for (int32_t v : fl) {
            if (bad[v] && key_of[v] >= 0) {
                n_fail++;
                slot_of[key_of[v]/n_expert][key_of[v] % n_expert] = -1;
                key_of[v] = -1;
                stamp[v]  = 0;
            }
        }
        fl.clear();
    }

    // a decode call: the experts of layer il the uploads and the CPU will read; the missing ones are read before it returns
    void ensure(int32_t il, const int32_t * es, size_t n, int32_t il2 = -1, const int32_t * es2 = nullptr, size_t n2 = 0) {
        if (pred_il == il) {
            for (size_t i = 0; i < n; ++i) {
                n_pred_used += std::find(pred_last.begin(), pred_last.end(), es[i]) != pred_last.end();
            }
        }
        pred_il = -1;
        if ((base == nullptr && !touch_only) || slot_of[il].empty() || (n == 0 && n2 == 0)) {
            return;
        }
        const int64_t t0 = ggml_time_us();
        if (no_wait && bulk && done.load(std::memory_order_acquire) < jobs.size()) {
            // a shadow pass's read-ahead (touch_bulk) is still running: it holds this layer's misses too (most of them), and
            // waiting for it here would stall the real pass; the readers fault on the rest
            n_bulk_skip++;
            return;
        }
        bulk = false;
        if (!touch_only && !filled_prev.empty()) {
            // the previous async call's reads (its experts were mostly read by now: the uploads and the CPU job waited)
            while (done.load(std::memory_order_acquire) < jobs.size()) {
            }
            clear_failed(filled_prev);
        }
        if (no_wait) {
            // the readers may still touch the previous layer's
            while (done.load(std::memory_order_acquire) < jobs.size()) {
            }
        }
        call++;
        n_calls++;
        if (trace) {
            // LLAMA_MOE_ARENA_TRACE=<path>: each call's experts, int32 (il << 16 | e) + 1, negated for a prompt call, 0 ends a call
            for (size_t i = 0; i < n; ++i) {
                const int32_t v = ((il << 16 | es[i]) + 1) * (cold_now ? -1 : 1);
                fwrite(&v, 4, 1, trace);
            }
            const int32_t z = 0;
            fwrite(&z, 4, 1, trace);
        }
        jobs.clear();
        std::vector<int32_t> filled;
        auto touch_expert = [&](int32_t l, int32_t e) {
            for (int32_t ip = 0; ip < n_parts; ++ip) {
                const part & p = parts[l][ip];
                const uint8_t * src = p.map + (size_t) e*p.nb2;
                for (size_t o = 0; o < p.nb2; o += chunk_size) {
                    jobs.push_back({ p.fd, p.off + (uint64_t) e*p.nb2 + o, std::min(chunk_size, p.nb2 - o), nullptr, -1, src + o });
                }
            }
        };
        for (size_t i = 0; i < n && touch_only; ++i) {
            touch_expert(il, es[i]);
            n_reads++;
        }
        if (touch_only && n2 > 0 && il2 >= 0 && !parts[il2].empty()) {
            pred_il = il2;
            pred_last.assign(es2, es2 + n2);
            for (size_t i = 0; i < n2; ++i) {
                touch_expert(il2, es2[i]);
            }
            n_pred += n2;
        }
        for (size_t i = 0; i < n && !touch_only; ++i) {
            const int32_t e = es[i];
            int32_t & s = slot_of[il][e];
            if (s >= 0) {
                if (stamp[s] != call) {
                    n_hits++;
                }
                stamp[s] = call;
                if (!cold_now) {
                    touch(s);
                }
                continue;
            }
            // the least recently used slot that this and the previous call do not use (only these are at the tail)
            int32_t v = phead[pool_of[il]];
            while (v >= 0 && stamp[v] + 1 >= call && stamp[v] != 0) {
                v = nxt[v];
            }
            if (v < 0) {
                n_full++;
                continue;
            }
            if (key_of[v] >= 0) {
                slot_of[key_of[v]/n_expert][key_of[v] % n_expert] = -1;
            }
            key_of[v] = (int64_t) il*n_expert + e;
            stamp[v]  = call;
            bad[v]    = 0;
            if (cold_now) {
                to_head(v);
            } else {
                touch(v);
            }
            s = v;
            filled.push_back(v);
            for (int32_t ip = 0; ip < n_parts; ++ip) {
                const part & p = parts[il][ip];
                const uint64_t a = p.off + (uint64_t) e*p.nb2;
                const uint64_t a0 = a & ~(uint64_t) (align - 1);
                const uint64_t a1 = GGML_PAD(a + p.nb2 + pad_after, align);
                uint8_t * dst = base + slot_addr[v] + part_off[il][ip];
                ptr[(size_t) v*n_parts + ip] = dst + (a - a0);
                const int32_t pi = v*n_parts + ip;
                pend[pi].store((int32_t) ((a1 - a0 + chunk_size - 1)/chunk_size), std::memory_order_relaxed);
                for (uint64_t o = a0; o < a1; o += chunk_size) {
                    jobs.push_back({ p.fd, o, (size_t) std::min<uint64_t>(chunk_size, a1 - o), dst + (o - a0), v, nullptr, pi });
                }
                bytes += (double) (a1 - a0);
            }
            n_reads++;
        }
        if (!jobs.empty()) {
            if (touch_only) {
                batch_begin();
            }
            next.store(0, std::memory_order_relaxed);
            done.store(0, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(mtx);
                gen++;
            }
            cv.notify_all();
            if (no_wait) {
                t_us += ggml_time_us() - t0;
                return;
            }
            if (async_reads && !cold_now) {
                filled_prev = std::move(filled);
                t_us += ggml_time_us() - t0;
                return;
            }
            work();
            while (done.load(std::memory_order_acquire) < jobs.size()) {
            }
            for (int32_t v : filled) {
                if (bad[v]) {
                    n_fail++;
                    slot_of[key_of[v]/n_expert][key_of[v] % n_expert] = -1;
                    key_of[v] = -1;
                    stamp[v]  = 0;
                }
            }
        }
        t_us += ggml_time_us() - t0;
    }

    // touch-only, no-wait: post the reads of several layers' experts as one batch (a GPU-only shadow pass, in layer order)
    void touch_bulk(const std::vector<std::pair<int32_t, std::vector<int32_t>>> & lst) {
        if (!touch_only || !no_wait || lst.empty()) {
            return;
        }
        while (done.load(std::memory_order_acquire) < jobs.size()) {
        }
        jobs.clear();
        for (const auto & le : lst) {
            if (le.first < 0 || parts[le.first].empty()) {
                continue;
            }
            for (int32_t e : le.second) {
                for (int32_t ip = 0; ip < n_parts; ++ip) {
                    const part & p = parts[le.first][ip];
                    const uint8_t * src = p.map + (size_t) e*p.nb2;
                    for (size_t o = 0; o < p.nb2; o += chunk_size) {
                        jobs.push_back({ p.fd, p.off + (uint64_t) e*p.nb2 + o, std::min(chunk_size, p.nb2 - o), nullptr, -1, src + o });
                    }
                }
                n_reads++;
            }
        }
        if (jobs.empty()) {
            return;
        }
        bulk = true;
        n_bulk++;
        batch_begin();
        next.store(0, std::memory_order_relaxed);
        done.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mtx);
            gen++;
        }
        cv.notify_all();
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

// the pinned RAM an auto-sized arena may take: MemAvailable, less the CMA region and a reserve (the server's prompt cache, host
// compute buffers, the OS). Pinned (long-term) pages cannot live in CMA, but MemAvailable counts it: kernel 7.0 shows ~7% of RAM
// there with CmaTotal 0 (pcbox1 128 GB: 9.0 GiB; a 120 GiB arena was OOM-killed with 9 GiB "free"). CmaFree changes as pinning
// migrates pages out, so the region itself (the per-zone "cma" count of /proc/zoneinfo) is taken when larger.
static size_t arena_auto_room(size_t & cma, size_t & rsv) {
    cma = rsv = 0;
#ifndef __linux__
    return 0;
#else
    size_t avail = 0, cma_free = 0;
    std::ifstream mi("/proc/meminfo");
    for (std::string k; mi >> k; ) {
        size_t v = 0;
        std::string unit;
        mi >> v >> unit;
        if (k == "MemAvailable:") {
            avail = v << 10;
        } else if (k == "CmaFree:") {
            cma_free = v << 10;
        }
    }
    size_t cma_region = 0;
    std::ifstream zi("/proc/zoneinfo");
    for (std::string k; zi >> k; ) {
        if (k == "cma") {
            size_t n = 0;
            if (zi >> n) {
                cma_region += n * (size_t) sysconf(_SC_PAGESIZE);
            } else {
                zi.clear();
            }
        }
    }
    cma = std::max(cma_free, cma_region);
    avail = avail > cma ? avail - cma : 0;
    rsv = getenv("LLAMA_MOE_ARENA_RESERVE_MIB") ? (size_t) atoll(getenv("LLAMA_MOE_ARENA_RESERVE_MIB")) << 20
                                                : ((size_t) 2 << 30) + llama_moe_ram_limit()/64;
    return avail > rsv ? avail - rsv : 0;
#endif
}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
    };

    // slot lending (llama_moe_cache::flex_release): the banks of every group live in one VMM buffer (ggml-cuda), each bank at a
    // chunk-aligned offset; the largest group lends the physical memory under its last slots [lend_keep, n_slots) to the compute
    // buffers of prompt batches (the zero slot after them stays mapped). The addresses never change: graphs stay valid
    using vmm_remap_t = size_t (*)(ggml_backend_buffer_t, size_t, size_t, bool);
    vmm_remap_t vmm_remap = nullptr;
    int32_t lend_ig = -1, lend_keep = 0;
    bool    lend_out = false;
    size_t  flex_bytes = 0, lend_bytes = 0;
    ggml_context_ptr bank_ctx;
    ggml_backend_buffer_ptr bank_buf;
    std::vector<size_t> lend_off;            // [ip] the lending group's bank offsets in bank_buf
    size_t  n_flex_release = 0, n_flex_back = 0, n_flex_evicted = 0;
    double  flex_us = 0.0;

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
        ggml_tensor * smap_dev = nullptr;   // GPU-only shadow: I32 [1, n_expert] on the device, uncached experts -> the zero slot
        ggml_tensor * shadow_ids = nullptr; // GPU-only shadow: the selected experts of the current shadow graph (an output)
        bool up_wait = false;               // LLAMA_MOE_CPU_AFTER_UP: this job waits for the call's uploads
        ggml_tensor * fmap_dev = nullptr;   // GPU fetch: I32 [1, n_expert] on the device, a slot or -(e+1) (read in place)
        int32_t fmap_row = -1;              // GPU fetch: its row of fmap_all (rows go group by group, not by layer)
        ggml_tensor * fetch_ids = nullptr;  // GPU fetch: the selected experts of the current decode graph (an output)

        // hybrid mode
        ggml_tensor * up = nullptr, * gate = nullptr, * down = nullptr, * gate_up = nullptr; // host experts
        ggml_tensor * cpu_out = nullptr;    // F32 [n_embd, max_batch] in host memory, added to the MoE output on the device
        ggml_tensor * x_dev = nullptr;      // expert input of the current graph [n_embd, n_tokens], nullptr: not hybrid
        ggml_tensor * w_dev = nullptr;      // router weights of the current graph [1, n_expert_used, n_tokens]
        llama_moe_cache::act_type act = llama_moe_cache::ACT_NONE;
        float limit = 0.0f;
        // expert biases (gpt-oss), F32 [rows, n_expert] in host memory, nullptr: none; bias_ok false: a bias the CPU cannot read
        const ggml_tensor * b_up = nullptr, * b_gate = nullptr, * b_down = nullptr, * b_gate_up = nullptr;
        bool bias_ok = true;
        std::vector<int32_t> map_dev;       // slot map as uploaded: the CPU's experts read the zero slot
        ggml_tensor * map_pin = nullptr;    // I32 [n_expert] in host memory, the copy of map_dev that is uploaded
        ggml_tensor * flag    = nullptr;    // I32 [4] in pinned memory, flag mode: see ggml_host_add
        // late pass (LLAMA_MOE_LATE): device slots and weights of the late experts [K*8], their pinned staging, the device flag the
        // copy stream bumps after the layer's late uploads (read by ggml_host_add's wait), its counter
        ggml_tensor * l_slots = nullptr, * l_w = nullptr, * l_slots_pin = nullptr, * l_w_pin = nullptr, * l_flag = nullptr;
        uint32_t l_seq = 0;
        bool late_graph = false;            // a graph built the late pass for this layer (late_barrier)
        ggml_tensor * db_out = nullptr;     // doorbell mode: the graph being built reads this layer's slots from this op (or nullptr)
        std::vector<float> x, w, out;       // host copies of the input and weights, the CPU's output
        std::vector<int32_t> cpu_experts;   // missed experts of the current ubatch that the CPU computes
        std::vector<int32_t> ids;           // selected experts of the current ubatch
        bool planned = false;               // the slot map callback ran in hybrid mode for the current ubatch
        int64_t t_start = 0;                // when the CPU job was started
        int64_t t_job   = 0;                // how long it ran (us)
        ggml_tensor * db = nullptr;         // doorbell mode: the control block (struct ggml_moe_db and its arrays) in pinned memory
        // prefill share: the expert input, router weights and selected experts of the current large-batch graph (p_x nullptr: none)
        ggml_tensor * p_x = nullptr, * p_w = nullptr;
        const ggml_tensor * p_ids = nullptr;
        ggml_tensor * pf_flag = nullptr;    // I32 [4] in pinned memory, the flag of the host_add of the prefill share
        bool pf_job = false;                // the CPU job of this layer is a prefill one: its output goes to pf_out
        ggml_tensor * la_pred = nullptr;    // lookahead: the predicted experts of the next layer [la_k, n_tokens] in the current graph
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches
    size_t n_cpu_experts = 0; // hybrid: missed experts computed by the CPU
    bool   flag_mode = false; // hybrid: the GPU reads the CPU's output from pinned memory after a flag (see hybrid_add)
    size_t n_flag_waits = 0, n_flag_timeouts = 0;
    double t_flag_wait_us = 0.0;
    size_t n_cpu_pairs   = 0; // hybrid: (token, expert) pairs computed by the CPU
    double t_cpu_us      = 0.0;
    // hybrid timers (host, us): sync for the ids, uploads, between the slot map and cpu_out copies, waiting for the CPU
    double t_sync_us = 0.0, t_upload_us = 0.0, t_gap_us = 0.0, t_wait_us = 0.0;
    // plan+uploads split (hybrid calls): plan, read-ahead posting, waiting for the CPU worker, the copies; the probe
    // (LLAMA_MOE_UPLOAD_PROBE=1, mincore, not counted in the times) counts the uploaded pages not in RAM when the copy starts
    double t_up_plan_us = 0.0, t_up_ensure_us = 0.0, t_up_wait_us = 0.0, t_up_copy_us = 0.0;
    size_t up_bytes = 0, up_pages = 0, up_pages_out = 0;
    const bool up_probe = getenv("LLAMA_MOE_UPLOAD_PROBE") && atoi(getenv("LLAMA_MOE_UPLOAD_PROBE")) != 0;
#ifdef __linux__
    const bool up_order = getenv("LLAMA_MOE_UPLOAD_ORDER") && atoi(getenv("LLAMA_MOE_UPLOAD_ORDER")) != 0;
#else
    const bool up_order = false;   // needs mincore
#endif
    size_t n_up_late = 0;
    std::vector<uint8_t> up_late;
#ifdef __linux__
    std::vector<unsigned char> mc_vec;
    // all pages of [p, p + n) in RAM
    bool pages_resident(const uint8_t * p, size_t n) {
        static const uintptr_t page = (uintptr_t) sysconf(_SC_PAGE_SIZE);
        const uintptr_t b = (uintptr_t) p & ~(page - 1);
        const size_t np = ((uintptr_t) p + n - b + page - 1)/page;
        mc_vec.resize(np);
        if (mincore((void *) b, np*page, mc_vec.data()) != 0) {
            return true;
        }
        for (unsigned char c : mc_vec) {
            if (!(c & 1)) {
                return false;
            }
        }
        return true;
    }
#endif
    int64_t t_planned = 0;
    size_t n_hybrid_calls = 0;

    // hybrid mode
    bool   hybrid = false;
    double hybrid_share = 0.0; // share of the misses uploaded
    int32_t tail_rank   = 0;   // LLAMA_MOE_TAIL_SKIP: skip missed experts that every token routes at this rank or lower (approximate)
    size_t  n_tail_skip = 0;   // (layer, expert) misses skipped
    std::vector<int32_t> tail_skipped;
    // LLAMA_MOE_CPU=auto: the share follows the measured balance, so that the CPU's experts end when the GPU reaches cpu_out:
    // every window of layer calls, the share goes up a step if the GPU mostly waited for the CPU, down if the CPU mostly idled
    bool   hybrid_auto = false;
    double ctl_wait_us = 0.0, ctl_idle_us = 0.0;
    int    ctl_calls = 0;
    int    ctl_window = 768; // layer calls (16 tokens of a 48-layer model)
    size_t n_share_changes = 0;
    // share rule (default in auto mode; LLAMA_MOE_SHARE_RULE=0 off, =<k> the factor): the share follows Strata's rule L/(L + k*R) (k 2.25 by default; Strata uses 1.85) instead of the waits:
    // L the host-to-device link (GB/s, probed at start), R the CPU's rate on missed experts (expert bytes / the job's wall
    // time per expert, measured). Uploads and the CPU read the same RAM, so sending more over the link slows the CPU;
    // the wait-based controller does not see that and overshoots (4 GiB, asrok: auto 0.25 120.8 tok/s, fixed 0.20 129.7)
    bool   share_rule = false;
    // k 2.25 (2026-11-09, Qwen3.8 IQ3_S W1 n=40): pcbox1 78.4/78.1 vs 76.3/76.1 at 1.85 (share 0.25 vs 0.30), cfl1 64.8/64.1 vs 64.2/64.2
    double rule_k = 2.25, link_gbs = 0.0, expert_bytes = 0.0;
    double rule_t_cpu = 0.0;
    size_t rule_n_cpu = 0;
    double rule_R_last = 0.0;
    // auto mode A/B gate: the CPU path has a fixed cost per layer (reading the expert input back, a second split for cpu_out),
    // so when few experts miss it is slower than uploading them all; every so often the gate runs a probe window with the CPU
    // path off and keeps whichever was faster, by the time from the first to the last cached layer per ubatch of each size
    bool    gate_on    = true;  // the decision: CPU path on or off
    std::atomic<bool> path_on{true}; // the CPU path is on in the graphs being built (differs from gate_on in a probe window)
    int     gate_phase = 0;     // 0 hold the decision, 1 probe window (the other state), 2 comparison window (the decision)
    bool    ubatch_on  = false; // the current ubatch's graph has the CPU path
    mutable std::atomic<bool> gate_changed{false};
    int32_t il_first = -1, il_last = -1; // first and last cached layer
    int64_t t_first = 0;                 // when the current ubatch reached the first cached layer
    int64_t t_first_last = 0;            // t_first of the ubatch just timed (share tuner)
    int64_t n_tok_cur = 0;
    struct span_acc { double us[9] = {}; int n[9] = {}; };
    span_acc acc_on, acc_off;
    int     gate_count = 0;              // ubatches in the current phase
    int     gate_hold  = 256;            // ubatches to stay before the next probe (doubles while the decision holds)
    static constexpr int gate_probe = 48; // ubatches per probe window and per comparison window
    size_t  n_gate_probes = 0, n_gate_off = 0, n_ubatch_on = 0, n_ubatch_off = 0;
    // share tuner (opt-in, LLAMA_MOE_SHARE_TUNE=1; no gain in W1 n=40: pcbox1 75.8/74.6 vs 75.8/74.7 off, cfl1 -1.3%): the rule's factor fits some boxes and not others
    // (pcbox1 14600K: rule 0.30, best fixed 0.20 by 2.6%; cfl1 i7-8700: rule 0.30 best). While the gate holds the CPU path
    // on, the tuner runs the share at rule + offset + {-0.1, 0, +0.1} in interleaved blocks of tune_block ubatches, compares
    // the mean ubatch span per size, and moves the offset 0.1 toward the fastest arm when it wins by tune_margin
    bool    share_tune  = false;
    double  share_off   = 0.0;           // offset on the rule's share, from the tuner
    double  tune_delta  = 0.0;           // the current arm's delta
    int     tune_arm    = 1, tune_count = 0, tune_cycles = 0;
    int     tune_hold   = 128;           // ubatches between trials (doubles while the offset holds)
    bool    tune_active = false;
    span_acc acc_arm[3];
    static constexpr int    tune_block  = 8;   // ubatches per arm block
    static constexpr int    tune_ncycle = 16;  // arm cycles per trial (3*8*16 = 384 ubatches)
    static constexpr double tune_margin = 0.01;
    size_t  n_tune_trials = 0, n_tune_moves = 0;
    // exclusive store mode keeps the CPU path in the graphs when the gate is off (only that path copies VRAM victims back to
    // the arena; the plain upload path lost them, pcbox1 32 GB gate starting off: 50-56 tok/s vs 145): off = every miss uploaded
    bool store_gate_path() const { return arena_store && arena_excl && hybrid_auto; }
    double cur_share() const {
        if (store_gate_path() && !path_on) {
            return 1.0;
        }
        if (!hybrid_auto && hybrid_share >= 1.0) {
            return 1.0; // LLAMA_MOE_CPU=1 or the arena fill: every miss uploaded, the CPU computes none
        }
        return std::clamp(hybrid_share + share_off + tune_delta, 0.05, 0.95);
    }
    // default: one less than the physical cores this process may run on (LLAMA_MOE_CPU_THREADS overrides); was hardware threads
    // / 2 - 1, which halved CPUs without SMT (Core Ultra, E-cores) and ignored a container's cpuset
    int     n_threads = std::max(1, moe_physical_cores() - 1);
    int64_t n_embd = 0;
    ggml_backend_ptr cpu_backend;
    std::unordered_map<const ggml_tensor *, int32_t> layer_of_out; // cpu_out -> layer
    std::vector<int32_t> pair_tok;  // CPU pairs of a layer: token, router weight index
    std::vector<int32_t> pair_wi;

    // the CPU's experts are computed by a worker thread, started before the uploads (which block the host when they read
    // pageable memory) and joined by the copy of cpu_out; it spins between jobs to wake up quickly
    // CPU graphs by number of pairs; all cached layers have experts of the same shape and type, so a graph is reused for any
    // layer by pointing its mul_mat_id nodes at that layer's experts
    struct cpu_graph {
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        ggml_cgraph * gf = nullptr;
        ggml_tensor * x = nullptr, * sel = nullptr, * y = nullptr;
        ggml_tensor * mm_gate_up = nullptr, * mm_up = nullptr, * mm_gate = nullptr, * mm_down = nullptr;
        llama_moe_cache::act_type act = llama_moe_cache::ACT_NONE;
        float limit = 0.0f;
    };
    std::unordered_map<int64_t, cpu_graph> cpu_graphs;

    // CPU expert kernel (default; LLAMA_MOE_CPU_KERNEL=ggml uses the ggml graph above): the worker and n_threads - 1 helpers
    // split each job by (expert, row block), with the weights of an expert read once for all of its tokens; the helpers spin
    // between jobs and sleep after a long idle time
    bool use_kernel = true;
    std::vector<std::thread> helpers;
    std::atomic<uint64_t> pool_epoch{0};
    std::atomic<int>      pool_done{0};
    std::atomic<int>      pool_next{0};
    std::atomic<bool>     pool_exit{false};
    std::function<void(int)> pool_fn;
    // LLAMA_MOE_CPU_GUEST=1: the main thread helps with the CPU job while it would otherwise spin in the device sync for the
    // next router (the GPU waits for this job anyway); pool_active marks a phase guests may join, guest_in counts guests in it
    bool guest = false;
    std::atomic<bool> pool_active{false};
    std::atomic<int>  guest_in{0};
    size_t n_guest_items = 0;
    struct kexpert { int32_t e; int32_t p0, p1; };     // an expert and its pairs [p0, p1) in pair order
    std::vector<kexpert> kexperts;
    std::vector<int32_t> kpair_tok, kpair_wi;          // pairs sorted by expert
    std::vector<uint8_t> kxq, kaq;                      // quantized expert input (per token) and activations (per pair)
    std::vector<float>   kact;                          // activations [n_pairs, n_ff]
    std::vector<float>   kdy;                           // LLAMA_MOE_CPU_DOWN_PE: down outputs [n_pairs, n_embd]
    q2_dot_t             q2_dot = q2_dot_select();
    q2_dot4_t            q2_dot4 = q2_dot4_select(); // LLAMA_MOE_CPU_DOT4=0: one row per call
    std::vector<q2_act>  kxp, kap;                      // Q2_0 fast path: prepared expert input (per token), activations (per pair)
    std::vector<int32_t> ktok;                          // tokens with CPU pairs
    bool                 dot4_check = getenv("LLAMA_MOE_CPU_DOT4_CHECK") != nullptr;

    std::thread worker;
    std::atomic<int32_t> job_layer{-1};  // layer to compute, -1 idle, -2 exit

    // doorbell mode (LLAMA_MOE_DOORBELL=1, pinned host memory, ubatches of up to 8 tokens): the slot lookup is a device op that
    // writes the selected experts into a pinned control block and waits there; a host thread plans the layer (LRU, CPU job, uploads)
    // and answers, and the device copies the uploads itself. No split, read back or launch per layer: one graph per ubatch
    int late_k = 0;                      // LLAMA_MOE_LATE=K: the late pass's slots per token, 0 off
    std::vector<moe_cache_lru::fill> fills_late; // the current call's late uploads
    std::vector<int32_t> late_slot;      // [n_expert] the slot of a late expert, -1 none
    ggml_backend_t lt_backend = nullptr; // the late uploads' stream
    ggml_tensor * lt_ring = nullptr;     // I32 [256] pinned: the flag values the copy stream writes
    uint32_t lt_ring_i = 0;
    std::vector<ggml_tensor *> lt_zero;  // [device] F32 [n_embd*8] zeros, the barrier's addend
    size_t n_late = 0, n_late_early = 0, n_late_calls = 0;
    bool doorbell = false;
    // doorbell background admission (LLAMA_MOE_DB_ASYNC=N, hybrid doorbell mode): the window's misses all go to the CPU (no
    // upload on the compute stream); after answering a layer the host claims LRU victims for up to N of its most-selected
    // misses and a thread copies them on a second stream, publishing each slot in the map only once its copy is done (the
    // GPU never reads a slot in flight). Victims never come from the slots answered for the running layer
    int db_async = 0;
    ggml_backend_t as_backend = nullptr;
    std::thread as_thread;
    std::mutex as_mtx;
    std::condition_variable as_cv;
    bool as_exit = false;
    struct as_job { int32_t ig, il, e, s; };
    std::vector<as_job> as_queue;
    std::vector<std::vector<uint8_t>> as_pend; // [group][key] a copy is queued or in flight
    size_t n_as_claimed = 0, n_as_published = 0, n_as_stale = 0;
    double t_as_copy_us = 0.0;
    static constexpr int64_t db_max_tok = 8;
    int64_t db_max_fill = 0;
    std::vector<int32_t> db_layers;
    std::thread db_thread;
    std::atomic<bool> db_exit{false};
    std::mutex mtx;                      // the LRU and the layer state: the doorbell thread and the copy callback (larger ubatches)
    size_t n_db_calls = 0, n_db_timeouts = 0;

    // lookahead prefetch (LLAMA_MOE_LOOKA=<B>, hybrid decode): at the slot map callback of layer il, a thread applies the router
    // of layer il+1 to il's expert input (the host copy read for the CPU path), takes the experts most predicted over the
    // ubatch's tokens (each token's top LLAMA_MOE_LOOKA_K, default 8) that are not cached, up to B, gives them slots and uploads
    // them on a second backend of the device (its own stream) while il runs; every later callback makes the compute stream wait
    // for those uploads first. Hash-routed layers (token id -> experts) are not predicted. Lossless: a wrong guess only costs
    // the upload and an evicted slot
    int     la_budget = 0, la_k = 8;
    // LLAMA_MOE_CPU_AFTER_UP=1 (hybrid decode): a layer's CPU job starts after that layer's uploads are done (the copy engine
    // then has the RAM bandwidth to itself; on a board whose RAM is not much faster than the link the two halve each other)
    bool    cpu_after_up = getenv("LLAMA_MOE_CPU_AFTER_UP") && atoi(getenv("LLAMA_MOE_CPU_AFTER_UP")) != 0;
    ggml_backend_event_t up_ev = nullptr;
    std::atomic<int> up_state{2};
    double  t_after_up_us = 0.0;
    size_t  n_after_up = 0;
    bool    la_all = false;                 // LLAMA_MOE_LOOKA_ALL=1: also in decode ubatches without the CPU path (gate off)
    size_t  n_la_all_jobs = 0;
    bool    la_pred_on = false;             // the graphs compute the next layer's predicted experts (lookahead or pre-read)
    // LLAMA_MOE_PREREAD=1 (hybrid decode): after its job, the CPU pool reads the predicted experts of the next layer that are not
    // cached (the ones the CPU will likely compute) into its caches, using RAM bandwidth that is idle between jobs; a new job
    // stops it within one 64 KiB chunk. Lossless (only a read)
    bool    preread = false;
    // LLAMA_MOE_WILLNEED=1 (experiment, mapped experts): right after a layer's plan, madvise(MADV_WILLNEED) each missed expert
    // (the uploads and the CPU's), so the pages of a mapped file that are not in RAM are read from the drive in parallel
    // instead of one page fault at a time by the upload copy or the CPU job. Lossless (only a read)
    bool    willneed = false;
    size_t  n_wn_calls = 0, n_wn_ranges = 0;
    double  t_wn_us = 0.0;
    std::unique_ptr<moe_arena> arena;       // LLAMA_MOE_ARENA_MIB, see moe_arena
    bool arena_store = false;               // LLAMA_MOE_ARENA_MIB=auto: the arena is the experts' only RAM copy
    bool arena_excl = false;                // LLAMA_MOE_ARENA_EXCL (store mode): see moe_arena::release
    double wb_bytes = 0.0;
    // early copy-back (exclusive arena, one device; LLAMA_MOE_ARENA_EARLY=1, opt-in: pcbox DS4.1 64 GB 10.20/10.09 vs 10.47/10.17
    // tok/s without, no gain): after a layer's uploads, the group's next
    // LRU victims that the arena lacks are copied back on a second stream (wb_backend) while the GPU computes, so the evictions
    // find them in RAM and the compute stream copies nothing back (DS4.1, 9 MiB experts: the inline copy-back, in the
    // uploads' stream order, took the layer's sync from 929 to 1660 us). Every reader of the arena syncs wb_backend first
    ggml_backend_t wb_backend = nullptr;
    size_t n_wb_early = 0;
    double wb_early_bytes = 0.0;
    // LLAMA_MOE_ARENA_EARLY_EV=1 (with EARLY=1): no host wait for the copy-backs where only the GPU reads the arena (every miss
    // uploaded): the compute stream waits for them on an event before its uploads, and the copy-backs wait for the uploads
    // (both directions at once share the host RAM bandwidth: pcbox1 DDR4 uploads 1113 -> 1598 ms, copy-backs 804 -> 1368 ms per 5 s)
    bool wb_ev_mode = false;
    ggml_backend_event_t wb_ev = nullptr, wbup_ev = nullptr;
    size_t n_wb_gpu_waits = 0;
    void wb_sync(ggml_backend_t gpu_waiter = nullptr) {
        if (!wb_backend) {
            return;
        }
        if (wb_ev_mode && gpu_waiter != nullptr && wb_ev != nullptr) {
            ggml_backend_event_wait(gpu_waiter, wb_ev);
            n_wb_gpu_waits++;
            return;
        }
        ggml_backend_synchronize(wb_backend);
    }
    // only the GPU reads the arena in this call: every miss is uploaded, no shadow pass
    bool gpu_only_arena(bool hyb) const { return !shadow && (!hyb || cur_share() >= 1.0); }
    bool excl_verify = getenv("LLAMA_MOE_ARENA_EXCL_VERIFY") && atoi(getenv("LLAMA_MOE_ARENA_EXCL_VERIFY")) != 0;
    std::vector<std::pair<int32_t, int32_t>> wb_last;   // excl_verify: the last call's copied back experts
    ggml_backend_buffer_ptr arena_buf;
    std::vector<int32_t> arena_need, arena_pred;
    int fault_pred = 0;                     // LLAMA_MOE_FAULT_PRED=K (with the no-wait fault threads): see moe_arena::ensure
    // LLAMA_MOE_FAULT_THREADS unset: the fault threads are made when the host experts that are not in VRAM are over half the
    // RAM, and used while the process takes many major page faults (fault_check); LLAMA_MOE_FAULT_AUTO=0: never
    bool    fault_auto = false, fault_pred_auto = false, fault_on = false;
    // LLAMA_MOE_ARENA_MIB unset: store mode (auto size, async reads, exclusive) when the host experts not in VRAM are over
    // 85% of the RAM (pcbox 32 GB Qwen3.8 IQ3_S: 34.1 / 35.1 vs 29.0 / 28.3 tok/s, 41% fewer SSD bytes); LLAMA_MOE_STORE_AUTO=0: never
    bool    store_auto = false;
    // LLAMA_MOE_COLD=1 (off by default): after an expert is uploaded from the mapping, its pages are marked MADV_COLD, so the
    // page cache drops them first. Slower: pcbox 30G cgroup 18.6 -> 14.3-15.2 tok/s, 20-40% more SSD reads (the VRAM cache
    // evicts and uploads them again soon after)
    bool    cold_on = false;
    size_t  n_cold = 0;
    // set_shadow: the current decode is a shadow pass; experts kept (in RAM, computed by the CPU) and left out (read for later)
    bool    shadow = false;
    size_t  n_shadow_calls = 0, n_shadow_kept = 0, n_shadow_dropped = 0, n_shadow_cut = 0;
    // LLAMA_MOE_SHADOW=2: GPU-only shadow pass, one graph without the per-layer callbacks: the experts not in VRAM read the zero
    // slot through a slot map on the device; after the pass the selected experts that are not RAM-resident are read ahead
    bool    shadow_gpu_mode = false, shadow_gpu = false;

    // LLAMA_MOE_GPU_FETCH=1 (one CUDA device, every expert in registered (pinned, mapped) host memory, decode ubatches of up to
    // 8 tokens): the decode graph has no per-layer split. Each layer's slot map lives on the device; an expert that is not
    // cached maps to -(e+1) and the MMVQ kernels read it in place from host memory. Before each ubatch the host reads the
    // previous decode graph's selected experts, updates the LRU, admits LLAMA_MOE_FETCH_SHARE (default 0.5) of the misses
    // (most used first) with copies on a second stream, and uploads the maps: an expert still being copied keeps reading in
    // place, an evicted one at once; the copies are published at the next ubatch (its stream waits for them). Lossless.
    bool    fetch_mode = false, fetch = false, fetch_armed = false;
    int64_t fetch_max = 0;                                  // the largest decode ubatch the fetch graphs take (MMVQ's limit)
    double  fetch_share = 0.5;
    ggml_tensor * fmap_all = nullptr;
    ggml_backend_t fb_backend = nullptr;
    ggml_backend_event_t fb_event = nullptr, fe_event = nullptr;
    struct fb_round { ggml_backend_event_t ev = nullptr; std::vector<std::pair<int32_t, int32_t>> slots; };
    std::deque<fb_round> fb_rounds;                         // admissions queued, not yet published (oldest first)
    std::vector<ggml_backend_event_t> fb_free_ev;
    std::vector<std::vector<uint8_t>> fb_pend;              // [group][slot] copies in flight
    std::vector<int64_t> fb_ids_n;
    std::thread fh_thread;
    std::atomic<int>  fh_job{0};
    std::atomic<bool> fh_busy{false};
    double  t_fetch_end_us = 0.0, t_fetch_helper_us = 0.0;
    std::vector<int32_t> fb_ids, fb_rest;
    ggml_backend_buffer_ptr fb_pin_buf;                     // pinned: the maps as uploaded, then the selected experts as read
    int32_t * fb_map_pin = nullptr, * fb_ids_pin = nullptr;
    std::vector<size_t> fb_ids_off;
    std::vector<moe_cache_lru::fill> fb_fills;
    size_t  n_fetch_ubatches = 0, n_fetch_admitted = 0, n_fetch_inplace = 0;
    double  t_fetch_us = 0.0;
    size_t  n_shadow_ram = 0;
    std::vector<int32_t> shadow_drop;
    // whether expert e of a layer is in RAM: the first, middle and last page of each of its tensors (mincore)
    bool expert_resident(const layer & l, int32_t e) const {
#ifdef __linux__
        static const uintptr_t page = (uintptr_t) sysconf(_SC_PAGE_SIZE);
        for (const ggml_tensor * w : l.experts) {
            const uintptr_t b = (uintptr_t) w->data + (uintptr_t) e*w->nb[2];
            for (const uintptr_t a : { b, b + w->nb[2]/2, b + w->nb[2] - 1 }) {
                unsigned char v = 0;
                if (mincore((void *) (a & ~(page - 1)), page, &v) == 0 && !(v & 1)) {
                    return false;
                }
            }
        }
#else
        GGML_UNUSED(l); GGML_UNUSED(e);
#endif
        return true;
    }
    int64_t fault_t0 = 0;
    long    fault_maj0 = 0;
    size_t  n_fault_switch = 0;
    double  t_fault_us = 0.0, t_fault_on_us = 0.0;
    // every 0.5 s: on above 500 major faults per second, off below 100 (fits in RAM: under 10; 32 GB for 78 GiB: thousands)
    void fault_check() {
#ifdef __linux__
        const int64_t now = ggml_time_us();
        if (now - fault_t0 < 500000) {
            return;
        }
        rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        if (fault_t0 > 0) {
            const double rate = (ru.ru_majflt - fault_maj0)*1e6/(now - fault_t0);
            t_fault_us += now - fault_t0;
            if (fault_on) {
                t_fault_on_us += now - fault_t0;
            }
            const bool on = fault_on ? rate > 100 : rate > 500;
            if (on != fault_on) {
                fault_on = on;
                n_fault_switch++;
                LLAMA_LOG_DEBUG("llama_moe_cache: fault threads %s (%.0f major faults/s)\n", on ? "on" : "off", rate);
            }
        }
        fault_t0   = now;
        fault_maj0 = ru.ru_majflt;
#endif
    }
    // an expert tensor of layer il from the arena, else from the model mapping
    const uint8_t * expert_data(const layer & l, int32_t il, const ggml_tensor * w, int32_t e) const {
        if (arena && (!l.pf_job || arena_store)) {
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                if (l.experts[ip] == w) {
                    if (const uint8_t * p = arena->get_ready(il, (int32_t) ip, e)) {
                        return p;
                    }
                    break;
                }
            }
        }
        return (const uint8_t *) w->data + (size_t) e*w->nb[2];
    }
    std::vector<std::pair<const uint8_t *, size_t>> pr_chunks; // set by the callback before it posts the job
    std::atomic<bool> pr_ready{false};
    std::atomic<bool> pr_running{false}, pr_stop{false};
    size_t  n_pr_jobs = 0, n_pr_aborted = 0, n_pr_experts = 0;
    double  pr_bytes = 0.0, t_pr_us = 0.0;
    volatile uint64_t pr_sink = 0;
    ggml_backend_t la_backend = nullptr;
    ggml_backend_event_t la_event = nullptr;
    std::vector<ggml_tensor *> la_gi;       // [layer] router weights of the layer (on the device), nullptr: not predicted
    std::vector<ggml_tensor *> la_bi;       // [layer] its selection bias, may be nullptr
    std::vector<uint8_t> la_hash;           // [layer] hash-routed
    uint32_t la_gating = 0;
    std::thread la_thread;
    std::atomic<int32_t> la_job{-1};        // target layer of the posted job, -1 none, -2 exit
    std::atomic<bool>    la_busy{false};    // a job is posted or running
    std::atomic<bool>    la_cancel{false};  // the next callback came before the job reached its slots: drop it
    bool    la_pending = false;             // uploads issued on la_backend that the compute stream has not waited for
    int32_t la_src = -1;                    // the layer whose callback posted the job (its slots are not evicted)
    std::vector<int32_t> la_ids;            // the job's predicted experts [n_tok, la_k], computed in the graph
    ggml_tensor * rb_la = nullptr;          // pinned read back of the predictions
    std::vector<std::vector<int32_t>> la_mark; // [group][slot] key+1 of a prefetched expert not used yet, 0 none
    // LLAMA_MOE_LOOKA_LAZY (default on): the compute stream waits for the prefetch uploads only when a layer reads or refills a
    // slot that is still being written, so the uploads may overlap more than one layer
    bool la_lazy = true;
    std::vector<std::vector<uint8_t>> la_pend; // [group][slot] written by uploads the compute stream has not waited for
    std::vector<std::pair<int32_t, int32_t>> la_pend_list;
    size_t n_la_joins = 0;
    size_t n_la_jobs = 0, n_la_issued = 0, n_la_useful = 0, n_la_skipped = 0, n_la_cancelled = 0;
    double t_la_us = 0.0, t_la_wait_us = 0.0;

    // prefill share (LLAMA_MOE_PREFILL_CPU=auto, or a fixed share 0..1; needs hybrid flag mode): ubatches of 33 to pf_max tokens
    // take the large-batch path, which uploads every expert they use that is not in the cache. Here, as with Strata's
    // STRATA_PREFILL_CPU_SHARE, the CPU computes a share of those experts, the ones fewest tokens use (at most pf_maxt), so they
    // are not uploaded: on the device their pairs are pointed at an uploaded expert the token does not use, with router weight 0
    // (MMQ needs distinct experts per token), and the GPU adds the CPU's output after the layer's experts (host_add).
    // auto: share = g/(c + g), c the CPU's time per expert, g the GPU's time per uploaded expert, both measured per layer
    bool    pf = false;
    double  pf_fixed = -1.0;
    int64_t pf_max = 1024;
    int32_t pf_maxt = 8;
    double  pf_c_us = 20.0, pf_g_us = 72.0;
    bool    pf_measured = false;
    ggml_tensor * pf_out = nullptr, * pf_x = nullptr, * pf_w = nullptr; // pinned: the CPU's output, read backs of x and w
    int32_t pf_prev_il = -1;            // the previous layer of the current ubatch that had a prefill CPU job
    int64_t pf_prev_t0 = 0, pf_prev_ncpu = 0, pf_prev_ngpu = 0;
    size_t  n_pf_layers = 0, n_pf_cpu = 0, n_pf_stream = 0, n_pf_pairs = 0, n_pf_mismatch = 0, n_pf_touch = 0;
    bool    pf_touch = true;            // LLAMA_MOE_PREFILL_TOUCH (see prefill_plan)
    int64_t pf_arena_us = 0;            // store mode: the last prefill_arena's read time
    double  t_pf_job_us = 0.0, t_pf_wait_us = 0.0, t_pf_plan_us = 0.0;
    std::vector<int32_t> pf_cnt;
    std::vector<uint8_t> pf_on_cpu;
    // CPU job phases (us, summed): wake (job posted to worker start), prep (pairs, input quantization), A (gate/up), A2, B (down), post
    double t_ph[6] = {};
    double t_hot[2] = {}; size_t n_hot = 0;
    // LLAMA_MOE_HOT_SAVE=<path>: at exit, one line "layer expert calls" per expert the decode plans selected, most calls first:
    // the hot list for LLAMA_MOE_HOT_LIST (collect it with the prompts the box will serve)
    std::string hot_save;
    std::vector<uint64_t> hot_count;
    int32_t hot_n_expert = 0;

    void save_hot() const {
        FILE * f = fopen(hot_save.c_str(), "w");
        if (f == nullptr) {
            LLAMA_LOG_WARN("llama_moe_cache: hot list: cannot write %s\n", hot_save.c_str());
            return;
        }
        std::vector<size_t> order;
        for (size_t k = 0; k < hot_count.size(); ++k) {
            if (hot_count[k] > 0) {
                order.push_back(k);
            }
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return hot_count[a] > hot_count[b]; });
        for (size_t k : order) {
            fprintf(f, "%zu %zu %llu\n", k / hot_n_expert, k % hot_n_expert, (unsigned long long) hot_count[k]);
        }
        fclose(f);
        LLAMA_LOG_INFO("llama_moe_cache: hot list: %zu experts written to %s\n", order.size(), hot_save.c_str());
    }

    // LLAMA_MOE_HOT_LIST=<path> (a LLAMA_MOE_HOT_SAVE file): read the hottest host experts into RAM (the page cache of the
    // mapped model), up to LLAMA_MOE_HOT_WARM_MIB or by default MemAvailable less 8 GiB, by 16 threads (MADV_WILLNEED, then
    // one read per page). The page cache still decides what stays afterwards; this only moves the first requests' reads from
    // the SSD to the time before them, like a hot RAM store does. By default in the background, hottest first, and only while
    // no ubatch has used the cache for 300 ms (the server is ready at once and requests keep the drive); LLAMA_MOE_HOT_SYNC=1:
    // at load, coldest first (if RAM runs short the kernel drops those before the hottest)
    std::thread hot_thread;
    std::atomic<bool> hot_stop{false};
    std::atomic<int64_t> hot_busy_us{0};   // the last cache callback

    // LLAMA_MOE_HOT_LIST (LLAMA_MOE_HOT_VRAM=0: off): at start, each group's empty cache slots take its hottest experts of the
    // list, placed coldest first so that the hottest are the most recently used (Strata pre-fills its slots from a shipped
    // profile): the first requests start from a warm cache instead of an empty one
    void fill_hot(const char * path) {
        std::ifstream f(path);
        std::vector<std::vector<std::pair<int32_t, int32_t>>> sel(groups.size());
        for (long long il, e, n; f >> il >> e >> n; ) {
            if (il < 0 || il >= (long long) layers.size() || layers[il].ig < 0 || e < 0 || e >= layers[il].slot_map->ne[1]) {
                continue;
            }
            auto & s = sel[layers[il].ig];
            if ((int32_t) s.size() < groups[layers[il].ig].n_slots) {
                s.push_back({ (int32_t) il, (int32_t) e });
            }
        }
        const int64_t t0 = ggml_time_us();
        size_t n = 0, bytes = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            for (auto it = sel[ig].rbegin(); it != sel[ig].rend(); ++it) {
                const auto [il, e] = *it;
                int32_t * map = g.lru.slot_map[il];
                const int32_t s = g.lru.head;
                if (map == nullptr || map[e] >= 0 || s < 0 || g.lru.key_of[s] >= 0) {
                    continue;
                }
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const size_t nb2 = l.experts[ip]->nb[2];
                    ggml_backend_tensor_set(g.banks[ip], (const uint8_t *) l.experts[ip]->data + (size_t) e*nb2, (size_t) s*nb2, nb2);
                    bytes += nb2;
                }
                g.lru.key_of[s] = il*g.lru.n_expert + e;
                map[e] = s;
                g.lru.touch(s);
                n++;
            }
        }
        LLAMA_LOG_INFO("llama_moe_cache: hot list: %zu experts (%.2f GiB) placed in the VRAM cache in %.1f s from %s\n",
            n, bytes/1073741824.0, (ggml_time_us() - t0)/1e6, path);
    }

    // store mode, at start (LLAMA_MOE_ARENA_FILL=0: off): when the VRAM cache and the arena together hold every expert, each
    // group's empty VRAM slots take an even share of its layers' experts and the arena reads the rest (Strata reads the
    // complement of its GPU cache into RAM at load). Exclusive mode keeps it that way (a VRAM victim goes back to the arena,
    // an upload frees its arena slot): decode and prompts read nothing from the SSD. pcbox1 32 GB + RTX 5090, Qwen3.8 IQ3_S:
    // filling by misses read 93 GB in a 16-question run. Returns false when the experts do not fit (nothing is placed).
    bool fill_complement() {
        if (!arena_store || !arena || arena->base == nullptr || (getenv("LLAMA_MOE_ARENA_FILL") && atoi(getenv("LLAMA_MOE_ARENA_FILL")) == 0)) {
            return false;
        }
        const int64_t t0 = ggml_time_us();
        const size_t n_pool = arena->pool_size.size();
        std::vector<int64_t> pool_free(n_pool, 0), pool_need(n_pool, 0);
        for (int32_t v = 0; v < arena->n_slots; ++v) {
            pool_free[arena->pool_of_slot[v]] += arena->key_of[v] < 0;
        }
        for (int64_t & f : pool_free) {
            f -= std::max<int64_t>(16, f/100);      // slack: VRAM victims copied back, the slots of the protected calls
        }
        std::vector<std::vector<int32_t>> to_vram(layers.size()), to_arena(layers.size());
        size_t n_all = 0, n_vram = 0;
        for (const group & g : groups) {
            int32_t n_free = 0;
            for (int32_t v = 0; v < g.n_slots; ++v) {
                n_free += g.lru.key_of[v] < 0;
            }
            const int32_t nl = (int32_t) g.layers.size();
            for (int32_t k = 0; k < nl; ++k) {
                const int32_t il = g.layers[k];
                const int32_t * map = g.lru.slot_map[il];
                if (map == nullptr) {
                    continue;
                }
                int32_t q = n_free/nl + (k < n_free % nl);
                const int32_t ne = (int32_t) layers[il].slot_map->ne[1];
                const int32_t ip = arena->pool_of[il];
                for (int32_t e = 0; e < ne; ++e) {
                    n_all++;
                    if (map[e] >= 0) {
                        n_vram++;
                    } else if (q > 0) {
                        to_vram[il].push_back(e);
                        q--;
                    } else if (ip >= 0 && arena->slot_of[il][e] < 0) {
                        to_arena[il].push_back(e);
                        pool_need[ip]++;
                    } else if (ip < 0) {
                        return false;               // a layer the arena does not hold
                    }
                }
            }
        }
        // a pool a few slots short (<= 3% of its need) is trimmed, one expert per layer round-robin, so the fill still holds all but
        // those (they stay on the SSD, read on a miss); before, one pool 8 short of 262 dropped the whole fill (pcbox1 32 GB + 5090,
        // Qwen3.8: a 16k prompt read 578 GiB from the SSD instead of nothing)
        size_t n_left = 0;
        for (size_t ip = 0; ip < n_pool; ++ip) {
            if (pool_need[ip] <= pool_free[ip]) {
                continue;
            }
            const int64_t over = pool_need[ip] - std::max<int64_t>(0, pool_free[ip]);
            if (over*100 > pool_need[ip]*3 || (getenv("LLAMA_MOE_ARENA_FILL_PART") && atoi(getenv("LLAMA_MOE_ARENA_FILL_PART")) == 0)) {
                LLAMA_LOG_INFO("llama_moe_cache: arena fill: the experts do not fit (pool %zu: %lld more needed, %lld free); filled by misses\n",
                    ip, (long long) pool_need[ip], (long long) pool_free[ip]);
                return false;
            }
            for (int64_t k = over; k > 0; ) {
                const int64_t k0 = k;
                for (size_t il = 0; il < layers.size() && k > 0; ++il) {
                    if (arena->pool_of[il] == (int32_t) ip && !to_arena[il].empty()) {
                        to_arena[il].pop_back();
                        k--;
                    }
                }
                if (k == k0) {
                    break;
                }
            }
            pool_need[ip] -= over;
            n_left += over;
        }
        // the VRAM share is staged through the arena's (pinned) slots, a layer at a time: O_DIRECT reads at the drive's rate and
        // DMA uploads (through the mapping: ~1.1 GB/s of page faults on pcbox1); the slots are released after the uploads
        auto wait_reads = [&] {
            while (arena->done.load(std::memory_order_acquire) < arena->jobs.size()) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        };
        size_t nv = 0, bv = 0;
        for (group & g : groups) {
            for (int32_t il : g.layers) {
                int32_t * map = g.lru.slot_map[il];
                const layer & l = layers[il];
                if (!to_vram[il].empty()) {
                    arena->ensure(il, to_vram[il].data(), to_vram[il].size());
                    wait_reads();
                }
                for (int32_t e : to_vram[il]) {
                    const int32_t v = g.lru.head;
                    if (map == nullptr || v < 0 || g.lru.key_of[v] >= 0) {
                        break;
                    }
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        const size_t nb2 = l.experts[ip]->nb[2];
                        const uint8_t * src = arena->get_ready(il, (int32_t) ip, e);
                        ggml_backend_tensor_set(g.banks[ip], src ? src : (const uint8_t *) l.experts[ip]->data + (size_t) e*nb2, (size_t) v*nb2, nb2);
                        bv += nb2;
                    }
                    arena->release(il, e);
                    g.lru.key_of[v] = il*g.lru.n_expert + e;
                    map[e] = v;
                    g.lru.touch(v);
                    nv++;
                }
            }
        }
        const double t1 = (ggml_time_us() - t0)/1e6;
        const double b0 = arena->bytes;
        size_t na = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            if (!to_arena[il].empty()) {
                arena->ensure((int32_t) il, to_arena[il].data(), to_arena[il].size());
                na += to_arena[il].size();
            }
        }
        wait_reads();
        const double t2 = (ggml_time_us() - t0)/1e6;
        if (n_left == 0) {
            LLAMA_LOG_INFO("llama_moe_cache: arena fill: all %zu experts held: %zu in VRAM before, %zu placed in VRAM (%.1f GiB, %.1f s), %zu read into the arena (%.1f GiB, %.1f s); nothing left on the SSD\n",
                n_all, n_vram, nv, bv/1073741824.0, t1, na, (arena->bytes - b0)/1073741824.0, t2 - t1);
        } else {
            LLAMA_LOG_INFO("llama_moe_cache: arena fill: %zu of %zu experts held: %zu in VRAM before, %zu placed in VRAM (%.1f GiB, %.1f s), %zu read into the arena (%.1f GiB, %.1f s); %zu left on the SSD (the arena was short)\n",
                n_all - n_left, n_all, n_vram, nv, bv/1073741824.0, t1, na, (arena->bytes - b0)/1073741824.0, t2 - t1, n_left);
        }
        arena->n_reads = arena->n_hits = arena->n_calls = 0;
        arena->bytes = arena->t_us = 0.0;
        if (hybrid_auto) {
            // every miss is in pinned RAM, as with LLAMA_MOE_PIN_EXPS, where the gate settles on GPU-only: upload every miss
            // (LLAMA_MOE_CPU=1). Not the gate's off state: graphs without the CPU path take the plain upload path, which
            // does not copy VRAM victims back (exclusive mode), so they fell back to the SSD and the gate saw GPU-only lose
            // (pcbox1 32 GB + 5090: LLAMA_MOE_CPU=1 143 / 145 tok/s, auto 104, gate starting off 50-56)
            hybrid_auto  = false;
            hybrid_share = 1.0;
            LLAMA_LOG_INFO("llama_moe_cache: arena fill: every miss is uploaded from pinned RAM (as LLAMA_MOE_CPU=1)\n");
        }
        return true;
    }

    void warm_hot(const char * path, bool sync) {
        std::ifstream f(path);
        std::vector<std::pair<int32_t, int32_t>> list;
        for (long long il, e, n; f >> il >> e >> n; ) {
            if (il >= 0 && il < (long long) layers.size() && layers[il].ig >= 0 && e >= 0 && e < layers[il].slot_map->ne[1]) {
                list.push_back({ (int32_t) il, (int32_t) e });
            }
        }
        if (list.empty()) {
            LLAMA_LOG_WARN("llama_moe_cache: hot list: no experts of this model in %s\n", path);
            return;
        }
        size_t budget = 0;
        if (const char * b = getenv("LLAMA_MOE_HOT_WARM_MIB")) {
            budget = (size_t) atoll(b) << 20;
        } else {
            std::ifstream mi("/proc/meminfo");
            std::string k;
            size_t v = 0;
            while (mi >> k >> v) {
                if (k == "MemAvailable:") {
                    budget = v << 10;
                    break;
                }
                std::getline(mi, k);
            }
            const size_t lim = llama_moe_ram_limit();
            if (lim > 0) {
                budget = std::min(budget, lim);
            }
            budget = budget > (8ull << 30) ? budget - (8ull << 30) : 0;
        }
        size_t n = 0, bytes = 0;
        for (; n < list.size(); ++n) {
            size_t b = 0;
            for (const ggml_tensor * t : layers[list[n].first].experts) {
                b += t->nb[2];
            }
            if (bytes + b > budget) {
                break;
            }
            bytes += b;
        }
        if (sync) {
            std::reverse(list.begin(), list.begin() + n);
        }
        const size_t page = 4096;
        std::atomic<int64_t> t_wait{0};
        const int64_t t0 = ggml_time_us();
        std::atomic<size_t> next{0};
        std::vector<std::thread> th;
        for (int i = 0; i < 16; ++i) {
            th.emplace_back([&] {
                uint64_t sink = 0;
                for (size_t j; !hot_stop.load(std::memory_order_relaxed) && (j = next.fetch_add(1)) < n; ) {
                    if (!sync) {
                        const int64_t w0 = ggml_time_us();
                        while (!hot_stop.load(std::memory_order_relaxed) && ggml_time_us() - hot_busy_us.load(std::memory_order_relaxed) < 300000) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        }
                        t_wait.fetch_add(ggml_time_us() - w0, std::memory_order_relaxed);
                    }
                    for (const ggml_tensor * t : layers[list[j].first].experts) {
                        const uint8_t * p = (const uint8_t *) t->data + (size_t) list[j].second*t->nb[2];
                        const uintptr_t a = (uintptr_t) p & ~(uintptr_t) (page - 1);
                        const uintptr_t z = (uintptr_t) p + t->nb[2];
#ifdef __linux__
                        madvise((void *) a, z - a, MADV_WILLNEED);
#endif
                        for (uintptr_t q = a; q < z; q += page) {
                            sink += *(const volatile uint8_t *) std::max(q, (uintptr_t) p);
                        }
                    }
                }
                GGML_UNUSED(sink);
            });
        }
        for (auto & t : th) {
            t.join();
        }
        const double dt = (ggml_time_us() - t0)/1e6;
        LLAMA_LOG_INFO("llama_moe_cache: hot list: %zu of %zu experts (%.1f GiB of %.1f GiB budget) read into RAM in %.1f s (%.2f GB/s)%s%s from %s\n",
            n, list.size(), bytes/1073741824.0, budget/1073741824.0, dt, bytes/1e9/std::max(dt, 1e-3),
            sync ? "" : format(" in the background, %.1f thread-s paused for requests", t_wait.load()/1e6).c_str(),
            hot_stop.load() ? " (stopped at exit)" : "", path);
    }

    // LLAMA_MOE_STATS_FILE=<path>: about once a second (at a cache callback) a JSON line of cumulative counters, for profiles
    FILE * stats_f = nullptr;
    int64_t stats_t = 0;
    void stats_tick() {
        const int64_t now = ggml_time_us();
        hot_busy_us.store(now, std::memory_order_relaxed);
        if (stats_f == nullptr || now - stats_t < 1000000) {
            return;
        }
        stats_t = now;
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
        fprintf(stats_f, "{\"t\": %.3f, \"cpu_job_us\": %.0f, \"cpu_pairs\": %zu, \"cpu_experts\": %zu, \"up_bytes\": %zu, \"copy_bytes\": %zu, "
            "\"hits\": %zu, \"misses\": %zu, \"flag_wait_us\": %.0f, \"la_issued\": %zu, \"pf_job_us\": %.0f, \"pf_cpu\": %zu, \"share\": %.3f, "
            "\"calls\": %zu, \"sync_us\": %.0f, \"plan_us\": %.0f}\n",
            ts.tv_sec + ts.tv_nsec*1e-9, t_cpu_us + t_pf_job_us, n_cpu_pairs + n_pf_pairs, n_cpu_experts + n_pf_cpu,
            stats_small.bytes + stats_large.bytes, stats_copy.bytes, stats_small.hits, stats_small.misses, t_flag_wait_us + t_pf_wait_us,
            n_la_issued, t_pf_job_us, n_pf_cpu, hybrid_share, n_hybrid_calls, t_sync_us, t_upload_us);
        fflush(stats_f);
    } // LLAMA_MOE_CPU_HOT_TEST: phases A..B cold, then hot
    size_t n_ph = 0;
    size_t n_ph_kexp = 0;   // (layer job, expert) groups of CPU pairs: n_cpu_pairs / n_ph_kexp = tokens per expert
    size_t n_ph_mt   = 0;   // of those, the groups the multi-token i-quant kernels took (2+ tokens)
    double t_db_us = 0.0, t_db_gpu_wait_us = 0.0;
    std::atomic<bool>    job_done{true};

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_tensor * rb_ids = nullptr, * rb_x = nullptr, * rb_w = nullptr; // pinned staging of the read backs of the slot map callback
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
            const std::map<ggml_backend_dev_t, size_t> & dev_budget, size_t flex_size_req = 0) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }
        for (const group & g : groups) {
            for (int32_t il : g.layers) {
                il_first = il_first < 0 ? il : std::min(il_first, il);
                il_last  = std::max(il_last, il);
            }
        }

        n_embd = model.hparams.n_embd;
        if (const char * sf = getenv("LLAMA_MOE_STATS_FILE")) {
            stats_f = fopen(sf, "w");
        }
        // the rule is the default in auto mode (DS4 16 GiB: 38.1 vs 33.4 tok/s for the wait-based controller; equal on Qwen);
        // LLAMA_MOE_SHARE_RULE=0 goes back to the waits, =<k> sets the factor
        share_rule = true;
        if (const char * r = getenv("LLAMA_MOE_SHARE_RULE")) {
            share_rule = atof(r) > 0.0;
            rule_k = atof(r) == 1.0 || atof(r) <= 0.0 ? 2.25 : atof(r);
        }
        if (const char * st = getenv("LLAMA_MOE_SHARE_TUNE")) {
            share_tune = atoi(st) != 0;
        }
        if (const char * s = getenv("LLAMA_MOE_CPU")) {
            hybrid       = true;
            hybrid_auto  = strcmp(s, "auto") == 0;
            hybrid_share = hybrid_auto ? 0.25 : std::clamp(atof(s), 0.0, 1.0);
            if (hybrid_auto && getenv("LLAMA_MOE_GATE_START") && atoi(getenv("LLAMA_MOE_GATE_START")) == 0) {
                gate_on = path_on = false; // testing: the gate starts with the CPU path off
            }
            if (const char * lk = getenv("LLAMA_MOE_LATE")) {
                late_k = std::clamp(atoi(lk), 0, 10);
            }
            if (const char * t = getenv("LLAMA_MOE_TAIL_SKIP")) {
                tail_rank = std::max(0, atoi(t));
                if (tail_rank > 0) {
                    LLAMA_LOG_WARN("%s: LLAMA_MOE_TAIL_SKIP=%d: missed experts that every token routes at rank %d or lower are skipped "
                        "(approximate, not the model's output)\n", __func__, tail_rank, tail_rank);
                }
            }
            if (const char * t = getenv("LLAMA_MOE_CPU_THREADS")) {
                n_threads = std::max(1, atoi(t));
            }
            n_embd = model.hparams.n_embd;
            cpu_backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
            if (!cpu_backend) {
                throw std::runtime_error("MoE cache hybrid mode: failed to create a CPU backend");
            }
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu_backend.get()));
            auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (set_n_threads) {
                set_n_threads(cpu_backend.get(), n_threads);
            }
            LLAMA_LOG_INFO("%s: hybrid mode: %s%.0f%% of the misses are uploaded, the CPU computes the others with %d threads\n", __func__,
                hybrid_auto ? "auto, starting at " : "", 100.0*hybrid_share, n_threads);
            if (const char * k = getenv("LLAMA_MOE_CPU_KERNEL")) {
                use_kernel = strcmp(k, "ggml") != 0;
            }
            guest = use_kernel && getenv("LLAMA_MOE_CPU_GUEST") && atoi(getenv("LLAMA_MOE_CPU_GUEST")) != 0;
            if (const char * p = getenv("LLAMA_MOE_PREFILL_CPU"); p && use_kernel && strcmp(p, "0") != 0) {
                pf       = true;
                pf_fixed = strcmp(p, "auto") == 0 || strcmp(p, "1") == 0 ? -1.0 : std::clamp(atof(p), 0.0, 1.0);
                if (const char * m = getenv("LLAMA_MOE_PREFILL_MAX")) {
                    pf_max = std::max(33, atoi(m));
                }
                if (const char * m = getenv("LLAMA_MOE_PREFILL_MAXT")) {
                    pf_maxt = std::max(1, atoi(m));
                }
                pf_touch = !getenv("LLAMA_MOE_PREFILL_TOUCH") || atoi(getenv("LLAMA_MOE_PREFILL_TOUCH")) != 0;
            }
            pin_cpus = pin_list();
            if (!pin_cpus.empty()) {
                std::string l;
                for (int i = 0; i < std::min<int>(n_threads, (int) pin_cpus.size()); ++i) l += (i ? "," : "") + std::to_string(pin_cpus[i]);
                LLAMA_LOG_INFO("%s: hybrid mode: CPU threads pinned to %s\n", __func__, l.c_str());
            }
            if (use_kernel) {
                for (int i = 1; i < n_threads; ++i) {
                    helpers.emplace_back([this, i] { helper_loop(i); });
                }
            }
            bool any_q2 = false;
            for (const group & g : groups) {
                for (const ggml_tensor * t : g.ref) any_q2 |= t->type == GGML_TYPE_Q2_0;
            }
            LLAMA_LOG_INFO("%s: hybrid mode: CPU experts by %s%s\n", __func__, use_kernel ? "the MoE cache's kernel" : "ggml graphs",
                use_kernel && q2_dot && any_q2 ? q2_dot_name() : "");
            worker = std::thread([this] { worker_loop(); });
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        // hybrid mode: one more, zero filled, which the experts computed by the CPU read on the device
        const int32_t n_extra = hybrid ? 2 : 1;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + n_extra), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free && dev_budget.count(dev)) {
                d.split = (double) dev_budget.at(dev);
            } else if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // slot lending: one GPU, a VMM-capable device (else the flex bytes stay free for the compute buffers)
        size_t vmm_chunk = 0;
        if (flex_size_req > 0 && devices.size() == 1 && !no_alloc && flex_size_req < size) {
            ggml_backend_dev_t dev = ggml_backend_get_device(devices[0].backend);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            auto fchunk = reg ? (size_t (*)(ggml_backend_buffer_type_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_vmm_chunk") : nullptr;
            vmm_remap   = reg ? (vmm_remap_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_vmm_buffer_remap") : nullptr;
            vmm_chunk   = fchunk && vmm_remap ? fchunk(devices[0].buft) : 0;
            if (vmm_chunk > 0) {
                flex_bytes = flex_size_req;
            } else {
                vmm_remap = nullptr;
                LLAMA_LOG_WARN("%s: slot lending: no VMM on this device; %zu MiB stay free for the prompt batches' compute buffers\n",
                    __func__, flex_size_req >> 20);
                size -= flex_size_req;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size()) + g.layers.size() + (late_k > 0 ? g.layers.size()*3 + 1 : 0) + g.layers.size() + 1; // + the GPU-only shadow's slot maps, the late pass, the GPU fetch maps
            n_tensors_host  += g.layers.size()*(hybrid ? 6 : 2) + (late_k > 0 ? g.layers.size()*2 + 1 : 0);
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host + 6);
        rb_ids = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, max_batch*64);
        rb_x   = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_F32, max_batch*model.hparams.n_embd);
        rb_w   = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_F32, max_batch*64);
        fault_pred = getenv("LLAMA_MOE_FAULT_PRED") ? std::clamp(atoi(getenv("LLAMA_MOE_FAULT_PRED")), 0, 16) : 0;
#ifdef __linux__
        if (hybrid) {
            // over the RAM the graphs also predict the next layer's experts for the threads (pcbox, Qwen3.8 IQ3_S, 40 GiB of
            // host experts not in VRAM: 32 GB 16.5 -> 23.6 tok/s with 1 predicted expert; 64 GB under 10 faults per second, off)
            size_t host = 0;
            for (const device & d : devices) {
                host += d.host_bytes;
            }
            const double spill = host > size ? (double) (host - size) : 0.0;
            const size_t ram   = llama_moe_ram_limit();
            fault_auto = !getenv("LLAMA_MOE_FAULT_THREADS") &&
                (!getenv("LLAMA_MOE_FAULT_AUTO") || atoi(getenv("LLAMA_MOE_FAULT_AUTO")) != 0) &&
                ram > 0 && spill > 0.5*ram;
            // store mode also when VRAM + RAM may hold every expert but the page cache cannot (all host experts over 85% of
            // the RAM: it would keep copies of the VRAM-cached ones too); fill_complement then places all of them at start
            // (pcbox1 32 GB + RTX 5090, Qwen3.8 IQ3_S: 24 GB not in VRAM of 30.7 GiB RAM was below the spill rule)
            // not when only part fits: pcbox1 128 GB + RTX 5090, DS4.1 IQ2_XS (114.6 GiB not in VRAM, arena 110 GiB) store
            // 30.2 / 30.0 tok/s vs mmap 33.1 / 32.0; experts over the RAM (DS4.1 64 GB, Qwen3.8 32 GB + 16 GB card): store wins
            size_t cma_s = 0, rsv_s = 0;
            const bool fits = (double) host > 0.85*ram && spill <= 0.97*(double) arena_auto_room(cma_s, rsv_s);
            store_auto = ram > 0 && (spill > ram || fits) && !getenv("LLAMA_MOE_ARENA_MIB") &&
                (!getenv("LLAMA_MOE_STORE_AUTO") || atoi(getenv("LLAMA_MOE_STORE_AUTO")) != 0);
            if (fault_auto && !getenv("LLAMA_MOE_FAULT_PRED") && spill > ram) {
                fault_pred      = 1;
                fault_pred_auto = true;
            }
            if (fault_auto) {
                LLAMA_LOG_INFO("%s: %.1f GiB of host experts not in VRAM, %.1f GiB of RAM: fault threads switched by the page fault rate%s\n",
                    __func__, spill/1073741824.0, ram/1073741824.0, fault_pred_auto ? ", 1 predicted expert per layer" : "");
            }
        }
        if (const char * c = getenv("LLAMA_MOE_COLD")) {
            cold_on = hybrid && atoi(c) != 0;
        }
        if (cold_on) {
            LLAMA_LOG_INFO("%s: uploaded experts are marked cold in the page cache (LLAMA_MOE_COLD=1)\n", __func__);
        }
#endif
        if ((getenv("LLAMA_MOE_LOOKA") && atoi(getenv("LLAMA_MOE_LOOKA")) > 0) || (getenv("LLAMA_MOE_PREREAD") && atoi(getenv("LLAMA_MOE_PREREAD")) > 0) || fault_pred > 0) {
            rb_la = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, max_batch*64); // la_k <= 64
        }
        if (pf) {
            pf_out = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_F32, pf_max*n_embd);
            pf_x   = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_F32, pf_max*n_embd);
            pf_w   = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_F32, pf_max*n_expert_used);
        }
        ctx_views = init_ctx(2);

        // pinned if the device has a host buffer type: an upload from pageable memory waits for the stream (the expert
        // uploads before it), then the host cannot launch the graph of the split while the experts are uploaded
        ggml_backend_buffer_type_t buft_host = nullptr;
        for (const device & d : devices) {
            if (d.host_bytes > 0 && buft_host == nullptr && getenv("LLAMA_MOE_PAGEABLE") == nullptr) {
                buft_host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(d.backend));
            }
        }
        if (buft_host == nullptr) {
            buft_host = ggml_backend_cpu_buffer_type();
        }
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        doorbell = getenv("LLAMA_MOE_DOORBELL") && atoi(getenv("LLAMA_MOE_DOORBELL")) != 0 && buft_host != ggml_backend_cpu_buffer_type() &&
            !no_alloc;
        size_t db_bytes = 0;
        if (doorbell) {
            size_t n_banks = 0;
            for (const group & g : groups) {
                n_banks = std::max(n_banks, g.ref.size());
            }
            db_max_fill = db_max_tok*n_expert_used*(int64_t) n_banks;
            const int64_t n_expert_max = model.hparams.n_expert;
            GGML_UNUSED(n_expert_max);
            db_bytes = GGML_PAD(sizeof(ggml_moe_db), 64) + GGML_PAD(db_max_tok*n_expert_used*sizeof(int32_t), 64) +
                GGML_PAD(db_max_tok*n_embd*sizeof(float), 64) + GGML_PAD(db_max_tok*n_expert_used*sizeof(float), 64) +
                GGML_PAD(db_max_tok*n_expert_used*sizeof(int32_t), 64) + 8 + GGML_PAD(db_max_fill*sizeof(ggml_moe_db_fill), 64);
            if (late_k > 0) {
                // the late section: n_late, pad, the late uploads, late slots and ranks [8*K] each
                db_bytes += 8 + db_max_fill*sizeof(ggml_moe_db_fill) + 2*db_max_tok*late_k*sizeof(int32_t);
            }
        }

        shadow_gpu_mode = getenv("LLAMA_MOE_SHADOW") && atoi(getenv("LLAMA_MOE_SHADOW")) == 2;
        fetch_mode = getenv("LLAMA_MOE_GPU_FETCH") && atoi(getenv("LLAMA_MOE_GPU_FETCH")) != 0 && devices.size() == 1 && !shadow_gpu_mode;
        if (fetch_mode) {
            if (const char * s = getenv("LLAMA_MOE_FETCH_SHARE")) {
                fetch_share = std::clamp(atof(s), 0.0, 1.0);
            }
            int32_t n_cached = 0, n_exp = 0;
            for (const group & g : groups) {
                if (g.n_slots > 0) {
                    n_cached += (int32_t) g.layers.size();
                    n_exp = std::max<int32_t>(n_exp, (int32_t) g.ref[0]->ne[2]);
                }
            }
            if (n_cached > 0) {
                fmap_all = ggml_new_tensor_2d(devices[0].ctx.get(), GGML_TYPE_I32, n_exp, n_cached);
                ggml_format_name(fmap_all, "moe_cache.fmap");
            } else {
                fetch_mode = false;
            }
        }
        int32_t fmap_row = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            ggml_context * bctx = ctx;  // slot lending: the banks and their views in their own context (a VMM buffer)
            if (flex_bytes > 0) {
                if (!bank_ctx) {
                    size_t nb = 0;
                    for (const group & gg : groups) {
                        nb += gg.ref.size()*(1 + gg.layers.size());
                    }
                    ggml_init_params bp = { nb*ggml_tensor_overhead() + 1024, nullptr, true };
                    bank_ctx.reset(ggml_init(bp));
                }
                bctx = bank_ctx.get();
            }
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(bctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_extra);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    // hybrid mode: the zero slot (index n_slots) is part of the view
                    ggml_tensor * cached = ggml_view_3d(bctx, bank, bank->ne[0], bank->ne[1], g.n_slots + (hybrid ? 1 : 0), bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                if (fetch_mode && g.id == 0) {
                    l.fmap_dev = ggml_view_2d(ctx, fmap_all, 1, g.ref[0]->ne[2], sizeof(int32_t), (size_t) fmap_row*fmap_all->nb[1]);
                    ggml_format_name(l.fmap_dev, "moe_cache.fmap-%d", il);
                    l.fmap_row = fmap_row++;
                }
                if (shadow_gpu_mode && hybrid) {
                    l.smap_dev = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.smap_dev, "moe_cache.smap_dev-%d", il);
                }
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (doorbell) {
                    l.db = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, db_bytes/sizeof(int32_t));
                    ggml_format_name(l.db, "moe_cache.db-%d", il);
                    db_layers.push_back(il);
                }
                if (hybrid) {
                    const llama_layer & ml = model.layers[il];
                    l.up      = ml.ffn_up_exps;
                    l.gate    = ml.ffn_gate_exps;
                    l.down    = ml.ffn_down_exps;
                    l.gate_up = ml.ffn_gate_up_exps;
                    l.b_up      = ml.ffn_up_exps_b;
                    l.b_gate    = ml.ffn_gate_exps_b;
                    l.b_down    = ml.ffn_down_exps_b;
                    l.b_gate_up = ml.ffn_gate_up_exps_b;
                    for (const ggml_tensor * b : { l.b_up, l.b_gate, l.b_down, l.b_gate_up }) {
                        l.bias_ok = l.bias_ok && (b == nullptr || (b->type == GGML_TYPE_F32 && ggml_is_contiguous(b) &&
                            b->buffer && ggml_backend_buffer_is_host(b->buffer)));
                    }
                    if (!l.bias_ok) {
                        LLAMA_LOG_WARN("%s: hybrid mode: layer %d's expert biases are not F32 in host memory; its missed experts are all uploaded\n",
                            __func__, il);
                    }
                    l.cpu_out = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_F32, n_embd, max_batch);
                    ggml_format_name(l.cpu_out, "moe_cache.cpu_out-%d", il);
                    layer_of_out[l.cpu_out] = il;
                    l.map_dev.resize(g.ref[0]->ne[2]);
                    l.map_pin = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, g.ref[0]->ne[2]);
                    ggml_format_name(l.map_pin, "moe_cache.map_pin-%d", il);
                    l.flag = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, 4);
                    ggml_format_name(l.flag, "moe_cache.flag-%d", il);
                    l.pf_flag = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, 4);
                    ggml_format_name(l.pf_flag, "moe_cache.pf_flag-%d", il);
                    if (late_k > 0) {
                        ggml_context * dc = devices[g.id].ctx.get();
                        l.l_slots     = ggml_new_tensor_1d(dc, GGML_TYPE_I32, late_k*8);
                        l.l_w         = ggml_new_tensor_1d(dc, GGML_TYPE_F32, late_k*8);
                        l.l_flag      = ggml_new_tensor_1d(dc, GGML_TYPE_I32, 4);
                        l.l_slots_pin = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, late_k*8);
                        l.l_w_pin     = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_F32, late_k*8);
                    }
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }
        if (late_k > 0 && hybrid) {
            lt_zero.assign(devices.size(), nullptr);
            for (size_t id = 0; id < devices.size(); ++id) {
                if (devices[id].ctx) {
                    lt_zero[id] = ggml_new_tensor_1d(devices[id].ctx.get(), GGML_TYPE_F32, n_embd*8);
                }
            }
            lt_ring = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, 256);
        }

        if (no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                // with slot lending the banks live in their own buffer: this context may then hold nothing to allocate
                bool any = false;
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr && !any; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    any = t->data == nullptr && t->view_src == nullptr;
                }
                if (!any) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            if (bank_ctx) {
                lend_alloc(vmm_chunk);
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
            LLAMA_LOG_INFO("%s: slot maps in %s\n", __func__, ggml_backend_buft_name(buft_host));

            // flag mode: the device reads cpu_out and the flags from pinned memory
            flag_mode = hybrid && buft_host != ggml_backend_cpu_buffer_type() && getenv("LLAMA_MOE_CPU_COPY") == nullptr;
            for (layer & l : layers) {
                if (l.flag != nullptr) {
                    memset(l.flag->data, 0, ggml_nbytes(l.flag));
                    memset(l.pf_flag->data, 0, ggml_nbytes(l.pf_flag));
                }
            }
            pf = pf && flag_mode;
            if (pf) {
                LLAMA_LOG_INFO("%s: hybrid prefill share: ubatches of 33 to %d tokens, the CPU computes %s of the missed experts used by at most %d tokens\n",
                    __func__, (int) pf_max, pf_fixed < 0.0 ? "a measured share" : (std::to_string(pf_fixed).substr(0, 4) + " of the uploads").c_str(), pf_maxt);
            }
            doorbell = doorbell && (!hybrid || flag_mode);
            if (late_k > 0) {
                // the late pass needs the hybrid layer state (expert input, router weights) and every miss uploaded
                const bool ok = hybrid && flag_mode && !up_order && lt_ring != nullptr && devices.size() == 1;
                if (ok) {
                    ggml_backend_dev_t dev = ggml_backend_get_device(devices[0].backend);
                    lt_backend = ggml_backend_dev_init(dev, nullptr);
                }
                if (!ok || !lt_backend) {
                    LLAMA_LOG_WARN("%s: LLAMA_MOE_LATE needs the hybrid flag mode (LLAMA_MOE_CPU) on one device without upload order: off\n", __func__);
                    late_k = 0;
                } else {
                    LLAMA_LOG_INFO("%s: late pass: the missed experts are uploaded on a second stream while the cached ones are computed, then a pass of %d slots per token\n", __func__, late_k);
                }
            }
            if (doorbell) {
                for (int32_t il : db_layers) {
                    layer & l = layers[il];
                    memset(l.db->data, 0, ggml_nbytes(l.db));
                    ggml_moe_db * h = (ggml_moe_db *) l.db->data;
                    size_t off = GGML_PAD(sizeof(ggml_moe_db), 64);
                    h->ids_off  = off; off += GGML_PAD(db_max_tok*n_expert_used*sizeof(int32_t), 64);
                    h->x_off    = off; off += GGML_PAD(db_max_tok*n_embd*sizeof(float), 64);
                    h->w_off    = off; off += GGML_PAD(db_max_tok*n_expert_used*sizeof(float), 64);
                    h->ans_off   = off;
                    h->slots_cap = GGML_PAD(db_max_tok*n_expert_used, 2);
                    h->max_fill  = db_max_fill;
                    h->pad[0]    = late_k;
                    GGML_ASSERT(off + h->slots_cap*sizeof(int32_t) + 8 + db_max_fill*sizeof(ggml_moe_db_fill) +
                        (late_k > 0 ? 8 + db_max_fill*sizeof(ggml_moe_db_fill) + 2*db_max_tok*late_k*sizeof(int32_t) : 0) <= ggml_nbytes(l.db));
                }
                db_thread = std::thread([this] { db_loop(); });
                LLAMA_LOG_INFO("%s: doorbell mode: ubatches of up to %d tokens look up their experts without a split per layer\n", __func__, (int) db_max_tok);
                if (const char * da = getenv("LLAMA_MOE_DB_ASYNC"); da && atoi(da) > 0 && hybrid && flag_mode) {
                    ggml_backend_dev_t dev = ggml_backend_get_device(devices[0].backend);
                    as_backend = ggml_backend_dev_init(dev, nullptr);
                    if (as_backend) {
                        db_async = atoi(da);
                        as_pend.resize(groups.size());
                        for (size_t ig = 0; ig < groups.size(); ++ig) {
                            as_pend[ig].assign((size_t) groups[ig].lru.n_expert*layers.size(), 0);
                        }
                        as_thread = std::thread([this] { as_loop(); });
                        LLAMA_LOG_INFO("%s: doorbell background admission: the misses go to the CPU, up to %d of each layer's are copied in on a second stream\n", __func__, db_async);
                    } else {
                        LLAMA_LOG_WARN("%s: doorbell background admission: no second backend for the device, off\n", __func__);
                    }
                }
            }
            if (hybrid) {
                LLAMA_LOG_INFO("%s: hybrid: the CPU's output is %s\n", __func__, flag_mode ? "read by the GPU after a flag (flag mode)" : "copied as a host weight");
            }
#ifdef __linux__
            willneed = getenv("LLAMA_MOE_WILLNEED") && atoi(getenv("LLAMA_MOE_WILLNEED")) > 0;
            if (willneed) {
                LLAMA_LOG_INFO("%s: missed experts are prefetched with madvise(MADV_WILLNEED) right after each layer's plan\n", __func__);
            }
#endif

                if (fault_auto && (!flag_mode || doorbell)) {
                    fault_auto = false;
                    if (fault_pred_auto) {
                        fault_pred      = 0;
                        fault_pred_auto = false;
                    }
                }
                const char * a = getenv("LLAMA_MOE_LOOKA");
                preread =getenv("LLAMA_MOE_PREREAD") && atoi(getenv("LLAMA_MOE_PREREAD")) > 0 && use_kernel;
                if (((a && atoi(a) > 0) || preread || fault_pred > 0) && hybrid && devices.size() == 1 && !doorbell && rb_la) {
                la_budget = a ? std::max(0, atoi(a)) : 0;
                la_k = la_budget > 0 ? 8 : 12;
                if (const char * k = getenv("LLAMA_MOE_LOOKA_K")) {
                    la_k = std::clamp(atoi(k), 1, 64);
                }
                la_pred_on = true;
                la_gating = model.hparams.expert_gating_func;
                la_gi.assign(layers.size(), nullptr);
                la_bi.assign(layers.size(), nullptr);
                size_t n_pred = 0;
                for (size_t il = 0; il < layers.size(); ++il) {
                    const llama_layer & ml = model.layers[il];
                    if (layers[il].ig < 0 || ml.ffn_gate_inp == nullptr || ml.ffn_gate_tid2eid != nullptr || ml.ffn_gate_inp->ne[0] != n_embd) {
                        continue;
                    }
                    la_gi[il] = ml.ffn_gate_inp;
                    la_bi[il] = ml.ffn_exp_probs_b;
                    n_pred++;
                }
                if (preread) {
                    LLAMA_LOG_INFO("%s: pre-read: the CPU pool reads each token's top %d predicted, uncached experts of the next layer between jobs (%zu layers)\n",
                        __func__, la_k, n_pred);
                }
                if (la_budget > 0) {
                la_mark.resize(groups.size());
                la_pend.resize(groups.size());
                for (size_t ig = 0; ig < groups.size(); ++ig) {
                    la_mark[ig].assign(groups[ig].n_slots, 0);
                    la_pend[ig].assign(groups[ig].n_slots, 0);
                }
                la_lazy = !getenv("LLAMA_MOE_LOOKA_LAZY") || atoi(getenv("LLAMA_MOE_LOOKA_LAZY")) != 0;
                ggml_backend_dev_t dev = ggml_backend_get_device(devices[0].backend);
                la_backend = ggml_backend_dev_init(dev, nullptr);
                la_event   = la_backend ? ggml_backend_event_new(dev) : nullptr;
                if (!la_backend || !la_event) {
                    LLAMA_LOG_WARN("%s: lookahead: no second backend or event for the device, off\n", __func__);
                    if (la_backend) { ggml_backend_free(la_backend); la_backend = nullptr; }
                } else {
                    la_thread = std::thread([this] { la_loop(); });
                    la_all = getenv("LLAMA_MOE_LOOKA_ALL") && atoi(getenv("LLAMA_MOE_LOOKA_ALL")) != 0;
                    LLAMA_LOG_INFO("%s: lookahead prefetch: up to %d experts per layer from each token's top %d, %zu layers predicted\n",
                        __func__, la_budget, la_k, n_pred);
                }
                }
            }
            const char * am = getenv("LLAMA_MOE_ARENA_MIB") ? getenv("LLAMA_MOE_ARENA_MIB") : store_auto ? "auto" : nullptr;
            const char * ft = getenv("LLAMA_MOE_FAULT_THREADS");
            // LLAMA_MOE_ARENA_MIB=auto: store mode, the arena is the experts' RAM copy (see moe_arena::drop_page_cache)
            const bool am_auto = am && strcmp(am, "auto") == 0;
            const bool am_on   = am && (atoll(am) > 0 || am_auto);
            if (store_auto && am_on && hybrid && flag_mode && !doorbell && !la_backend) {
                LLAMA_LOG_INFO("%s: store mode: the experts do not fit in the page cache beside the VRAM cache; the arena holds their RAM copy (LLAMA_MOE_STORE_AUTO=0: off)\n", __func__);
            }
            fault_auto = fault_auto && !am_on && !la_backend;
            const int n_ft = ft ? atoi(ft) : fault_auto ? 16 : 0;
            if ((am_on || n_ft > 0) && hybrid && flag_mode && !doorbell && !la_backend) {
                const bool touch = !am_on;
                std::vector<std::vector<ggml_tensor *>> ex(layers.size());
                int32_t n_exp = 0;
                for (size_t il = 0; il < layers.size(); ++il) {
                    if (layers[il].ig >= 0) {
                        ex[il]  = layers[il].experts;
                        n_exp   = (int32_t) layers[il].slot_map->ne[1];
                    }
                }
                arena = std::make_unique<moe_arena>();
                arena->touch_only = touch;
                arena->no_wait    = touch && (fault_auto || (getenv("LLAMA_MOE_FAULT_NOWAIT") && atoi(getenv("LLAMA_MOE_FAULT_NOWAIT")) != 0));
                arena->chunk_size = touch ? 128 << 10 : 1 << 20;
                if (const char * c = getenv("LLAMA_MOE_ARENA_CHUNK_KIB")) {
                    arena->chunk_size = std::max<size_t>(4, (size_t) atoll(c)) << 10;
                }
                const int n_rd = touch ? n_ft : getenv("LLAMA_MOE_ARENA_THREADS") ? std::max(0, atoi(getenv("LLAMA_MOE_ARENA_THREADS"))) : am_auto ? 8 : 4;
                if (touch && arena->init(ex, n_exp)) {
                    arena->start(nullptr, 0, n_rd);
                    LLAMA_LOG_INFO("%s: fault threads: %d threads%s touch each layer's missed experts in %zu KiB pieces%s\n",
                        __func__, n_rd, arena->no_wait ? " (no wait)" : " and the caller", arena->chunk_size >> 10,
                        fault_auto ? ", while there are many page faults" : "");
                } else if (!touch && arena->init(ex, n_exp)) {
                    size_t size = (size_t) std::max(0LL, atoll(am)) << 20;
                    if (am_auto) {
                        // the RAM the kernel could give after dropping the experts' pages, less a reserve (the server's prompt
                        // cache, host compute buffers, the OS); never more than all the experts
                        const size_t n_drop = arena->drop_page_cache();
                        size_t cma = 0, rsv = 0;
                        const size_t room = arena_auto_room(cma, rsv);
                        const size_t all = arena->all_bytes();
                        size = std::min(all, room);
                        LLAMA_LOG_INFO("%s: arena auto: %.1f GiB of expert pages dropped from the page cache, %.1f GiB available (CMA %.1f GiB not counted), reserve %.1f GiB, all experts %.1f GiB\n",
                            __func__, n_drop/1073741824.0, (room + rsv)/1073741824.0, cma/1073741824.0, rsv/1073741824.0, all/1073741824.0);
                    }
                    const int64_t t0 = ggml_time_us();
                    arena_buf.reset(ggml_backend_buft_alloc_buffer(buft_host, size + moe_arena::align));
                    if (arena_buf) {
                        uint8_t * mem = (uint8_t *) GGML_PAD((uintptr_t) ggml_backend_buffer_get_base(arena_buf.get()), moe_arena::align);
                        arena->start(mem, size, n_rd);
                        arena_store = am_auto;
                        arena->async_reads = am_auto && (!getenv("LLAMA_MOE_ARENA_ASYNC") || atoi(getenv("LLAMA_MOE_ARENA_ASYNC")) != 0);
                        arena_excl = am_auto && (!getenv("LLAMA_MOE_ARENA_EXCL") || atoi(getenv("LLAMA_MOE_ARENA_EXCL")) != 0);
                        if (arena_excl && devices.size() == 1 && getenv("LLAMA_MOE_ARENA_EARLY") && atoi(getenv("LLAMA_MOE_ARENA_EARLY")) != 0) {
                            wb_backend = ggml_backend_dev_init(ggml_backend_get_device(devices[0].backend), nullptr);
                            if (wb_backend && getenv("LLAMA_MOE_ARENA_EARLY_EV") && atoi(getenv("LLAMA_MOE_ARENA_EARLY_EV")) != 0) {
                                ggml_backend_dev_t dv = ggml_backend_get_device(devices[0].backend);
                                wb_ev = ggml_backend_event_new(dv);
                                wbup_ev = ggml_backend_event_new(dv);
                                wb_ev_mode = wb_ev != nullptr && wbup_ev != nullptr;
                            }
                        }
                        LLAMA_LOG_INFO("%s: arena: %d experts in %zu pools (slots of up to %.2f MiB) in %zu MiB of %s (%.1f s), O_DIRECT reads of %zu KiB by %d readers and the caller\n",
                            __func__, arena->n_slots, arena->pool_size.size(), arena->slot_size/1048576.0, size >> 20, ggml_backend_buft_name(buft_host),
                            (ggml_time_us() - t0)/1e6, arena->chunk_size >> 10, n_rd);
                    } else {
                        LLAMA_LOG_WARN("%s: arena: cannot allocate %zu MiB, no arena\n", __func__, size >> 20);
                        arena.reset();
                    }
                } else {
                    arena.reset();
                }
            }
        if (hybrid && share_rule) {
                probe_link();
            }
            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        if (fetch_mode && !no_alloc) {
            fetch_init();
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        if (bank_buf) {
            ggml_backend_buffer_set_usage(bank_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }

        if (!no_alloc) {
            for (const layer & l : layers) {
                if (l.ig >= 0) {
                    hot_n_expert = (int32_t) l.slot_map->ne[1];
                }
            }
            if (const char * hs = getenv("LLAMA_MOE_HOT_SAVE"); hs && *hs && hot_n_expert > 0) {
                hot_save = hs;
                hot_count.assign(layers.size()*hot_n_expert, 0);
                for (group & g : groups) {
                    g.lru.hot = &hot_count;
                }
            }
            if (const char * hl = getenv("LLAMA_MOE_HOT_LIST"); hl && *hl && hot_n_expert > 0) {
                if (!getenv("LLAMA_MOE_HOT_VRAM") || atoi(getenv("LLAMA_MOE_HOT_VRAM")) != 0) {
                    fill_hot(hl);
                }
                if (arena_store) {
                    // store mode: the list's hottest experts read into the arena, not the page cache
                    std::ifstream f(hl);
                    std::vector<std::pair<int32_t, int32_t>> list;
                    for (long long il, e, n; f >> il >> e >> n; ) {
                        if (il >= 0 && il < (long long) layers.size() && layers[il].ig >= 0 && e >= 0 && e < layers[il].slot_map->ne[1]) {
                            list.push_back({ (int32_t) il, (int32_t) e });
                        }
                    }
                    const int64_t t0 = ggml_time_us();
                    const size_t n_rd = arena->preload(list);
                    const double dt = (ggml_time_us() - t0)/1e6;
                    LLAMA_LOG_INFO("llama_moe_cache: arena: %zu experts of the hot list read in %.1f s (%.2f GB/s)\n",
                        n_rd, dt, arena->bytes/1e9/std::max(dt, 1e-3));
                    arena->n_reads = arena->n_hits = arena->n_calls = 0;
                    arena->bytes = arena->t_us = 0.0;
                } else if (getenv("LLAMA_MOE_HOT_SYNC") && atoi(getenv("LLAMA_MOE_HOT_SYNC")) != 0) {
                    warm_hot(hl, true);
                } else {
                    hot_thread = std::thread([this, p = std::string(hl)] { warm_hot(p.c_str(), false); });
                }
            }
            if (arena_store) {
                fill_complement();
                arena->drop_page_cache();
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // store mode: a prompt's experts of layer il that the VRAM cache does not hold, read into the arena (cold end) before
    // the copies and the CPU's prompt job read them; called once per layer and ubatch, after the backend's sync
    void prefill_arena(const ggml_tensor * w, const std::vector<bool> & used) {
        pf_arena_us = 0;
        if (!arena_store || !arena) {
            return;
        }
        wb_sync();
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return;
        }
        const int32_t il = it->second.il;
        const layer & l = layers[il];
        if (l.ig < 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mtx);
        const int32_t * slots = groups[l.ig].lru.slot_map[il];
        arena_need.clear();
        for (int32_t e = 0; e < (int32_t) used.size(); ++e) {
            if (used[e] && (slots == nullptr || slots[e] < 0)) {
                arena_need.push_back(e);
            }
        }
        const int64_t t0 = ggml_time_us();
        arena->ensure_cold(il, arena_need.data(), arena_need.size());
        pf_arena_us = ggml_time_us() - t0;
    }

    // store mode: expert e of w in the arena, nullptr: read it from the mapping
    const uint8_t * arena_expert(const ggml_tensor * w, int64_t e) const {
        if (!arena_store || !arena) {
            return nullptr;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return nullptr;
        }
        const int32_t il = it->second.il;
        const layer & l = layers[il];
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            if (l.experts[ip] == w) {
                return arena->get_ready(il, (int32_t) ip, (int32_t) e);
            }
        }
        return nullptr;
    }

    ~impl() {
        if (hot_thread.joinable()) {
            hot_stop.store(true);
            hot_thread.join();
        }
        if (!hot_save.empty()) {
            save_hot();
        }
        if (stats_f) {
            stats_t = 0;
            stats_tick();
            fclose(stats_f);
        }
        if (la_thread.joinable()) {
            la_wait_idle();
            la_job.store(-2);
            la_thread.join();
        }
        if (fh_thread.joinable()) {
            fetch_wait_helper();
            fh_job.store(-1, std::memory_order_release);
            fh_thread.join();
        }
        if (fb_backend) {
            ggml_backend_synchronize(fb_backend);
        }
        for (auto & r : fb_rounds) {
            fb_free_ev.push_back(r.ev);
        }
        for (ggml_backend_event_t e : fb_free_ev) {
            ggml_backend_event_free(e);
        }
        if (fe_event) {
            ggml_backend_event_free(fe_event);
        }
        if (up_ev) {
            ggml_backend_event_free(up_ev);
        }
        if (fb_event) {
            ggml_backend_event_free(fb_event);
        }
        if (fb_backend) {
            ggml_backend_free(fb_backend);
        }
        if (la_backend) {
            ggml_backend_synchronize(la_backend);
        }
        if (la_event) {
            ggml_backend_event_free(la_event);
        }
        if (la_backend) {
            ggml_backend_free(la_backend);
        }
        if (lt_backend) {
            ggml_backend_synchronize(lt_backend);
            ggml_backend_free(lt_backend);
            lt_backend = nullptr;
        }
        if (wb_backend) {
            ggml_backend_synchronize(wb_backend);
            if (wb_ev) { ggml_backend_event_free(wb_ev); wb_ev = nullptr; }
            if (wbup_ev) { ggml_backend_event_free(wbup_ev); wbup_ev = nullptr; }
            ggml_backend_free(wb_backend);
            wb_backend = nullptr;
        }
        if (as_thread.joinable()) {
            {
                std::lock_guard<std::mutex> lk(as_mtx);
                as_exit = true;
            }
            as_cv.notify_all();
            as_thread.join();
        }
        if (as_backend) {
            ggml_backend_synchronize(as_backend);
            ggml_backend_free(as_backend);
            as_backend = nullptr;
        }
        if (db_thread.joinable()) {
            db_exit.store(true);
            db_thread.join();
        }
        if (worker.joinable()) {
            wait_cpu();
            job_layer.store(-2);
            worker.join();
        }
        pool_exit.store(true);
        for (auto & h : helpers) {
            h.join();
        }
        log_stats();
    }

    // LLAMA_MOE_CPU_PIN=1: the worker and the helpers are pinned one per physical core, interleaved over the L3 domains (CCDs),
    // so that a chiplet CPU's per-CCD memory links are used evenly; =<list> pins to the given CPUs in order (e.g. 0,8,1,9)
    std::vector<int> pin_cpus;
    static std::vector<int> pin_list() {
        std::vector<int> res;
#ifdef __linux__
        const char * e = getenv("LLAMA_MOE_CPU_PIN");
        if (e == nullptr || strcmp(e, "0") == 0) {
            return res;
        }
        if (strchr(e, ',') != nullptr) {
            for (const char * p = e; *p; ) {
                res.push_back(atoi(p));
                p = strchr(p, ',');
                if (p == nullptr) break;
                p++;
            }
            return res;
        }
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) != 0) {
            return res;
        }
        auto read = [](const std::string & path) { std::ifstream f(path); std::string v; std::getline(f, v); return v; };
        auto first = [](const std::string & list) { return atoi(list.c_str()); };
        std::map<std::string, std::vector<int>> groups; // L3 domain -> first sibling of each core
        std::set<int> seen;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &set)) continue;
            const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c);
            const int core = first(read(base + "/topology/thread_siblings_list"));
            if (seen.count(core)) continue;
            seen.insert(core);
            groups[read(base + "/cache/index3/shared_cpu_list")].push_back(core);
        }
        for (size_t i = 0; ; ++i) {
            bool any = false;
            for (auto & g : groups) {
                if (i < g.second.size()) { res.push_back(g.second[i]); any = true; }
            }
            if (!any) break;
        }
#endif
        return res;
    }
    void pin_self(int ith) {
#ifdef __linux__
        if (ith < (int) pin_cpus.size()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(pin_cpus[ith], &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
#endif
    }

    // a job still scoring is cancelled (a late prefetch would only delay the layer it was for); one already
    // giving out slots is waited for, that part is short
    void la_wait_idle() {
        if (!la_busy.load(std::memory_order_acquire)) {
            return;
        }
        const int64_t t0 = ggml_time_us();
        la_cancel.store(true, std::memory_order_release);
        while (la_busy.load(std::memory_order_acquire)) {
        }
        la_cancel.store(false, std::memory_order_relaxed);
        t_la_wait_us += ggml_time_us() - t0;
    }

    static float la_dot(const float * a, const float * b, int64_t n) {
        static const auto fn = la_dot_select();
        return fn(a, b, n);
    }

    // before the compute stream touches the banks again: it waits for the prefetch uploads (called with mtx held, after la_wait_idle)
    void la_join(ggml_backend_t backend) {
        if (la_pending) {
            ggml_backend_event_wait(backend, la_event);
            la_pending = false;
            for (const auto & p : la_pend_list) {
                la_pend[p.first][p.second] = 0;
            }
            la_pend_list.clear();
            n_la_joins++;
        }
    }

    // lazy join: wait only if one of the slots is still being written
    template <typename F>
    void la_join_if(ggml_backend_t backend, int32_t ig, F slots_of) {
        if (!la_pending) {
            return;
        }
        if (!la_lazy) {
            la_join(backend);
            return;
        }
        bool hit = false;
        slots_of([&](int32_t s) { hit = hit || (s >= 0 && la_pend[ig][s]); });
        if (hit) {
            la_join(backend);
        }
    }

    void la_loop() {
        for (;;) {
            int32_t tl;
            idle_wait([&] { return (tl = la_job.load(std::memory_order_acquire)) != -1; });
            if (tl == -2) {
                return;
            }
            const int64_t t0 = ggml_time_us();
            const layer & lt = layers[tl];
            const int64_t n_expert = groups[lt.ig].lru.n_expert;
            const int64_t n_tok = (int64_t) la_ids.size()/la_k;
            // per expert: tokens that have it in their predicted top la_k, best rank
            std::vector<int32_t> cnt(n_expert, 0), best(n_expert, 1 << 30);
            for (int64_t t = 0; t < n_tok; ++t) {
                for (int64_t r = 0; r < la_k; ++r) {
                    const int32_t e = la_ids[t*la_k + r];
                    if (e < 0 || e >= n_expert) continue;
                    cnt[e]++;
                    best[e] = std::min<int32_t>(best[e], (int32_t) r);
                }
            }
            std::vector<int32_t> cand;
            for (int64_t e = 0; e < n_expert; ++e) {
                if (cnt[e] > 0) cand.push_back((int32_t) e);
            }
            std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return cnt[a] != cnt[b] ? cnt[a] > cnt[b] : best[a] < best[b]; });

            std::vector<moe_cache_lru::fill> fills;
            if (la_cancel.load(std::memory_order_acquire)) {
                n_la_cancelled++;
                cand.clear();
            }
            {
                std::lock_guard<std::mutex> lock(mtx);
                group & g = groups[lt.ig];
                moe_cache_lru & lru = g.lru;
                int32_t * slots = lru.slot_map[tl];
                for (int32_t e : cand) {
                    if ((int) fills.size() >= la_budget) break;
                    if (slots[e] >= 0) continue;
                    const int32_t s = lru.head;
                    // the slots of the layer that is running are not evicted
                    if (lru.key_of[s] >= 0 && lru.key_of[s] / lru.n_expert == la_src) {
                        n_la_skipped++;
                        break;
                    }
                    if (lru.key_of[s] >= 0) {
                        lru.slot_map[lru.key_of[s] / lru.n_expert][lru.key_of[s] % lru.n_expert] = -1;
                    }
                    lru.key_of[s] = tl*lru.n_expert + e;
                    slots[e] = s;
                    lru.touch(s);
                    la_mark[lt.ig][s] = tl*lru.n_expert + e + 1;
                    la_pend[lt.ig][s] = 1;
                    la_pend_list.push_back({ lt.ig, s });
                    fills.push_back({ e, s });
                }
                for (const auto & f : fills) {
                    for (size_t ip = 0; ip < lt.experts.size(); ++ip) {
                        const ggml_tensor * w = lt.experts[ip];
                        const size_t esz = w->nb[2];
                        ggml_backend_tensor_set_async(la_backend, g.banks[ip], (const uint8_t *) w->data + f.expert*esz, f.slot*esz, esz);
                    }
                }
                if (!fills.empty()) {
                    ggml_backend_event_record(la_event, la_backend);
                    la_pending = true;
                }
                n_la_issued += fills.size();
                n_la_jobs++;
                stats_small.bytes += 0; // uploads are counted in n_la_issued
            }
            t_la_us += ggml_time_us() - t0;
            la_job.store(-1, std::memory_order_relaxed);
            la_busy.store(false, std::memory_order_release);
        }
    }

    void worker_loop() {
        pin_self(0);
        for (;;) {
            int32_t il;
            idle_wait([&] { return (il = job_layer.load(std::memory_order_acquire)) != -1; });
            if (il == -2) {
                return;
            }
            compute_cpu(layers[il]);
            if (layers[il].pf_job) {
                post_pf_out(layers[il]);
            } else if (flag_mode) {
                const int64_t tq = ggml_time_us();
                post_cpu_out(layers[il]);
                t_ph[5] += ggml_time_us() - tq;
            }
            const bool pr = preread && pr_ready.exchange(false, std::memory_order_relaxed) && !layers[il].pf_job;
            if (pr) {
                pr_running.store(true, std::memory_order_release); // before job_done: the callback that sees it done waits for this
            }
            job_layer.store(-1, std::memory_order_relaxed);
            job_done.store(true, std::memory_order_release);
            if (pr) {
                run_preread();
                pr_running.store(false, std::memory_order_release);
            }
        }
    }

    // reads the pre-read chunks with the pool until done or a new job is posted
    void run_preread() {
        const int64_t t0 = ggml_time_us();
        const int64_t n = (int64_t) pr_chunks.size();
        std::atomic<bool> aborted{false};
        pool_run([&](int) {
            uint64_t acc = 0;
            for (int64_t i; (i = pool_next.fetch_add(1, std::memory_order_relaxed)) < n; ) {
                if (job_layer.load(std::memory_order_relaxed) != -1 || pr_stop.load(std::memory_order_relaxed)) {
                    aborted.store(true, std::memory_order_relaxed);
                    break;
                }
                const uint64_t * p = (const uint64_t *) pr_chunks[i].first;
                const size_t m = pr_chunks[i].second / 8;
                for (size_t j = 0; j < m; j += 8) {
                    acc += p[j];
                }
            }
            pr_sink = pr_sink + acc;
        });
        n_pr_jobs++;
        n_pr_aborted += aborted.load();
        pr_bytes += (double) std::min<int64_t>(pool_next.load(), n) * 65536.0;
        t_pr_us += ggml_time_us() - t0;
    }

    // after the spin phase of a waiting loop: yield during the first second without work (decode keeps the wake-up immediate),
    // then sleep on every pass (200 us, 1 ms after 10 s) so an idle server costs no CPU (issue #2: the old loop slept only one
    // pass in 1024 and yielded in the others, ~0.5 core per thread in sched_yield while idle)
    static void idle_pause(std::chrono::steady_clock::time_point t_idle) {
        const auto idle = std::chrono::steady_clock::now() - t_idle;
        if (idle < std::chrono::seconds(1)) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(idle < std::chrono::seconds(10) ? std::chrono::microseconds(200) : std::chrono::microseconds(1000));
        }
    }

    // spin until ready() (a new job), then idle_pause
    template <typename F>
    static void idle_wait(F ready) {
        int64_t spins = 0;
        auto t0 = std::chrono::steady_clock::now();
        while (!ready()) {
            if (++spins < 4096) {
                continue;
            }
            idle_pause(t0);
        }
    }

    void helper_loop(int ith) {
        pin_self(ith);
        uint64_t seen = 0;
        for (;;) {
            uint64_t e = 0;
            idle_wait([&] { return pool_exit.load(std::memory_order_relaxed) || (e = pool_epoch.load(std::memory_order_acquire)) != seen; });
            if (pool_exit.load()) {
                return;
            }
            seen = e;
            pool_fn(ith);
            pool_done.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    // runs fn(ith) on the worker (ith 0) and every helper, returns when all are done
    void pool_run(std::function<void(int)> fn) {
        pool_fn = std::move(fn);
        pool_next.store(0, std::memory_order_relaxed);
        pool_done.store(0, std::memory_order_relaxed);
        if (guest) {
            pool_active.store(true, std::memory_order_seq_cst);
        }
        pool_epoch.fetch_add(1, std::memory_order_acq_rel);
        pool_fn(0);
        const int n = (int) helpers.size();
        while (pool_done.load(std::memory_order_acquire) < n) {
        }
        if (guest) {
            // a guest that saw the phase active is still inside pool_fn: wait for it before the next phase reuses pool_fn
            pool_active.store(false, std::memory_order_seq_cst);
            while (guest_in.load(std::memory_order_seq_cst) != 0) {
            }
        }
    }

    // the calling thread works on the running CPU job's phases until the job is done (LLAMA_MOE_CPU_GUEST)
    void pool_help() {
        while (!job_done.load(std::memory_order_acquire)) {
            if (!pool_active.load(std::memory_order_seq_cst)) {
                continue;
            }
            guest_in.fetch_add(1, std::memory_order_seq_cst);
            if (pool_active.load(std::memory_order_seq_cst)) {
                pool_fn((int) helpers.size() + 1);
                n_guest_items++;
            }
            guest_in.fetch_sub(1, std::memory_order_seq_cst);
        }
    }

    void wait_cpu() {
        while (!job_done.load(std::memory_order_acquire)) {
        }
    }

    // the link speed for the share rule: uploads of 8 experts of the first cached layer into its first slots (empty at start)
    void probe_link() {
        for (const group & g : groups) {
            if (g.n_slots < 8 || g.layers.empty()) {
                continue;
            }
            const layer & l = layers[g.layers[0]];
            ggml_backend_t be = devices[g.id].backend;
            expert_bytes = 0.0;
            for (const ggml_tensor * t : l.experts) {
                expert_bytes += t->nb[2];
            }
            double best = 0.0;
            for (int rep = 0; rep < 4; ++rep) {
                const int64_t t0 = ggml_time_us();
                size_t bytes = 0;
                // store mode: the uploads read the arena (pinned), not the mapping (pageable after the page cache drop:
                // pcbox1 32 GB + 5090 probed 17 GB/s vs 42.6 from pinned memory, and the rule gave the CPU 75% of the misses)
                const uint8_t * src_arena = arena_store && arena && arena->base ? arena->base : nullptr;
                size_t off_arena = 0;
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    ggml_backend_tensor_set_async(be, g.banks[ip], src_arena ? src_arena + off_arena : (const uint8_t *) w->data, 0, 8*w->nb[2]);
                    off_arena += 8*w->nb[2];
                    bytes += 8*w->nb[2];
                }
                ggml_backend_synchronize(be);
                const double gbs = bytes/(double) std::max<int64_t>(1, ggml_time_us() - t0)/1e3;
                best = rep > 0 ? std::max(best, gbs) : 0.0; // the first one warms up
            }
            link_gbs = best;
            LLAMA_LOG_INFO("%s: hybrid share rule: link %.1f GB/s, share = L/(L + %.2f*R) from the CPU's measured rate\n", __func__, link_gbs, rule_k);
            return;
        }
    }

    // the layer's experts can be computed by the CPU (set_hybrid_layer's conditions apart from the graph's)
    bool hybrid_layer_ok(int32_t il) const {
        const layer & l = layers[il];
        return hybrid && l.cpu_out != nullptr && l.down != nullptr && (l.gate_up || (l.up && l.gate)) && cpu_can(l, l.act);
    }

    // the CPU path can compute the layer with this activation: biases and the gpt-oss activation only in the kernel
    bool cpu_can(const layer & l, llama_moe_cache::act_type act) const {
        const bool biased = l.b_up || l.b_gate || l.b_down || l.b_gate_up;
        return l.bias_ok && (use_kernel || (!biased && act != llama_moe_cache::ACT_SWIGLU_OAI));
    }

    ggml_tensor * doorbell_slots(ggml_context * ctx, int32_t il, ggml_tensor * ids, ggml_tensor * x, ggml_tensor * w, int64_t n_tokens, int64_t n_used) const {
        if (!doorbell || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0 || layers[il].db == nullptr) {
            return nullptr;
        }
        const layer & l = layers[il];
        const_cast<layer &>(l).db_out = nullptr;
        if (n_tokens < 1 || n_tokens > db_max_tok || n_used > n_expert_used ||
                std::min(n_tokens*n_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        if (x == nullptr || w == nullptr || !hybrid_layer_ok(il)) {
            x = w = nullptr;
        }
        ggml_tensor * db = ggml_moe_doorbell(ctx, ids, x, w, (ggml_moe_db *) l.db->data, (int) n_used);
        const_cast<layer &>(l).db_out = db;
        return db;
    }

    void db_loop() {
        int64_t spins = 0;
        auto t_idle = std::chrono::steady_clock::now();
        while (!db_exit.load(std::memory_order_relaxed)) {
            bool any = false;
            for (int32_t il : db_layers) {
                volatile ggml_moe_db * h = (volatile ggml_moe_db *) layers[il].db->data;
                if (h->req != h->ack) {
                    db_service(il);
                    any = true;
                }
            }
            if (any) {
                spins = 0;
                continue;
            }
            if (++spins < 4096) {
                continue;
            }
            if (spins == 4096) {
                t_idle = std::chrono::steady_clock::now();
            }
            idle_pause(t_idle);
        }
    }

    // background admission: claim victims for the layer's most-selected CPU experts (l.cpu_experts, most tokens first);
    // called with mtx held, after the answer. A victim is unmapped now (no later answer points at it) and published by
    // as_loop once its copy is done; the slots answered for this layer (in use by the GPU) are never taken
    void as_claim(int32_t il, const layer & l, group & g) {
        moe_cache_lru & lru = g.lru;
        const int32_t n_exp = lru.n_expert;
        std::vector<as_job> jobs;
        for (int32_t e : l.cpu_experts) {
            if ((int) jobs.size() >= db_async) break;
            const int32_t key = il*n_exp + e;
            if (as_pend[l.ig][key] || lru.slot_map[il][e] >= 0) continue;
            const int32_t s = lru.head;
            const int32_t vk = lru.key_of[s];
            if (vk >= 0 && vk / n_exp == il) break; // the running layer's slots stay
            if (vk >= 0) {
                lru.slot_map[vk / n_exp][vk % n_exp] = -1;
                lru.key_of[s] = -1;
            }
            lru.touch(s); // to the MRU end: not claimed again while in flight
            as_pend[l.ig][key] = 1;
            jobs.push_back({ l.ig, il, e, s });
        }
        if (!jobs.empty()) {
            n_as_claimed += jobs.size();
            {
                std::lock_guard<std::mutex> lk(as_mtx);
                as_queue.insert(as_queue.end(), jobs.begin(), jobs.end());
            }
            as_cv.notify_one();
        }
    }

    void as_loop() {
        std::vector<as_job> jobs;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(as_mtx);
                as_cv.wait(lk, [&] { return as_exit || !as_queue.empty(); });
                if (as_exit) {
                    return;
                }
                jobs.swap(as_queue);
                as_queue.clear();
            }
            const int64_t t0 = ggml_time_us();
            for (const as_job & j : jobs) {
                const layer & l = layers[j.il];
                group & g = groups[j.ig];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const size_t esz = w->nb[2];
                    ggml_backend_tensor_set_async(as_backend, g.banks[ip], (const uint8_t *) w->data + (size_t) j.e*esz, (size_t) j.s*esz, esz);
                }
            }
            ggml_backend_synchronize(as_backend);
            t_as_copy_us += ggml_time_us() - t0;
            std::lock_guard<std::mutex> lock(mtx);
            for (const as_job & j : jobs) {
                moe_cache_lru & lru = groups[j.ig].lru;
                as_pend[j.ig][j.il*lru.n_expert + j.e] = 0;
                // the slot may have been claimed again or the expert placed another way meanwhile: then drop the copy
                if (lru.key_of[j.s] != -1 || lru.slot_map[j.il][j.e] >= 0) {
                    n_as_stale++;
                    continue;
                }
                lru.key_of[j.s] = j.il*lru.n_expert + j.e;
                lru.slot_map[j.il][j.e] = j.s;
                lru.touch(j.s);
                n_as_published++;
            }
            jobs.clear();
        }
    }

    // a doorbell of layer il: the copy callback's work, with the ids (and x, w) read from the control block, then the answer
    void db_service(int32_t il) {
        std::lock_guard<std::mutex> lock(mtx);
        const int64_t t0 = ggml_time_us();
        std::atomic_thread_fence(std::memory_order_acquire);
        layer & l = layers[il];
        group & g = groups[l.ig];
        volatile ggml_moe_db * h = (volatile ggml_moe_db *) l.db->data;
        char * base = (char *) l.db->data;
        const int64_t n_tok  = h->n_tok;
        const int64_t n_used = h->n_used;
        const bool    hyb    = h->hyb && hybrid_layer_ok(il);
        const int32_t * h_ids = (const int32_t *) (base + h->ids_off);
        ids.assign(h_ids, h_ids + n_tok*n_used);

        if (hybrid_auto && !shadow) {
            gate_step(il, n_tok, hyb, t0);
        }
        if (hyb) {
            const float * h_x = (const float *) (base + h->x_off);
            const float * h_w = (const float *) (base + h->w_off);
            l.x.assign(h_x, h_x + n_tok*n_embd);
            l.w.assign(h_w, h_w + n_tok*n_used);
            volatile uint32_t * f = (volatile uint32_t *) l.flag->data;
            const double wait_us = f[2]/1000.0;
            t_flag_wait_us += wait_us;
            n_flag_waits++;
            if (hybrid_auto) {
                share_update(wait_us, wait_us < 2.0 ? 10.0 : 0.0);
            }
            f[2] = 0;
            n_flag_timeouts = std::max<size_t>(n_flag_timeouts, f[3]);
        }

        size_t n_hit = 0;
        if (hyb) {
            g.lru.plan_hybrid(il, ids.data(), ids.size(), db_async > 0 ? 0.0 : cur_share(), fills, l.cpu_experts, n_hit, n_expert_used, tail_rank, &tail_skipped);
            n_tail_skip += tail_skipped.size();
        } else if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        // late pass (the graph joins: the backend set n_fill = K): the late uploads are listed after the early ones and the
        // device copies them on a second stream while the first pass runs (it reads the zero slot for them)
        uint32_t * h_nfill = (uint32_t *) (base + h->ans_off + h->slots_cap*sizeof(int32_t));
        ggml_moe_db_fill * h_fill = (ggml_moe_db_fill *) (h_nfill + 2);
        uint32_t * h_nlate = (uint32_t *) (h_fill + h->max_fill);
        const bool late_on = late_k > 0 && hyb && h->n_fill == (uint32_t) late_k && l.late_graph && n_tok <= 8;
        fills_late.clear();
        if (late_k > 0 && h->n_fill == (uint32_t) late_k) {
            int32_t * h_ls = (int32_t *) ((ggml_moe_db_fill *) (h_nlate + 2) + h->max_fill);
            if (late_on) {
                late_split(l, g, h_ls, nullptr, h_ls + db_max_tok*late_k);
            } else {
                // the graph's late pass still runs: zero slots, nothing to upload
                std::fill(h_ls, h_ls + n_tok*late_k, g.n_slots);
                std::fill(h_ls + db_max_tok*late_k, h_ls + db_max_tok*late_k + n_tok*late_k, 0);
                h_nlate[0] = 0;
            }
        }

        int32_t * h_slots = (int32_t *) (base + h->ans_off);
        const int32_t * map = (const int32_t *) l.slot_map->data;
        const int64_t n_expert = l.slot_map->ne[1];
        if (hyb) {
            // the experts computed by the CPU read the zero slot on the device, so they add nothing there
            std::copy(map, map + n_expert, l.map_dev.begin());
            for (int32_t e : l.cpu_experts) {
                l.map_dev[e] = g.n_slots;
            }
            for (int32_t e : tail_skipped) {
                l.map_dev[e] = g.n_slots;
            }
            for (const auto & f : fills_late) {
                l.map_dev[f.expert] = g.n_slots;
            }
            for (size_t i = 0; i < ids.size(); ++i) {
                h_slots[i] = l.map_dev[ids[i]];
            }
            l.ids = ids;
            wait_cpu();
            l.pf_job = false;
            l.t_start = ggml_time_us();
            job_done.store(false, std::memory_order_relaxed);
            job_layer.store(il, std::memory_order_release);
            n_cpu_experts += l.cpu_experts.size();
        } else {
            for (size_t i = 0; i < ids.size(); ++i) {
                h_slots[i] = map[ids[i]];
            }
        }

        // the uploads, consecutive experts going to consecutive slots in one piece
        size_t bytes = 0;
        h_nfill[0] = (uint32_t) db_fill_list(l, g, fills, h_fill, bytes);
        if (late_on) {
            h_nlate[0] = (uint32_t) db_fill_list(l, g, fills_late, (ggml_moe_db_fill *) (h_nlate + 2), bytes);
        }

        t_db_gpu_wait_us += h->wait_ns/1000.0;
        n_db_timeouts = std::max<size_t>(n_db_timeouts, h->timeouts);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        h->ack = h->req;
        std::atomic_thread_fence(std::memory_order_seq_cst);

        stats & st = stats_small;
        st.hits   += n_hit;
        st.misses += fills.size() + fills_late.size() + (hyb ? l.cpu_experts.size() : 0);
        st.bytes  += bytes;
        if (db_async > 0 && hyb && !l.cpu_experts.empty()) {
            as_claim(il, l, g);
        }
        n_db_calls++;
        t_db_us += ggml_time_us() - t0;
        if (hyb) {
            n_hybrid_calls++;
            t_upload_us += ggml_time_us() - t0;
        }
    }

    // doorbell: the upload list of fills (consecutive experts going to consecutive slots in one piece), returns its length
    int64_t db_fill_list(const layer & l, const group & g, const std::vector<moe_cache_lru::fill> & fl, ggml_moe_db_fill * out, size_t & bytes) const {
        int64_t n_fill = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w = l.experts[ip];
            ggml_tensor * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fl.size();) {
                size_t n = 1;
                while (i + n < fl.size() && fl[i + n].expert == fl[i].expert + (int32_t) n && fl[i + n].slot == fl[i].slot + (int32_t) n) {
                    n++;
                }
                GGML_ASSERT(n_fill < db_max_fill);
                out[n_fill].src = (uint64_t) (uintptr_t) ((const uint8_t *) w->data + fl[i].expert*expert_size);
                out[n_fill].dst = (uint64_t) (uintptr_t) ((uint8_t *) bank->data + fl[i].slot*expert_size);
                out[n_fill].n   = n*expert_size;
                n_fill++;
                bytes += n*expert_size;
                i += n;
            }
        }
        return n_fill;
    }

    // late pass: moves the uploads in fills to fills_late unless a token would need more than late_k late slots (those stay
    // early), and writes the late slots and router weights of each token (zero slot padding) into the layer's pinned copies
    void late_split(layer & lh, group & g, int32_t * ps, float * pw, int32_t * pr = nullptr) {
        const int64_t n_tok = (int64_t) ids.size()/n_expert_used;
        late_slot.assign(g.lru.n_expert, -1);
        for (const auto & f : fills) {
            late_slot[f.expert] = f.slot;
        }
        for (int64_t t = 0; t < n_tok; ++t) {
            int c = 0;
            for (int64_t r = 0; r < n_expert_used; ++r) {
                const int32_t e = ids[t*n_expert_used + r];
                if (late_slot[e] >= 0) {
                    if (c < late_k) {
                        c++;
                    } else {
                        late_slot[e] = -1; // early for every token
                        n_late_early++;
                    }
                }
            }
        }
        for (int64_t t = 0; t < n_tok; ++t) {
            int c = 0;
            for (int64_t r = 0; r < n_expert_used; ++r) {
                const int32_t e = ids[t*n_expert_used + r];
                if (late_slot[e] >= 0) {
                    ps[t*late_k + c] = late_slot[e];
                    if (pr) {
                        pr[t*late_k + c] = (int32_t) r;
                    } else {
                        pw[t*late_k + c] = lh.w[t*n_expert_used + r];
                    }
                    c++;
                }
            }
            for (; c < late_k; ++c) {
                ps[t*late_k + c] = g.n_slots;
                if (pr) {
                    pr[t*late_k + c] = 0;
                } else {
                    pw[t*late_k + c] = 0.0f;
                }
            }
        }
        std::vector<moe_cache_lru::fill> early;
        for (const auto & f : fills) {
            (late_slot[f.expert] >= 0 ? fills_late : early).push_back(f);
        }
        fills.swap(early);
        n_late += fills_late.size();
        n_late_calls++;
    }

    // late pass: the late uploads on the second stream, then a new value of the layer's flag the barrier waits for, written
    // after them on the same stream
    size_t late_copy(int32_t il, layer & lh, group & g) {
        size_t bytes = 0;
        for (size_t ip = 0; ip < lh.experts.size(); ++ip) {
            const ggml_tensor * w    = lh.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (const auto & f : fills_late) {
                const uint8_t * p = arena ? arena->get_ready(il, (int32_t) ip, f.expert) : nullptr;
                ggml_backend_tensor_set_async(lt_backend, bank, p ? p : (const uint8_t *) w->data + f.expert*expert_size, f.slot*expert_size, expert_size);
                bytes += expert_size;
            }
        }
        int32_t * ring = (int32_t *) lt_ring->data;
        const uint32_t ri = lt_ring_i++ % 256;
        ring[ri] = (int32_t) ++lh.l_seq;
        ggml_backend_tensor_set_async(lt_backend, lh.l_flag, &ring[ri], 0, sizeof(int32_t));
        return bytes;
    }

    // flag mode: write the CPU's output where the GPU reads it, then release the GPU's wait
    void post_cpu_out(layer & l) {
        memcpy(l.cpu_out->data, l.out.data(), l.out.size()*sizeof(float));
        std::atomic_thread_fence(std::memory_order_release);
        volatile uint32_t * f = (volatile uint32_t *) l.flag->data;
        f[0] = f[0] + 1;
    }

    void post_pf_out(layer & l) {
        memcpy(pf_out->data, l.out.data(), l.out.size()*sizeof(float));
        std::atomic_thread_fence(std::memory_order_release);
        volatile uint32_t * f = (volatile uint32_t *) l.pf_flag->data;
        f[0] = f[0] + 1;
    }

    // lookahead: the router (and selection bias) of layer il+1, to be applied to layer il's expert input in the graph
    ggml_tensor * lookahead_router(int32_t il, ggml_tensor ** bias, int * k) const {
        if (!la_pred_on || il < 0 || il + 1 >= (int32_t) layers.size() || layers[il].ig < 0 || la_gi[il + 1] == nullptr ||
                layers[il + 1].ig != layers[il].ig) {
            return nullptr;
        }
        *bias = la_bi[il + 1];
        *k    = la_k;
        return la_gi[il + 1];
    }

    void set_lookahead_layer(int32_t il, ggml_tensor * pred) {
        if (il >= 0 && il < (int32_t) layers.size()) {
            layers[il].la_pred = pred;
        }
    }

    bool prefill_enabled(int64_t n_tokens) const {
        return pf && n_tokens > max_batch && n_tokens <= pf_max;
    }

    // a large-batch graph registers (or with x nullptr clears) the tensors of the prefill share of layer il
    ggml_tensor * set_prefill_layer(int32_t il, ggml_tensor * x, ggml_tensor * w, const ggml_tensor * ids, llama_moe_cache::act_type act, float limit) {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        layer & l = layers[il];
        if (!pf || x == nullptr || !hybrid_layer_ok(il) || act == llama_moe_cache::ACT_NONE || !cpu_can(l, act) || x->ne[1] > pf_max) {
            l.p_x = nullptr;
            return nullptr;
        }
        l.p_x   = x;
        l.p_w   = w;
        l.p_ids = ids;
        l.act   = act;
        l.limit = limit;
        return pf_out;
    }

    ggml_tensor * prefill_add(ggml_context * ctx, int32_t il, ggml_tensor * moe_out) {
        const layer & l = layers[il];
        if (!ggml_is_contiguous(moe_out)) {
            moe_out = ggml_cont(ctx, moe_out);
        }
        return ggml_host_add(ctx, moe_out, (const float *) pf_out->data, (uint32_t *) l.pf_flag->data);
    }

    // large-batch copy callback, once per ubatch and layer with the selected experts ids_data (as read, ids' strides) and the
    // experts to upload: picks the CPU's experts, takes them out of used, points their pairs on the device elsewhere, starts the job
    bool prefill_plan(ggml_backend_t backend, const ggml_tensor * w, const ggml_tensor * ids_t, std::vector<int32_t> & ids_data, std::vector<bool> & used) {
        if (!pf) {
            return false;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return false;
        }
        const int32_t il = it->second.il;
        layer & l = layers[il];
        if (l.p_x == nullptr) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mtx);
        stats_tick();
        const int64_t t0 = ggml_time_us();
        const int64_t n_used = ids_t->ne[0], n_tok = ids_t->ne[1];
        const int64_t n_expert = (int64_t) used.size();

        // the previous layer of this ubatch: its CPU and GPU times (the backend was synchronized before this callback)
        if (pf_prev_il >= 0 && pf_prev_il < il && pf_fixed < 0.0) {
            const layer & lp = layers[pf_prev_il];
            const double wait_us = ((volatile uint32_t *) lp.pf_flag->data)[2]/1000.0;
            const double c = (double) lp.t_job/pf_prev_ncpu;
            // store mode: this layer's arena reads (before this call) are not the GPU's time
            const double g = std::max(0.0, (double) (t0 - pf_prev_t0) - wait_us - (double) pf_arena_us)/std::max<int64_t>(pf_prev_ngpu, 1);
            constexpr double a = 0.25;
            pf_c_us = pf_measured ? pf_c_us + a*(c - pf_c_us) : c;
            pf_g_us = pf_measured ? pf_g_us + a*(g - pf_g_us) : g;
            pf_measured = true;
            t_pf_wait_us += wait_us;
        }
        pf_prev_il = -1;

        auto post_zero = [&]() {
            // the graph adds pf_out after the layer's experts in any case
            memset(pf_out->data, 0, n_tok*n_embd*sizeof(float));
            std::atomic_thread_fence(std::memory_order_release);
            volatile uint32_t * f = (volatile uint32_t *) l.pf_flag->data;
            f[0] = f[0] + 1;
        };
        if (l.p_ids != ids_t || l.p_x->ne[1] != n_tok || ggml_nelements(l.p_w) != n_used*n_tok) {
            if (n_pf_mismatch++ == 0) {
                LLAMA_LOG_WARN("llama_moe_cache: prefill share: layer %d's registered tensors do not match the graph, no CPU share\n", il);
            }
            wait_cpu();
            post_zero();
            return true;
        }
        wait_cpu();

        // the selected experts in token order
        l.ids.resize(n_tok*n_used);
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t k = 0; k < n_used; ++k) {
                l.ids[t*n_used + k] = ids_data[t*ids_t->nb[1]/sizeof(int32_t) + k*ids_t->nb[0]/sizeof(int32_t)];
            }
        }
        pf_cnt.assign(n_expert, 0);
        for (int32_t e : l.ids) {
            pf_cnt[e]++;
        }
        // the candidates: experts that would be uploaded (not in the cache), used by at most pf_maxt tokens, fewest first
        const group & g = groups[l.ig];
        const int32_t * slots = g.lru.slot_map[il];
        std::vector<std::pair<int32_t, int32_t>> cand;
        int64_t n_stream = 0;
        for (int32_t e = 0; e < n_expert; ++e) {
            if (pf_cnt[e] == 0 || (slots != nullptr && slots[e] >= 0)) {
                continue;
            }
            n_stream++;
            if (pf_cnt[e] <= pf_maxt) {
                cand.emplace_back(pf_cnt[e], e);
            }
        }
        std::sort(cand.begin(), cand.end());
        // the fault threads read the layer's missed experts (the GPU's copies and the CPU's) in parallel while the copy and
        // the CPU job fault on them: the copy alone reads the mapping at queue depth ~1 (pcbox1 DS4.1: 1.9 GB/s vs 5.7 the
        // drive does). LLAMA_MOE_PREFILL_TOUCH=0: off
        if (pf_touch && arena && arena->touch_only && arena->no_wait) {
            if (fault_auto) {
                fault_check();
            }
            if (!fault_auto || fault_on) {
                std::vector<std::pair<int32_t, std::vector<int32_t>>> lst(1);
                lst[0].first = il;
                for (int32_t e = 0; e < n_expert; ++e) {
                    if (pf_cnt[e] > 0 && (slots == nullptr || slots[e] < 0)) {
                        lst[0].second.push_back(e);
                    }
                }
                arena->touch_bulk(lst);
                n_pf_touch += lst[0].second.size();
            }
        }
        const double share = pf_fixed >= 0.0 ? pf_fixed : std::clamp(pf_g_us/(pf_c_us + pf_g_us), 0.05, 0.9);
        // the CPU's pairs are pointed at experts the GPU computes and their tokens do not use: keep enough of them
        int64_t n_used_e = 0;
        for (int32_t e = 0; e < n_expert; ++e) {
            n_used_e += pf_cnt[e] > 0;
        }
        const int64_t take = std::max<int64_t>(0, std::min<int64_t>({ (int64_t) cand.size(), std::llround(share*n_stream), n_used_e - 2*n_used }));
        n_pf_layers++;
        n_pf_stream += n_stream;
        if (take == 0) {
            post_zero();
            t_pf_plan_us += ggml_time_us() - t0;
            return true;
        }

        ggml_backend_tensor_get_async(backend, l.p_x, pf_x->data, 0, ggml_nbytes(l.p_x));
        ggml_backend_tensor_get_async(backend, l.p_w, pf_w->data, 0, ggml_nbytes(l.p_w));

        pf_on_cpu.assign(n_expert, 0);
        l.cpu_experts.clear();
        const int32_t zero = g.n_slots;
        std::fill(l.map_dev.begin(), l.map_dev.end(), -1);
        for (int64_t i = 0; i < take; ++i) {
            const int32_t e = cand[i].second;
            pf_on_cpu[e] = 1;
            used[e] = false;
            l.cpu_experts.push_back(e);
            l.map_dev[e] = zero;
        }
        // the experts the GPU computes, most used first: the CPU's pairs are pointed at the first one their token does not use
        std::vector<std::pair<int32_t, int32_t>> top;
        for (int32_t e = 0; e < n_expert; ++e) {
            if (pf_cnt[e] > 0 && !pf_on_cpu[e]) {
                top.emplace_back(-pf_cnt[e], e);
            }
        }
        std::sort(top.begin(), top.end());
        GGML_ASSERT((int64_t) top.size() >= 2*n_used && "prefill share: too few experts left on the GPU");
        ggml_backend_synchronize(backend);
        wb_sync();
        l.x.assign((const float *) pf_x->data, (const float *) pf_x->data + n_tok*n_embd);
        l.w.assign((const float *) pf_w->data, (const float *) pf_w->data + n_tok*n_used);
        float * wd = (float *) pf_w->data;
        size_t n_pairs = 0;
        std::vector<int32_t> row(n_used);
        for (int64_t t = 0; t < n_tok; ++t) {
            const int32_t * te = l.ids.data() + t*n_used;
            std::copy(te, te + n_used, row.begin());
            for (int64_t k = 0; k < n_used; ++k) {
                if (!pf_on_cpu[te[k]]) {
                    continue;
                }
                // not used by the token, nor taken by one of its earlier CPU pairs
                int32_t r = -1;
                for (const auto & p : top) {
                    if (std::find(row.begin(), row.end(), p.second) == row.end()) {
                        r = p.second;
                        break;
                    }
                }
                GGML_ASSERT(r >= 0);
                row[k] = r;
                ids_data[t*ids_t->nb[1]/sizeof(int32_t) + k*ids_t->nb[0]/sizeof(int32_t)] = r;
                wd[t*n_used + k] = 0.0f;
                n_pairs++;
            }
        }
        // the job: the worker reads l.ids, l.x, l.w, l.cpu_experts and l.map_dev
        l.pf_job  = true;
        l.t_start = ggml_time_us();
        job_done.store(false, std::memory_order_relaxed);
        job_layer.store(il, std::memory_order_release);

        ggml_backend_tensor_set_async(backend, const_cast<ggml_tensor *>(ids_t), ids_data.data(), 0, ggml_nbytes(ids_t));
        ggml_backend_tensor_set_async(backend, l.p_w, pf_w->data, 0, ggml_nbytes(l.p_w));
        // the ids are pageable: that copy returned once staged, and pf_w is not written again before the next layer's sync
        static const bool dbg = getenv("LLAMA_MOE_PREFILL_DEBUG") != nullptr;
        if (dbg) {
            std::vector<int32_t> chk(ids_data.size());
            ggml_backend_tensor_get(ids_t, chk.data(), 0, ggml_nbytes(ids_t));
            int bad = 0, dup = 0;
            for (int64_t t = 0; t < n_tok; ++t) {
                for (int64_t k = 0; k < n_used; ++k) {
                    const int32_t e = chk[t*ids_t->nb[1]/4 + k*ids_t->nb[0]/4];
                    bad += e < 0 || e >= n_expert;
                    for (int64_t j = 0; j < k; ++j) {
                        dup += chk[t*ids_t->nb[1]/4 + j*ids_t->nb[0]/4] == e;
                    }
                }
            }
            LLAMA_LOG_INFO("pf dbg: layer %d n_tok %d n_used %d nb %zu %zu nbytes %zu (data %zu) stream %d take %d pairs %zu bad %d dup %d same %d w %s\n",
                il, (int) n_tok, (int) n_used, ids_t->nb[0], ids_t->nb[1], ggml_nbytes(ids_t), ids_data.size()*4, (int) n_stream, (int) take, n_pairs, bad, dup,
                (int) (chk == ids_data), ggml_get_name(l.p_w));
        }

        pf_prev_il   = il;
        pf_prev_t0   = l.t_start;
        pf_prev_ncpu = take;
        pf_prev_ngpu = n_stream - take;
        n_pf_cpu    += take;
        n_pf_pairs  += n_pairs;
        t_pf_plan_us += ggml_time_us() - t0;
        return true;
    }

    ggml_tensor * hybrid_add(ggml_context * ctx, int32_t il, ggml_tensor * moe_out) {
        const layer & l = layers[il];
        if (flag_mode) {
            if (!ggml_is_contiguous(moe_out)) {
                moe_out = ggml_cont(ctx, moe_out);
            }
            return ggml_host_add(ctx, moe_out, (const float *) l.cpu_out->data, (uint32_t *) l.flag->data);
        }
        ggml_tensor * cpu_out = ggml_view_2d(ctx, l.cpu_out, moe_out->ne[0], moe_out->ne[1], l.cpu_out->nb[1], 0);
        return ggml_add(ctx, moe_out, cpu_out);
    }

    // LLAMA_MOE_WILLNEED: ask the kernel to read the missed experts' pages now, all of them at once
    void willneed_experts(const std::vector<ggml_tensor *> & experts, const std::vector<moe_cache_lru::fill> & fills, const std::vector<int32_t> & cpu) {
#ifdef __linux__
        const int64_t t0 = ggml_time_us();
        static const uintptr_t page = (uintptr_t) sysconf(_SC_PAGE_SIZE);
        auto one = [&](int32_t e) {
            for (const ggml_tensor * w : experts) {
                const uintptr_t b = (uintptr_t) w->data + (uintptr_t) e*w->nb[2];
                const uintptr_t a = b & ~(page - 1);
                madvise((void *) a, b + w->nb[2] - a, MADV_WILLNEED);
                n_wn_ranges++;
            }
        };
        for (int32_t e : cpu) {
            one(e);
        }
        for (const moe_cache_lru::fill & f : fills) {
            one(f.expert);
        }
        n_wn_calls++;
        t_wn_us += ggml_time_us() - t0;
#else
        GGML_UNUSED(experts); GGML_UNUSED(fills); GGML_UNUSED(cpu);
#endif
    }

    // auto share: the CPU was late by wait us (the GPU or host waited for it), or done early by idle us, per layer call
    void share_update(double wait_us, double idle_us) {
        ctl_wait_us += wait_us;
        ctl_idle_us += idle_us;
        if (++ctl_calls < ctl_window) {
            return;
        }
        const double wait = ctl_wait_us/ctl_calls, idle = ctl_idle_us/ctl_calls;
        const double old = hybrid_share;
        if (share_rule && link_gbs > 0.0) {
            const double dt = t_cpu_us - rule_t_cpu;
            const size_t dn = n_cpu_experts - rule_n_cpu;
            rule_t_cpu = t_cpu_us;
            rule_n_cpu = n_cpu_experts;
            ctl_wait_us = ctl_idle_us = 0.0;
            ctl_calls = 0;
            if (dn < 64 || dt <= 0.0) {
                return;
            }
            const double R = expert_bytes*dn/dt/1e3; // GB/s
            const double target = std::clamp(std::round(link_gbs/(link_gbs + rule_k*R)*20.0)/20.0, 0.10, 0.60);
            // a step at a time toward the target
            hybrid_share = target > hybrid_share + 0.025 ? hybrid_share + 0.05 : target < hybrid_share - 0.025 ? hybrid_share - 0.05 : hybrid_share;
            if (hybrid_share != old) {
                n_share_changes++;
                LLAMA_LOG_DEBUG("llama_moe_cache: hybrid share %.2f -> %.2f (rule: link %.1f GB/s, CPU %.1f GB/s)\n", old, hybrid_share, link_gbs, R);
            }
            rule_R_last = R;
            return;
        }
        if (wait > idle + 5.0) {
            hybrid_share = std::min(0.95, hybrid_share + 0.05);
        } else if (idle > wait + 5.0) {
            hybrid_share = std::max(0.05, hybrid_share - 0.05); // keep admitting a few, so the cache follows the conversation
        }
        if (hybrid_share != old) {
            n_share_changes++;
            LLAMA_LOG_DEBUG("llama_moe_cache: hybrid share %.2f -> %.2f (CPU late %.1f us, idle %.1f us per call)\n", old, hybrid_share, wait, idle);
        }
        ctl_wait_us = ctl_idle_us = 0.0;
        ctl_calls = 0;
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    // GPU fetch: register every cache view with the address of its layer's experts in host memory (they must be registered
    // host memory, LLAMA_MOE_PIN_EXPS), a second stream for the admissions
    void fetch_init() {
        const char * why = nullptr;
        ggml_backend_dev_t dev = ggml_backend_get_device(devices[0].backend);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto fn = reg ? (bool (*)(const ggml_tensor *, const void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_fetch_register") : nullptr;
        auto fmax = reg ? (int (*)(int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_fetch_max_batch") : nullptr;
        fetch_max = 8;
        if (fmax != nullptr) {
            for (const auto & [w, b] : bindings) {
                fetch_max = std::min<int64_t>(fetch_max, fmax((int) w->type));
            }
        }
        if (fn == nullptr || fmax == nullptr) {
            why = "the device's backend has no in-place expert reads";
        } else if (fetch_max < 1) {
            why = "an expert type without MMVQ";
        } else if (arena && !arena->touch_only) {
            why = "the experts' RAM copy is an arena (store mode)";
        }
        for (const auto & [w, b] : bindings) {
            if (why) {
                break;
            }
            const layer & l = layers[b.il];
            for (const ggml_tensor * t : { l.b_up, l.b_gate, l.b_down, l.b_gate_up }) {
                if (t != nullptr) {
                    why = "expert biases";
                }
            }
            if (!why && !fn(b.cached, w->data)) {
                why = "the experts are not in registered host memory (LLAMA_MOE_PIN_EXPS)";
            }
        }
        if (!why) {
            fb_backend = ggml_backend_dev_init(dev, nullptr);
            fb_event   = fb_backend ? ggml_backend_event_new(dev) : nullptr;
            if (!fb_backend || !fb_event) {
                why = "no second stream";
            }
        }
        if (why) {
            LLAMA_LOG_WARN("%s: GPU fetch: off (%s)\n", __func__, why);
            return;
        }
        fb_pend.resize(groups.size());
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            fb_pend[ig].assign(groups[ig].n_slots, 0);
        }
        // pinned staging: one async upload of the maps, one sync for every layer's selected experts
        {
            const size_t n_map = ggml_nelements(fmap_all);
            const size_t n_ids = (size_t) layers.size()*fetch_max*n_expert_used;
            ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(dev);
            fb_pin_buf.reset(ggml_backend_buft_alloc_buffer(hbt ? hbt : ggml_backend_cpu_buffer_type(), (n_map + n_ids)*sizeof(int32_t) + 64));
            if (!fb_pin_buf) {
                LLAMA_LOG_WARN("%s: GPU fetch: off (no staging buffer)\n", __func__);
                return;
            }
            fb_map_pin = (int32_t *) ggml_backend_buffer_get_base(fb_pin_buf.get());
            fb_ids_pin = fb_map_pin + n_map;
            fb_ids_off.assign(layers.size(), SIZE_MAX);
            fb_ids_n.assign(layers.size(), 0);
        }
        fe_event = ggml_backend_event_new(dev);
        if (!fe_event) {
            LLAMA_LOG_WARN("%s: GPU fetch: off (no event)\n", __func__);
            return;
        }
        fh_thread = std::thread([this] { fetch_loop(); });
        fetch = true;
        LLAMA_LOG_INFO("%s: GPU fetch: decode ubatches of up to %lld tokens read uncached experts in place from host memory, %.0f%% of the misses admitted between ubatches\n",
            __func__, (long long) fetch_max, 100.0*fetch_share);
    }

    ggml_tensor * fetch_slot_map(int32_t il, int64_t n_tokens) const {
        return fetch && n_tokens <= fetch_max && il >= 0 && il < (int32_t) layers.size() && layers[il].ig >= 0 ? layers[il].fmap_dev : nullptr;
    }

    // before each ubatch: the previous decode graph's experts -> LRU and admissions; then the device maps for this one
    // admissions are published two rounds after they are queued (round r's copies run while rounds r and r+1 compute and
    // draft), so a graph never waits for copies queued just before it; a round's slots keep reading in place until then
    void fetch_publish(int keep) {
        ggml_backend_t be = devices[0].backend;
        while ((int) fb_rounds.size() > keep) {
            fb_round & r = fb_rounds.front();
            ggml_backend_event_wait(be, r.ev);
            for (const auto & p : r.slots) {
                fb_pend[p.first][p.second]--;
            }
            fb_free_ev.push_back(r.ev);
            fb_rounds.pop_front();
        }
    }

    void fetch_wait_helper() {
        while (fh_busy.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    // before each ubatch (after the helper is done with the previous decode graph): publish, then this ubatch's maps
    void fetch_begin(int64_t n_tokens) {
        if (!fetch) {
            return;
        }
        const int64_t t0 = ggml_time_us();
        fetch_wait_helper();
        std::lock_guard<std::mutex> lock(mtx);
        ggml_backend_t be = devices[0].backend;
        if (n_tokens > fetch_max) {
            // the usual path reads the banks: every admission first
            fetch_publish(0);
            fetch_armed = false;
            t_fetch_us += ggml_time_us() - t0;
            return;
        }
        fetch_publish(1);
        // this decode ubatch's maps: cached and not being copied -> the slot, else read in place
        const int64_t n_exp = fmap_all->ne[0];
        for (int32_t il = 0; il < (int32_t) layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.ig < 0 || l.fmap_dev == nullptr) {
                continue;
            }
            const int32_t * sm = groups[l.ig].lru.slot_map[il];
            const std::vector<uint8_t> & pend = fb_pend[l.ig];
            int32_t * m = fb_map_pin + (size_t) l.fmap_row*n_exp;
            for (int64_t e = 0; e < ggml_nelements(l.fmap_dev); ++e) {
                const int32_t s = sm[e];
                m[e] = s >= 0 && !pend[s] ? s : -(int32_t) (e + 1);
            }
        }
        // the previous upload from the staging buffer is done: a whole graph ran after it, and the helper synchronized since
        ggml_backend_tensor_set_async(be, fmap_all, fb_map_pin, 0, ggml_nbytes(fmap_all));
        fetch_armed = true;
        n_fetch_ubatches++;
        t_fetch_us += ggml_time_us() - t0;
    }

    // after a decode graph is launched: queue the reads of its selected experts behind it, the helper does the rest
    void fetch_end() {
        if (!fetch || !fetch_armed) {
            return;
        }
        fetch_armed = false;
        const int64_t t0 = ggml_time_us();
        ggml_backend_t be = devices[0].backend;
        const size_t cap = (size_t) fetch_max*n_expert_used;
        for (int32_t il = 0; il < (int32_t) layers.size(); ++il) {
            const layer & l = layers[il];
            fb_ids_off[il] = SIZE_MAX;
            fb_ids_n[il] = 0;
            if (l.ig < 0 || l.fetch_ids == nullptr || l.fetch_ids->buffer == nullptr || (size_t) ggml_nelements(l.fetch_ids) > cap) {
                continue;
            }
            fb_ids_off[il] = (size_t) il*cap;
            fb_ids_n[il]   = ggml_nelements(l.fetch_ids);
            ggml_backend_tensor_get_async(be, l.fetch_ids, fb_ids_pin + fb_ids_off[il], 0, ggml_nbytes(l.fetch_ids));
        }
        ggml_backend_event_record(fe_event, be);
        t_fetch_end_us += ggml_time_us() - t0;
        fh_busy.store(true, std::memory_order_relaxed);
        fh_job.store(1, std::memory_order_release);
    }

    // helper thread: the graph's experts -> LRU, admissions queued on the second stream
    void fetch_loop() {
        for (;;) {
            int j;
            idle_wait([&] { return (j = fh_job.load(std::memory_order_acquire)) != 0; });
            if (j < 0) {
                return;
            }
            const int64_t t0 = ggml_time_us();
            ggml_backend_event_synchronize(fe_event);
            {
                std::lock_guard<std::mutex> lock(mtx);
                fb_round r;
                for (int32_t il = 0; il < (int32_t) layers.size(); ++il) {
                    layer & l = layers[il];
                    if (fb_ids_off[il] == SIZE_MAX) {
                        continue;
                    }
                    group & g = groups[l.ig];
                    size_t n_hit = 0;
                    g.lru.plan_hybrid(il, fb_ids_pin + fb_ids_off[il], fb_ids_n[il], fetch_share, fb_fills, fb_rest, n_hit);
                    stats_small.hits   += n_hit;
                    stats_small.misses += fb_fills.size() + fb_rest.size();
                    n_fetch_inplace    += fb_fills.size() + fb_rest.size();
                    for (const auto & f : fb_fills) {
                        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                            const ggml_tensor * w = l.experts[ip];
                            const size_t esz = w->nb[2];
                            ggml_backend_tensor_set_async(fb_backend, g.banks[ip], (const uint8_t *) w->data + f.expert*esz, f.slot*esz, esz);
                            stats_small.bytes += esz;
                        }
                        fb_pend[l.ig][f.slot]++;
                        r.slots.push_back({ l.ig, f.slot });
                    }
                }
                if (!r.slots.empty()) {
                    if (fb_free_ev.empty()) {
                        fb_free_ev.push_back(ggml_backend_event_new(ggml_backend_get_device(fb_backend)));
                    }
                    r.ev = fb_free_ev.back();
                    fb_free_ev.pop_back();
                    ggml_backend_event_record(r.ev, fb_backend);
                    n_fetch_admitted += r.slots.size();
                    fb_rounds.push_back(std::move(r));
                }
            }
            t_fetch_helper_us += ggml_time_us() - t0;
            fh_job.store(0, std::memory_order_relaxed);
            fh_busy.store(false, std::memory_order_release);
        }
    }

    ggml_tensor * shadow_slot_map(int32_t il) const {
        return shadow_gpu && il >= 0 && il < (int32_t) layers.size() && layers[il].ig >= 0 ? layers[il].smap_dev : nullptr;
    }

    // GPU-only shadow: point the device slot maps at the cache as it is (uncached experts -> the zero slot)
    void shadow_gpu_begin() {
        std::vector<int32_t> m;
        for (layer & l : layers) {
            l.shadow_ids = nullptr;
            if (l.ig < 0 || l.smap_dev == nullptr) {
                continue;
            }
            const int32_t * hs = (const int32_t *) l.slot_map->data;
            m.assign(hs, hs + ggml_nelements(l.slot_map));
            for (int32_t & v : m) {
                v = v >= 0 ? v : (int32_t) groups[l.ig].n_slots;
            }
            ggml_backend_tensor_set(l.smap_dev, m.data(), 0, m.size()*sizeof(int32_t));
        }
    }

    // after the GPU-only shadow pass: read the selected experts, and read ahead the uncached ones that are not in RAM
    void shadow_gpu_end() {
        std::vector<int32_t> ids, drop;
        std::vector<std::pair<int32_t, std::vector<int32_t>>> lst;
        for (int32_t il = 0; il < (int32_t) layers.size(); ++il) {
            layer & l = layers[il];
            if (l.ig < 0 || l.shadow_ids == nullptr) {
                continue;
            }
            ids.resize(ggml_nelements(l.shadow_ids));
            ggml_backend_tensor_get(l.shadow_ids, ids.data(), 0, ids.size()*sizeof(int32_t));
            l.shadow_ids = nullptr;
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            const int32_t * hs = (const int32_t *) l.slot_map->data;
            drop.clear();
            for (int32_t e : ids) {
                if (e < 0 || e >= ggml_nelements(l.slot_map) || hs[e] >= 0) {
                    continue;
                }
                if (expert_resident(l, e)) {
                    n_shadow_ram++;
                } else {
                    drop.push_back(e);
                }
            }
            n_shadow_dropped += drop.size();
            if (!drop.empty()) {
                lst.emplace_back(il, drop);
            }
        }
        arena->touch_bulk(lst);
    }

    // the banks in one VMM buffer, each at a chunk-aligned offset; the largest group lends its last slots
    void lend_alloc(size_t chunk) {
        ggml_backend_dev_t dev = ggml_backend_get_device(devices[0].backend);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto falloc = (ggml_backend_buffer_t (*)(ggml_backend_buffer_type_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_vmm_buffer_alloc");
        ggml_backend_buffer_type_t buft = devices[0].buft;
        // the lending group: the largest banks
        size_t best = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            size_t gb = 0;
            for (const ggml_tensor * bank : groups[ig].banks) {
                gb += ggml_nbytes(bank);
            }
            if (groups[ig].n_slots > 0 && gb > best) {
                best = gb;
                lend_ig = (int32_t) ig;
            }
        }
        // the other banks packed first (128-byte aligned), then the lending group's, each at a chunk-aligned offset
        std::vector<std::pair<ggml_tensor *, size_t>> place;
        size_t off = 0;
        const size_t align = ggml_backend_buft_get_alignment(buft);
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                if (((int32_t) ig == lend_ig) != (pass == 1)) {
                    continue;
                }
                for (ggml_tensor * bank : groups[ig].banks) {
                    off = pass == 1 ? GGML_PAD(off, chunk) : GGML_PAD(off, align);
                    place.push_back({ bank, off });
                    off += ggml_backend_buft_get_alloc_size(buft, bank);
                }
            }
        }
        off = GGML_PAD(off, chunk);
        bank_buf.reset(falloc ? falloc(buft, off) : nullptr);
        if (!bank_buf) {
            LLAMA_LOG_ERROR("%s: slot lending: could not allocate %.0f MiB of VMM for the banks\n", __func__, off/1048576.0);
            throw std::runtime_error("failed to allocate the MoE cache banks (VMM)");
        }
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(bank_buf.get());
        for (auto & [t, o] : place) {
            GGML_ASSERT(ggml_backend_tensor_alloc(bank_buf.get(), t, base + o) == GGML_STATUS_SUCCESS);
        }
        for (ggml_tensor * t = ggml_get_first_tensor(bank_ctx.get()); t != nullptr; t = ggml_get_next_tensor(bank_ctx.get(), t)) {
            if (t->view_src != nullptr) {
                GGML_ASSERT(ggml_backend_view_init(t) == GGML_STATUS_SUCCESS);
            }
        }
        ggml_backend_buffer_clear(bank_buf.get(), 0);
        {
            size_t raw = 0;
            for (const group & gg : groups) {
                for (const ggml_tensor * bank : gg.banks) {
                    raw += ggml_nbytes(bank);
                }
            }
            LLAMA_LOG_INFO("%s: slot lending: %zu groups' banks in a VMM buffer of %.0f MiB (%.0f MiB of banks)\n", __func__,
                groups.size(), off/1048576.0, raw/1048576.0);
        }
        if (lend_ig < 0) {
            return;
        }
        group & g = groups[lend_ig];
        size_t es = 0;
        for (const ggml_tensor * t : g.ref) {
            es += t->nb[2];
        }
        const int32_t n_lend = (int32_t) std::min<size_t>((flex_bytes + g.ref.size()*2*chunk + es - 1)/es, g.n_slots);
        lend_keep = g.n_slots - n_lend;
        if (lend_keep < n_expert_used*8) {
            LLAMA_LOG_WARN("%s: slot lending: the cache is too small to lend %zu MiB, off\n", __func__, flex_bytes >> 20);
            lend_ig = -1;
            return;
        }
        lend_off.clear();
        for (ggml_tensor * bank : g.banks) {
            lend_off.push_back((size_t) ((uint8_t *) bank->data - base));
        }
        size_t covered = 0;
        for (size_t ip = 0; ip < g.banks.size(); ++ip) {
            const size_t a = lend_off[ip] + (size_t) lend_keep*g.banks[ip]->nb[2];
            const size_t b = lend_off[ip] + (size_t) g.n_slots*g.banks[ip]->nb[2];
            const size_t a1 = GGML_PAD(a, chunk), b1 = b/chunk*chunk;
            covered += b1 > a1 ? b1 - a1 : 0;
        }
        lend_bytes = covered;
        LLAMA_LOG_INFO("%s: slot lending: group %d lends slots %d-%d of %d (%.0f MiB of whole %.0f MiB chunks) to prompt batches' compute buffers\n",
            __func__, lend_ig, lend_keep, g.n_slots - 1, g.n_slots, covered/1048576.0, chunk/1048576.0);
    }

    void flex_release() {
        if (lend_ig < 0 || lend_out) {
            return;
        }
        const int64_t t0 = ggml_time_us();
        wb_sync();
        group & g = groups[lend_ig];
        ggml_backend_t be = devices[g.id].backend;
        ggml_backend_synchronize(be);
        std::lock_guard<std::mutex> lock(mtx);
        uint8_t * dst[16];
        size_t n_back = 0, n_ev = 0;
        for (int32_t s = lend_keep; s < g.n_slots; ++s) {
            const int32_t k = g.lru.key_of[s];
            if (k >= 0) {
                const int32_t vl = k / g.lru.n_expert, ve = k % g.lru.n_expert;
                // exclusive store mode: an expert in a VRAM slot has no RAM copy; copy it back (else it is read from the SSD later)
                if (arena && arena_excl && layers[vl].experts.size() <= 16 && arena->get(vl, 0, ve) == nullptr && arena->claim_wb(vl, ve, dst)) {
                    for (size_t ip = 0; ip < layers[vl].experts.size(); ++ip) {
                        ggml_tensor * bank = g.banks[ip];
                        const size_t es  = layers[vl].experts[ip]->nb[2];
                        const size_t off = (size_t) s*es;
                        ggml_backend_tensor_get_async(be, bank, dst[ip], off, std::min(es + moe_arena::pad_after, ggml_nbytes(bank) - off));
                    }
                    n_back++;
                }
                if (g.lru.slot_map[vl] != nullptr) {
                    g.lru.slot_map[vl][ve] = -1;
                }
                g.lru.key_of[s] = -1;
                n_ev++;
            }
            g.lru.unlink(s);
        }
        ggml_backend_synchronize(be);
        for (size_t ip = 0; ip < g.banks.size(); ++ip) {
            const size_t es = g.banks[ip]->nb[2];
            vmm_remap(bank_buf.get(), lend_off[ip] + (size_t) lend_keep*es, (size_t) (g.n_slots - lend_keep)*es, false);
        }
        lend_out = true;
        n_flex_release++;
        n_flex_back += n_back;
        n_flex_evicted += n_ev;
        flex_us += ggml_time_us() - t0;
    }

    void flex_restore() {
        if (lend_ig < 0 || !lend_out) {
            return;
        }
        const int64_t t0 = ggml_time_us();
        std::lock_guard<std::mutex> lock(mtx);
        group & g = groups[lend_ig];
        {
            // the prompt batches' temporaries grew the backend's memory pool: give that memory back first
            ggml_backend_dev_t dev = ggml_backend_get_device(devices[g.id].backend);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            auto ftrim = (void (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_pool_trim");
            if (ftrim) {
                ftrim(devices[g.id].backend);
            }
        }
        for (size_t ip = 0; ip < g.banks.size(); ++ip) {
            const size_t es = g.banks[ip]->nb[2];
            const size_t off = (size_t) lend_keep*es, sz = (size_t) (g.n_slots - lend_keep)*es;
            const size_t want = sz >= 2*vmm_chunk_size() ? 1 : 0;
            if (vmm_remap(bank_buf.get(), lend_off[ip] + off, sz, true) == 0 && want) {
                throw std::runtime_error("MoE cache: the lent slots could not be mapped again after a prompt batch");
            }
            ggml_backend_tensor_memset(g.banks[ip], 0, off, sz);  // the MMQ padding past an expert reads the next slot
        }
        for (int32_t s = g.n_slots - 1; s >= lend_keep; --s) {
            g.lru.push_head(s);
        }
        lend_out = false;
        flex_us += ggml_time_us() - t0;
    }

    size_t vmm_chunk_size() const {
        ggml_backend_dev_t dev = ggml_backend_get_device(devices[0].backend);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto fchunk = (size_t (*)(ggml_backend_buffer_type_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_vmm_chunk");
        return fchunk ? fchunk(devices[0].buft) : 0;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last, int64_t e0, const void ** src_out = nullptr) {
        hot_busy_us.store(ggml_time_us(), std::memory_order_relaxed);
        if (la_backend) {
            la_wait_idle();
            std::lock_guard<std::mutex> lock(mtx);
            la_join(backend);
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        if (src_out != nullptr) {
            // the caller copies them (one gather kernel for the whole range)
            *src_out = (const uint8_t *) bank->data + (size_t) slots[e]*bank->nb[2];
            stats_copy.hits  += n;
            stats_copy.bytes += (size_t) n*bank->nb[2];
            return n;
        }
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  (e - e0)*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];
        if (la_backend) {
            la_wait_idle();
        }
        std::lock_guard<std::mutex> lock(mtx);

        GGML_ASSERT(backend == devices[g.id].backend);
        stats_tick();

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        // read back into pinned memory: a read back into pageable memory is a slow staged copy
        GGML_ASSERT(ggml_nelements(sel) <= ggml_nelements(rb_ids));
        // hybrid mode: also read the expert input and the router weights, the CPU computes the missed experts that are not uploaded
        layer & lh = layers[il];
        const bool hyb = hybrid && lh.x_dev != nullptr;
        {
            // LLAMA_MOE_RB_PACK (default 1): one backend kernel writes all the read backs into the pinned buffers instead
            // of one D2H copy each (CUDA/HIP: ggml_backend_get_tensors_packed); 0: the copies
            typedef bool (*get_packed_t)(ggml_backend_t, int, const ggml_tensor * const *, void * const *, const size_t *);
            static const bool rb_pack = getenv("LLAMA_MOE_RB_PACK") == nullptr || atoi(getenv("LLAMA_MOE_RB_PACK")) != 0;
            static get_packed_t get_packed = nullptr;
            static bool get_packed_init = false;
            if (rb_pack && !get_packed_init) {
                get_packed_init = true;
                ggml_backend_dev_t dev = ggml_backend_get_device(backend);
                ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
                get_packed = reg ? (get_packed_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_tensors_packed") : nullptr;
                LLAMA_LOG_INFO("%s: packed routing read back: %s\n", __func__, get_packed ? "on" : "not supported by the backend");
            }
            const ggml_tensor * rb_src[4]; void * rb_dst[4]; size_t rb_n[4]; int n_rb = 0;
            rb_src[n_rb] = sel; rb_dst[n_rb] = rb_ids->data; rb_n[n_rb++] = ggml_nbytes(sel);
            if (hyb) {
                GGML_ASSERT(ggml_nelements(lh.w_dev) == ggml_nelements(sel));
                GGML_ASSERT(ggml_nelements(lh.x_dev) <= ggml_nelements(rb_x) && ggml_nelements(lh.w_dev) <= ggml_nelements(rb_w));
                rb_src[n_rb] = lh.x_dev; rb_dst[n_rb] = rb_x->data; rb_n[n_rb++] = ggml_nbytes(lh.x_dev);
                rb_src[n_rb] = lh.w_dev; rb_dst[n_rb] = rb_w->data; rb_n[n_rb++] = ggml_nbytes(lh.w_dev);
                if (la_pred_on && lh.la_pred != nullptr) {
                    GGML_ASSERT(ggml_nelements(lh.la_pred) <= ggml_nelements(rb_la));
                    rb_src[n_rb] = lh.la_pred; rb_dst[n_rb] = rb_la->data; rb_n[n_rb++] = ggml_nbytes(lh.la_pred);
                }
            }
            if (!(get_packed && get_packed(backend, n_rb, rb_src, rb_dst, rb_n))) {
                for (int i = 0; i < n_rb; ++i) {
                    ggml_backend_tensor_get_async(backend, rb_src[i], rb_dst[i], 0, rb_n[i]);
                }
            }
        }
        // LLAMA_MOE_LOOKA_ALL: the prediction of a decode ubatch without the CPU path
        const bool la_off = !hyb && la_all && la_backend && lh.la_pred != nullptr && ggml_nelements(lh.la_pred) <= ggml_nelements(rb_la);
        if (la_off) {
            ggml_backend_tensor_get_async(backend, lh.la_pred, rb_la->data, 0, ggml_nbytes(lh.la_pred));
        }
        const int64_t ts0 = ggml_time_us();
        if (guest && hyb && flag_mode) {
            pool_help();
        }
        ggml_backend_synchronize(backend);
        wb_sync(gpu_only_arena(hyb) ? backend : nullptr);
        const int64_t ts1 = ggml_time_us();
        ids.assign((const int32_t *) rb_ids->data, (const int32_t *) rb_ids->data + ggml_nelements(sel));
        if (hyb) {
            // the worker is idle (wait_cpu below) or computes another layer
            lh.x.assign((const float *) rb_x->data, (const float *) rb_x->data + ggml_nelements(lh.x_dev));
            lh.w.assign((const float *) rb_w->data, (const float *) rb_w->data + ggml_nelements(lh.w_dev));
        }

        if (hybrid_auto && !shadow) {
            gate_step(il, (int64_t) ids.size()/n_expert_used, hyb, ts1);
        }

        if (hyb && flag_mode) {
            // the GPU's wait for the CPU in the previous ubatch of this layer (the sync above waited for it)
            volatile uint32_t * f = (volatile uint32_t *) lh.flag->data;
            const double wait_us = f[2]/1000.0;
            t_flag_wait_us += wait_us;
            n_flag_waits++;
            if (hybrid_auto && !shadow) {
                // no wait: the CPU was done before the GPU needed it, by an unknown margin
                share_update(wait_us, wait_us < 2.0 ? 10.0 : 0.0);
            }
            f[2] = 0;
            n_flag_timeouts = std::max<size_t>(n_flag_timeouts, f[3]);
        }

        size_t n_hit = 0;
        if (la_backend) {
            // the layer reads a slot that is still being written
            const int32_t * smj = g.lru.slot_map[il];
            la_join_if(backend, l.ig, [&](auto use) { for (int32_t e : ids) use(smj[e]); });
            // prefetched experts used for the first time
            const int32_t * sm = g.lru.slot_map[il];
            for (int32_t e : ids) {
                if (sm[e] >= 0 && la_mark[l.ig][sm[e]] == il*g.lru.n_expert + e + 1) {
                    la_mark[l.ig][sm[e]] = 0;
                    n_la_useful++;
                }
            }
        }
        if (hyb) {
            // a shadow pass uploads nothing (share 0): the VRAM cache stays as the real pass left it
            g.lru.plan_hybrid(il, ids.data(), ids.size(), shadow ? 0.0 : cur_share(), fills, lh.cpu_experts, n_hit, n_expert_used, tail_rank, &tail_skipped);
            n_tail_skip += tail_skipped.size();
            if (shadow) {
                // the missed experts still on the SSD are left out like skipped ones, and read for the real pass
                // LLAMA_MOE_SHADOW_TOPR=r: the RAM-resident ones too unless some token ranks them in its top r (less CPU work)
                static const int topr = getenv("LLAMA_MOE_SHADOW_TOPR") ? atoi(getenv("LLAMA_MOE_SHADOW_TOPR")) : 0;
                auto top_r = [&](int32_t e) {
                    for (size_t i = 0; i < ids.size(); ++i) {
                        if (ids[i] == e && (int) (i % n_expert_used) < topr) {
                            return true;
                        }
                    }
                    return false;
                };
                shadow_drop.clear();
                size_t n_cut = 0;
                lh.cpu_experts.erase(std::remove_if(lh.cpu_experts.begin(), lh.cpu_experts.end(), [&](int32_t e) {
                    if (expert_resident(l, e)) {
                        if (topr <= 0 || top_r(e)) {
                            return false;
                        }
                        tail_skipped.push_back(e);
                        n_cut++;
                        return true;
                    }
                    shadow_drop.push_back(e);
                    return true;
                }), lh.cpu_experts.end());
                n_shadow_kept    += lh.cpu_experts.size();
                n_shadow_dropped += shadow_drop.size();
                n_shadow_cut     += n_cut;
                tail_skipped.insert(tail_skipped.end(), shadow_drop.begin(), shadow_drop.end());
                if (!shadow_drop.empty()) {
                    arena->ensure(il, shadow_drop.data(), shadow_drop.size());
                }
            }
        } else if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        const int64_t tu0 = ggml_time_us();
        if (willneed) {
            willneed_experts(l.experts, fills, hyb ? lh.cpu_experts : std::vector<int32_t>());
        }
        if (fault_auto) {
            fault_check();
        }
        if (arena && !shadow && (!fault_auto || fault_on) && hyb && ids.size() <= (size_t) 8*n_expert_used) {
            arena_need.assign(lh.cpu_experts.begin(), lh.cpu_experts.end());
            for (const moe_cache_lru::fill & f : fills) {
                arena_need.push_back(f.expert);
            }
            // the next layer's predicted experts that are not in VRAM, best rank first over the tokens, at most fault_pred
            arena_pred.clear();
            if (fault_pred > 0 && arena->no_wait && la_pred_on && lh.la_pred != nullptr && il + 1 < (int32_t) layers.size() && layers[il + 1].ig == l.ig) {
                const int32_t * sm = g.lru.slot_map[il + 1];
                const int32_t * pr = (const int32_t *) rb_la->data;
                const int64_t k = lh.la_pred->ne[0], nt = ggml_nelements(lh.la_pred)/k;
                for (int64_t r = 0; r < k && (int) arena_pred.size() < fault_pred; ++r) {
                    for (int64_t t = 0; t < nt && (int) arena_pred.size() < fault_pred; ++t) {
                        const int32_t e = pr[t*k + r];
                        if (e >= 0 && e < g.lru.n_expert && sm[e] < 0 && std::find(arena_pred.begin(), arena_pred.end(), e) == arena_pred.end()) {
                            arena_pred.push_back(e);
                        }
                    }
                }
            }
            arena->ensure(il, arena_need.data(), arena_need.size(), il + 1, arena_pred.data(), arena_pred.size());
            static const bool demote = getenv("LLAMA_MOE_ARENA_DEMOTE") && atoi(getenv("LLAMA_MOE_ARENA_DEMOTE")) != 0;
            if (arena_store && demote) {
                // the fills are the tail of arena_need
                const size_t nc = lh.cpu_experts.size();
                arena->demote(il, arena_need.data() + nc, arena_need.size() - nc);
            }
        }

        const int64_t tu1 = ggml_time_us();
        // late pass: every upload goes to the second stream unless a token would need more than late_k late slots (those stay
        // early: uploaded before the first pass as before); the late ones read the zero slot in the first pass
        const bool late_on = late_k > 0 && hyb && !shadow && ids.size() <= (size_t) 8*n_expert_used && lh.l_slots != nullptr && lh.late_graph;
        fills_late.clear();
        if (late_on) {
            late_split(lh, g, (int32_t *) lh.l_slots_pin->data, (float *) lh.l_w_pin->data);
        }
        if (hyb) {
            // the experts computed by the CPU read the zero slot on the device, so they add nothing there
            const int32_t * map = (const int32_t *) src->data;
            std::copy(map, map + lh.map_dev.size(), lh.map_dev.begin());
            for (int32_t e : lh.cpu_experts) {
                lh.map_dev[e] = g.n_slots;
            }
            for (const auto & f : fills_late) {
                lh.map_dev[f.expert] = g.n_slots;
            }
            for (int32_t e : tail_skipped) {
                lh.map_dev[e] = g.n_slots;
            }
            lh.ids = ids;
            wait_cpu();
            lh.pf_job = false;
            if (preread && lh.la_pred != nullptr && il + 1 < (int32_t) layers.size() && layers[il + 1].ig == l.ig) {
                // the previous pre-read may still read the list
                pr_stop.store(true, std::memory_order_release);
                while (pr_running.load(std::memory_order_acquire)) {
                }
                pr_stop.store(false, std::memory_order_relaxed);
                const layer & ln = layers[il + 1];
                const int32_t * sm = g.lru.slot_map[il + 1];
                const int32_t * pr = (const int32_t *) rb_la->data;
                const int64_t np = ggml_nelements(lh.la_pred);
                pr_chunks.clear();
                std::vector<int32_t> seen;
                for (int64_t i = 0; i < np; ++i) {
                    const int32_t e = pr[i];
                    if (e < 0 || e >= g.lru.n_expert || sm[e] >= 0 || std::find(seen.begin(), seen.end(), e) != seen.end()) continue;
                    seen.push_back(e);
                    for (const ggml_tensor * w : ln.experts) {
                        const uint8_t * p = (const uint8_t *) w->data + e*w->nb[2];
                        for (size_t o = 0; o < w->nb[2]; o += 65536) {
                            pr_chunks.push_back({ p + o, std::min<size_t>(65536, w->nb[2] - o) });
                        }
                    }
                }
                n_pr_experts += seen.size();
                pr_ready.store(!pr_chunks.empty(), std::memory_order_relaxed);
            }
            lh.t_start = ggml_time_us();
            // LLAMA_MOE_CPU_AFTER_UP: the job waits for this call's uploads (below), so they do not share RAM bandwidth with it
            lh.up_wait = cpu_after_up && !lh.cpu_experts.empty() && !fills.empty();
            up_state.store(0, std::memory_order_relaxed);
            job_done.store(false, std::memory_order_relaxed);
            job_layer.store(il, std::memory_order_release);
        }

        // a refilled slot that is still being written
        if (la_backend) {
            la_join_if(backend, l.ig, [&](auto use) { for (const auto & f : fills) use(f.slot); });
        }
        const int64_t tu2 = ggml_time_us();
#ifdef __linux__
        if (up_probe && hyb) {
            static const uintptr_t page = (uintptr_t) sysconf(_SC_PAGE_SIZE);
            std::vector<unsigned char> vec;
            for (const ggml_tensor * w : l.experts) {
                for (const auto & f : fills) {
                    const uintptr_t b = ((uintptr_t) w->data + f.expert*w->nb[2]) & ~(page - 1);
                    const uintptr_t e = (uintptr_t) w->data + (f.expert + 1)*w->nb[2];
                    const size_t np = (e - b + page - 1)/page;
                    vec.resize(np);
                    if (mincore((void *) b, np*page, vec.data()) == 0) {
                        up_pages += np;
                        for (unsigned char c : vec) {
                            up_pages_out += !(c & 1);
                        }
                    }
                }
            }
        }
#endif
        const int64_t tu3 = ggml_time_us();
        // exclusive arena: the VRAM victims are copied back into the arena first, on this stream, so each reads its slot
        // before the uploads below overwrite it (plus the pad the prompt copies read past an expert: the next slot's start)
        const bool excl_on = arena && arena_excl && hyb && !shadow && !late_on && !la_backend && !doorbell &&
            ids.size() <= (size_t) 8*n_expert_used;
        if (excl_verify && arena) {
            if (wb_ev_mode && wb_backend) {
                ggml_backend_synchronize(wb_backend); // checking only: the copy-backs are otherwise waited for on the GPU
            }
            // the sync at the top of this call finished the previous call's copies
            for (const auto & le : wb_last) {
                arena->verify(le.first, le.second);
            }
            wb_last.clear();
        }
        if (excl_on) {
            uint8_t * dst[16];
            GGML_ASSERT(l.experts.size() <= 16);
            for (const auto & f : fills) {
                if (f.victim < 0) {
                    continue;
                }
                const int32_t vl = f.victim / g.lru.n_expert, ve = f.victim % g.lru.n_expert;
                if (layers[vl].experts.size() != l.experts.size() || !arena->claim_wb(vl, ve, dst)) {
                    continue;
                }
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank = g.banks[ip];
                    const size_t es  = layers[vl].experts[ip]->nb[2];
                    const size_t off = (size_t) f.slot*es;
                    const size_t nb  = std::min(es + moe_arena::pad_after, ggml_nbytes(bank) - off);
                    ggml_backend_tensor_get_async(backend, bank, dst[ip], off, nb);
                    wb_bytes += nb;
                }
                if (excl_verify) {
                    wb_last.push_back({ vl, ve });
                }
            }
        }
        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together.
        // LLAMA_MOE_UPLOAD_ORDER=1 (hybrid decode): the pieces whose pages are all in RAM (mincore) first, the rest after,
        // so the reads still in flight for those have time to land (the pageable copy waits on each missing page)
        size_t bytes = 0;
        const bool order = up_order && hyb && ids.size() <= (size_t) 8*n_expert_used;
        // the pieces left for the second pass, decided once (a piece can land in RAM between the passes)
        up_late.assign(order ? l.experts.size()*fills.size() : 0, 0);
        for (int pass = 0; pass < (order ? 2 : 1); ++pass)
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                const uint8_t * p = arena ? arena->get_ready(il, (int32_t) ip, fills[i].expert) : nullptr;
                size_t n = 1;
                while (p == nullptr && i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n &&
                       (arena == nullptr || arena->get(il, (int32_t) ip, fills[i + n].expert) == nullptr)) {
                    n++;
                }
#ifdef __linux__
                if (order) {
                    uint8_t & late = up_late[ip*fills.size() + i];
                    if (pass == 0 && p == nullptr && !pages_resident((const uint8_t *) w->data + fills[i].expert*expert_size, n*expert_size)) {
                        late = 1;
                        n_up_late++;
                    }
                    if (late != (pass == 1)) {
                        i += n;
                        continue;
                    }
                }
#endif
                ggml_backend_tensor_set_async(backend, bank, p ? p : (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
#ifdef __linux__
                if (cold_on && p == nullptr && ids.size() <= (size_t) 8*n_expert_used) {
                    // the whole pages inside the experts: a page shared with a neighbour stays as it is
                    static const uintptr_t page = (uintptr_t) sysconf(_SC_PAGE_SIZE);
                    const uintptr_t b = ((uintptr_t) w->data + fills[i].expert*expert_size + page - 1) & ~(page - 1);
                    const uintptr_t e = ((uintptr_t) w->data + (fills[i].expert + n)*expert_size) & ~(page - 1);
                    if (e > b) {
                        madvise((void *) b, e - b, MADV_COLD);
                        n_cold += n;
                    }
                }
#endif
                bytes += n*expert_size;
                i += n;
            }
        }
        if (excl_on) {
            // uploaded: they leave the arena (the uploads read them before the next layer call's sync)
            for (const auto & f : fills) {
                arena->release(il, f.expert);
            }
        }
        if (excl_on && wb_backend && g.lru.freq_ld == 0.0) {
            // early copy-back of the next victims (see wb_backend), as many as this call uploaded
            if (wb_ev_mode) {
                // after this call's uploads, so the two directions do not share the host RAM bandwidth
                ggml_backend_event_record(wbup_ev, backend);
                ggml_backend_event_wait(wb_backend, wbup_ev);
            }
            const size_t want = std::max<size_t>(fills.size(), 1);
            size_t n_q = 0;
            uint8_t * dst[16];
            int32_t s = g.lru.head;
            for (size_t step = 0; s >= 0 && step < 2*want + 4 && n_q < want; ++step, s = g.lru.next[s]) {
                const int32_t k = g.lru.key_of[s];
                if (k < 0) {
                    continue;
                }
                const int32_t vl = k / g.lru.n_expert, ve = k % g.lru.n_expert;
                if (vl == il || layers[vl].experts.size() != l.experts.size() || arena->get(vl, 0, ve) != nullptr || !arena->claim_wb(vl, ve, dst)) {
                    continue;
                }
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank = g.banks[ip];
                    const size_t es  = layers[vl].experts[ip]->nb[2];
                    const size_t off = (size_t) s*es;
                    const size_t nb  = std::min(es + moe_arena::pad_after, ggml_nbytes(bank) - off);
                    ggml_backend_tensor_get_async(wb_backend, bank, dst[ip], off, nb);
                    wb_early_bytes += nb;
                }
                if (excl_verify) {
                    wb_last.push_back({ vl, ve });
                }
                n_q++;
                n_wb_early++;
            }
            if (wb_ev_mode) {
                ggml_backend_event_record(wb_ev, wb_backend);
            }
        }
        if (late_on) {
            // the compute stream's small uploads (slot map, late slots and weights) go first: on the same copy engine they
            // would otherwise queue behind the late experts and the first pass would wait for them (no overlap)
            const int64_t n_tok = (int64_t) ids.size()/n_expert_used;
            ggml_backend_tensor_set_async(backend, lh.l_slots, lh.l_slots_pin->data, 0, n_tok*late_k*sizeof(int32_t));
            ggml_backend_tensor_set_async(backend, lh.l_w,     lh.l_w_pin->data,     0, n_tok*late_k*sizeof(float));
            std::copy(lh.map_dev.begin(), lh.map_dev.end(), (int32_t *) lh.map_pin->data);
            ggml_backend_tensor_set_async(backend, dst, lh.map_pin->data, 0, ggml_nbytes(src));
            bytes += late_copy(il, lh, g);
        }
        if (hyb && cpu_after_up) {
            if (lh.up_wait) {
                if (up_ev == nullptr) {
                    up_ev = ggml_backend_event_new(ggml_backend_get_device(backend));
                }
                if (up_ev) {
                    ggml_backend_event_record(up_ev, backend);
                }
            }
            up_state.store(lh.up_wait && up_ev ? 1 : 2, std::memory_order_release);
        }

        if (hyb) {
            t_sync_us   += ts1 - ts0;
            const int64_t tu4 = ggml_time_us();
            t_upload_us    += tu4 - ts1 - (tu3 - tu2);
            t_up_plan_us   += tu0 - ts1;
            t_up_ensure_us += tu1 - tu0;
            t_up_wait_us   += tu2 - tu1;
            t_up_copy_us   += tu4 - tu3;
            up_bytes       += bytes;
            n_hybrid_calls++;
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + fills_late.size() + (hyb ? lh.cpu_experts.size() : 0);
        st.bytes  += bytes;

        if (!hyb) {
            // the next copy synchronizes the backend before it changes the slot map again
            ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
            const int32_t tl = il + 1;
            if (la_off && ids.size() <= (size_t) 8*n_expert_used && tl < (int32_t) layers.size() && la_gi[tl] != nullptr && layers[tl].ig == l.ig) {
                la_ids.assign((const int32_t *) rb_la->data, (const int32_t *) rb_la->data + ggml_nelements(lh.la_pred));
                la_src = il;
                n_la_all_jobs++;
                la_busy.store(true, std::memory_order_relaxed);
                la_job.store(tl, std::memory_order_release);
            }
            return true;
        }

        if (!late_on) {
            std::copy(lh.map_dev.begin(), lh.map_dev.end(), (int32_t *) lh.map_pin->data);
            ggml_backend_tensor_set_async(backend, dst, lh.map_pin->data, 0, ggml_nbytes(src));
        }
        n_cpu_experts += lh.cpu_experts.size();
        lh.planned = true;
        t_planned  = ggml_time_us();

        // lookahead: predict the next layer from this layer's expert input
        const int32_t tl = il + 1;
        if (la_backend && lh.la_pred != nullptr && tl < (int32_t) layers.size() && la_gi[tl] != nullptr && layers[tl].ig == l.ig) {
            la_ids.assign((const int32_t *) rb_la->data, (const int32_t *) rb_la->data + ggml_nelements(lh.la_pred));
            la_src = il;
            la_busy.store(true, std::memory_order_relaxed);
            la_job.store(tl, std::memory_order_release);
        }

        return true;
    }

    // share tuner, after each timed ubatch: ok = this ubatch ran with the CPU path in the held state (no gate probe)
    void tune_step(int64_t n_tok, bool ok, double span_us) {
        if (!ok) {
            if (tune_active) { // a gate probe or the path off: drop the trial, retry after the hold
                tune_active = false;
                tune_delta  = 0.0;
                tune_count  = 0;
            }
            return;
        }
        if (!tune_active) {
            if (++tune_count < tune_hold) {
                return;
            }
            tune_active = true;
            tune_count = tune_cycles = 0;
            tune_arm = 0;
            for (auto & a : acc_arm) { a = span_acc(); }
            tune_delta = -0.1;
            n_tune_trials++;
            return; // the next ubatch is the first of arm 0
        }
        acc_arm[tune_arm].us[n_tok] += span_us;
        acc_arm[tune_arm].n[n_tok]++;
        if (++tune_count < tune_block) {
            return;
        }
        tune_count = 0;
        tune_arm = (tune_arm + 1) % 3;
        if (tune_arm == 0 && ++tune_cycles >= tune_ncycle) {
            // mean span per ubatch size, weighted by the smallest count over the arms
            double t[3] = {0.0, 0.0, 0.0};
            for (int b = 1; b <= 8; ++b) {
                const int w = std::min({acc_arm[0].n[b], acc_arm[1].n[b], acc_arm[2].n[b]});
                if (w >= 8) {
                    for (int a = 0; a < 3; ++a) { t[a] += w*acc_arm[a].us[b]/acc_arm[a].n[b]; }
                }
            }
            int best = 1;
            if (t[1] > 0.0) {
                for (int a : {0, 2}) {
                    if (t[a] < (1.0 - tune_margin)*t[1] && t[a] < t[best]) { best = a; }
                }
            }
            const double old = share_off;
            tune_active = false;
            tune_delta  = 0.0;
            if (best != 1) {
                share_off = std::clamp(share_off + (best == 0 ? -0.1 : 0.1), -0.4, 0.4);
            }
            if (share_off != old) {
                n_tune_moves++;
                tune_hold = 128;
                LLAMA_LOG_INFO("llama_moe_cache: hybrid share tuner: offset %+.1f -> %+.1f (span %.0f / %.0f / %.0f us at %+.1f / 0 / +0.1, share now %.2f)\n",
                    old, share_off, t[0], t[1], t[2], -0.1, cur_share());
            } else {
                tune_hold = std::min(2*tune_hold, 4096);
                LLAMA_LOG_DEBUG("llama_moe_cache: hybrid share tuner: offset %+.1f holds (span %.0f / %.0f / %.0f us)\n", share_off, t[0], t[1], t[2]);
            }
            return;
        }
        tune_delta = tune_arm == 0 ? -0.1 : tune_arm == 1 ? 0.0 : 0.1;
    }

    // auto mode A/B gate, at the slot map copy of each cached layer (after the sync): times each ubatch from the first to the
    // last cached layer, and switches the CPU path for the next graphs
    void gate_step(int32_t il, int64_t n_tok, bool hyb, int64_t t) {
        // with shadow passes the CPU path stays on: they compute the RAM-resident misses on it (set_shadow)
        if (n_tok < 1 || n_tok > 8 || n_shadow_calls > 0) {
            return;
        }
        if (il == il_first) {
            t_first   = t;
            n_tok_cur = n_tok;
            ubatch_on = hyb;
        }
        if (il != il_last || t_first == 0 || n_tok != n_tok_cur) {
            return;
        }
        span_acc & acc = ubatch_on ? acc_on : acc_off;
        t_first_last = t_first;
        acc.us[n_tok] += (double) (t - t_first);
        acc.n[n_tok]++;
        (ubatch_on ? n_ubatch_on : n_ubatch_off)++;
        t_first = 0;
        if (share_tune && share_rule && hybrid_auto) {
            tune_step(n_tok, ubatch_on && gate_phase == 0 && gate_on, (double) (t - t_first_last));
        }

        auto set_path = [&](bool on) {
            gate_changed = gate_changed || path_on != on;
            path_on = on;
        };
        if (++gate_count < (gate_phase == 0 ? gate_hold : gate_probe)) {
            return;
        }
        gate_count = 0;
        if (gate_phase == 0) {
            // probe the other state, then measure the decision right after it, so both see about the same conversation
            acc_on = acc_off = span_acc();
            gate_phase = 1;
            set_path(!gate_on);
            n_gate_probes++;
        } else if (gate_phase == 1) {
            gate_phase = 2;
            set_path(gate_on);
        } else {
            // compare the mean span of each ubatch size seen in both windows, weighted by the smaller count
            double t_on = 0.0, t_off = 0.0;
            for (int b = 1; b <= 8; ++b) {
                const int w = std::min(acc_on.n[b], acc_off.n[b]);
                if (w >= 4) {
                    t_on  += w*acc_on.us[b]/acc_on.n[b];
                    t_off += w*acc_off.us[b]/acc_off.n[b];
                }
            }
            // the CPU path stays on only if it wins by 2%: in a tie it costs CPU time and cache admissions for nothing
            // (at 24 GiB of Qwen3.8 Q2_0 the two are within 2% and keeping the old decision on a tie lost 2-7%)
            bool on = gate_on;
            if (t_on > 0.0 && t_off > 0.0) {
                on = t_on < 0.98*t_off;
            }
            gate_hold = on == gate_on ? std::min(2*gate_hold, 4096) : 256;
            // a probe of the losing state costs its whole window (at 4 GiB, 48 ubatches that upload every miss): when one state
            // wins by 15% or more, wait 8x longer before the next probe (LLAMA_MOE_GATE_BACKOFF=0: the old schedule)
            static const bool backoff = !getenv("LLAMA_MOE_GATE_BACKOFF") || atoi(getenv("LLAMA_MOE_GATE_BACKOFF")) != 0;
            if (backoff && on == gate_on && t_on > 0.0 && t_off > 0.0 && std::min(t_on, t_off) < 0.85*std::max(t_on, t_off)) {
                gate_hold = std::min(8*gate_hold, 32768);
            }
            LLAMA_LOG_DEBUG("llama_moe_cache: hybrid gate: span on %.0f us, off %.0f us, hold %d\n", t_on, t_off, gate_hold);
            if (on != gate_on) {
                LLAMA_LOG_INFO("llama_moe_cache: hybrid gate: CPU path %s (ubatch span with it %.0f us, without %.0f us)\n",
                    on ? "on" : "off", t_on, t_off);
            }
            n_gate_off += on ? 0 : 1;
            gate_on    = on;
            gate_phase = 0;
            set_path(gate_on);
        }
    }

    // hybrid mode: the copy of cpu_out (or a view of it) to the device, after the GPU has been given the cached experts:
    // the CPU computes the missed experts of the layer meanwhile, then uploads the weighted sum
    bool copy_out(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
        const ggml_tensor * base = src->view_src ? src->view_src : src;
        const auto it = layer_of_out.find(base);
        if (it == layer_of_out.end()) {
            return false;
        }
        layer & l = layers[it->second];
        GGML_ASSERT(l.planned && "MoE cache hybrid mode: cpu_out copied without a hybrid slot map copy");
        l.planned = false;
        // the scheduler does not wait for the expert uploads before this callback, the controller below expects it did
        ggml_backend_synchronize(backend);
        wb_sync(gpu_only_arena(true) ? backend : nullptr);
        const int64_t tw0 = ggml_time_us();
        wait_cpu();
        const int64_t tw1 = ggml_time_us();
        t_gap_us  += tw0 - t_planned;
        t_wait_us += tw1 - tw0;
        if (hybrid_auto) {
            // waited: the CPU was late; idle: the CPU finished before the GPU side reached cpu_out
            share_update(tw1 - tw0, std::max<int64_t>(0, (tw0 - l.t_start) - l.t_job));
        }
        GGML_ASSERT((int64_t) l.out.size() == ggml_nelements(src));
        std::copy(l.out.begin(), l.out.end(), (float *) src->data);
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(dst));
        return true;
    }

    cpu_graph & get_cpu_graph(const layer & l, int64_t n_pairs) {
        cpu_graph & cg = cpu_graphs[n_pairs];
        if (cg.gf && cg.act == l.act && cg.limit == l.limit) {
            return cg;
        }
        ggml_init_params params = {
            /*.mem_size   =*/ 16*ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        cg = cpu_graph();
        cg.ctx.reset(ggml_init(params));
        ggml_context * c = cg.ctx.get();

        cg.x   = ggml_new_tensor_3d(c, GGML_TYPE_F32, n_embd, 1, n_pairs);
        cg.sel = ggml_new_tensor_2d(c, GGML_TYPE_I32, 1, n_pairs);
        ggml_tensor * gate = nullptr;
        ggml_tensor * up   = nullptr;
        if (l.gate_up) {
            ggml_tensor * gu = cg.mm_gate_up = ggml_mul_mat_id(c, l.gate_up, cg.x, cg.sel);
            const int64_t n_ff = gu->ne[0]/2;
            gate = ggml_view_3d(c, gu, n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], 0);
            up   = ggml_view_3d(c, gu, n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], n_ff*gu->nb[0]);
        } else {
            up   = cg.mm_up   = ggml_mul_mat_id(c, l.up,   cg.x, cg.sel);
            gate = cg.mm_gate = ggml_mul_mat_id(c, l.gate, cg.x, cg.sel);
        }
        ggml_tensor * act = l.act == llama_moe_cache::ACT_SWIGLU_CLAMP ?
            ggml_swiglu_clamp(c, gate, up, l.limit) : ggml_swiglu_split(c, gate, up);
        cg.y = cg.mm_down = ggml_mul_mat_id(c, l.down, act, cg.sel); // [n_embd, 1, n_pairs]

        cg.gf = ggml_new_graph_custom(c, 16, false);
        ggml_build_forward_expand(cg.gf, cg.y);

        cg.buf.reset(ggml_backend_alloc_ctx_tensors(c, cpu_backend.get()));
        GGML_ASSERT(cg.buf);
        cg.act   = l.act;
        cg.limit = l.limit;
        return cg;
    }

    // the kernel: y[t] += w * down_e(act(gate_e x[t], up_e x[t])) for the CPU's pairs, with ggml's CPU dot products
    void compute_cpu_kernel(layer & l, int64_t n_tokens, int64_t n_used) {
        const int32_t zero = groups[l.ig].n_slots;
        const int32_t l_il = (int32_t) (&l - layers.data());
        // pairs grouped by expert, in the order of the experts' first use
        kexperts.clear();
        kpair_tok.clear();
        kpair_wi.clear();
        for (int32_t e : l.cpu_experts) {
            kexpert ke = { e, (int32_t) kpair_tok.size(), 0 };
            for (int64_t t = 0; t < n_tokens; ++t) {
                for (int64_t k = 0; k < n_used; ++k) {
                    if (l.ids[t*n_used + k] == e && l.map_dev[e] == zero) {
                        kpair_tok.push_back((int32_t) t);
                        kpair_wi.push_back((int32_t) (t*n_used + k));
                    }
                }
            }
            ke.p1 = (int32_t) kpair_tok.size();
            if (ke.p1 > ke.p0) {
                kexperts.push_back(ke);
            }
        }
        const int64_t n_pairs = kpair_tok.size();
        if (n_pairs == 0) {
            return;
        }
        const int64_t tp0 = ggml_time_us();

        const ggml_tensor * wg = l.gate_up ? l.gate_up : l.gate;
        const ggml_tensor * wu = l.gate_up ? l.gate_up : l.up;
        const ggml_tensor * wd = l.down;
        const int64_t n_ff    = l.down->ne[0];
        const int64_t u_off   = l.gate_up ? n_ff : 0;   // row offset of up in a fused gate_up
        const auto * tr_gu = ggml_get_type_traits_cpu(wg->type);
        const auto * tr_d  = ggml_get_type_traits_cpu(wd->type);
        GGML_ASSERT(tr_gu->vec_dot && tr_d->vec_dot && (l.gate_up || wu->type == wg->type));
        const auto * tr_xq = ggml_get_type_traits_cpu(tr_gu->vec_dot_type);
        const auto * tr_aq = ggml_get_type_traits_cpu(tr_d->vec_dot_type);
        const size_t xq_row = ggml_row_size(tr_gu->vec_dot_type, n_embd);
        const size_t aq_row = ggml_row_size(tr_d->vec_dot_type, n_ff);
        const bool fast = q2_dot && wg->type == GGML_TYPE_Q2_0 && wu->type == GGML_TYPE_Q2_0 && wd->type == GGML_TYPE_Q2_0 &&
            tr_gu->vec_dot_type == GGML_TYPE_Q8_0 && tr_d->vec_dot_type == GGML_TYPE_Q8_0 && n_embd % 64 == 0 && n_ff % 64 == 0;
        const int64_t nbx = n_embd/64, nbf = n_ff/64;
        // i-quant experts with 2+ tokens: one decode of each row for all of an expert's tokens (llama-moe-iq.h, ggml's results)
        const llama_moe_iq::dot_mt_t mt_gu = fast ? nullptr : iq_mt(wg->type);
        const llama_moe_iq::dot_mt_t mt_d  = fast ? nullptr : iq_mt(wd->type);
        const int32_t mt_min_gu = iq_choose(wg->type).min_nt, mt_min_d = iq_choose(wd->type).min_nt;

        // the expert input of each token with CPU pairs, quantized once (and prepared for the fast path), one token per thread
        kxq.resize(xq_row*n_tokens);
        if (fast) {
            kxp.resize(nbx*n_tokens);
            kap.resize(nbf*n_pairs);
        }
        kact.resize(n_pairs*n_ff);
        kaq.resize(aq_row*n_pairs);
        ktok.clear();
        for (int64_t t = 0; t < n_tokens; ++t) {
            if (std::find(kpair_tok.begin(), kpair_tok.end(), (int32_t) t) != kpair_tok.end()) {
                ktok.push_back((int32_t) t);
            }
        }
        const int64_t n_qtok = ktok.size();
        auto prep_tok = [&](int64_t t) {
            tr_xq->from_float(l.x.data() + t*n_embd, kxq.data() + t*xq_row, n_embd);
            if (fast) {
                q2_prepare((const q8_block *) (kxq.data() + t*xq_row), n_embd, kxp.data() + t*nbx);
            }
        };
        if (n_qtok == 1) {
            prep_tok(ktok[0]);
        } else {
            pool_run([&](int) {
                for (int64_t i; (i = pool_next.fetch_add(1, std::memory_order_relaxed)) < n_qtok; ) {
                    prep_tok(ktok[i]);
                }
            });
        }
        const bool dot4 = fast && q2_dot4 != nullptr;
        if (dot4 && dot4_check) {
            // once: the 4-row kernel against the 1-row one on real data (LLAMA_MOE_CPU_DOT4_CHECK=1)
            dot4_check = false;
            const kexpert & ke = kexperts[0];
            const char * bg = (const char *) expert_data(l, l_il, wg, ke.e);
            float o4[4];
            q2_dot4(n_embd, (const q2_block *) bg, wg->nb[1], kxp.data() + kpair_tok[ke.p0]*nbx, o4);
            for (int r = 0; r < 4; ++r) {
                const float o1 = q2_dot(n_embd, (const q2_block *) (bg + r*wg->nb[1]), kxp.data() + kpair_tok[ke.p0]*nbx);
                LLAMA_LOG_INFO("llama_moe_cache: dot4 check row %d: %.9g vs %.9g %s\n", r, o4[r], o1, o4[r] == o1 ? "same" : "DIFFERENT");
            }
        }
        const int64_t tp1 = ggml_time_us();

        // rows per gate/up work item (LLAMA_MOE_CPU_RB, a multiple of 4 up to 256; default 16): larger items read longer
        // contiguous runs of each expert's rows (hardware prefetch); the down phase keeps 16-row output blocks
        static const int64_t rb_a = [] {
            const char * v = getenv("LLAMA_MOE_CPU_RB");
            const int64_t r = v ? atoll(v) : 16;
            return std::clamp<int64_t>(r - r % 4, 4, 256);
        }();
        // LLAMA_MOE_CPU_DOWN_PE=<rows> (opt-in): down work items are (expert, block of <rows> output rows), each reading one
        // contiguous run of the expert's down rows; their outputs go to a per-pair buffer, summed afterwards in the same
        // order as the default (expert order, then pair order), so the result is the same bits
        static const int64_t rb_pe = [] {
            const char * v = getenv("LLAMA_MOE_CPU_DOWN_PE");
            const int64_t r = v ? atoll(v) : 0;
            return r <= 0 ? (int64_t) 0 : std::clamp<int64_t>(r - r % 4, 4, 4096);
        }();
        constexpr int64_t rb = 16;  // rows per down output block
        const int64_t nb_ff = (n_ff + rb_a - 1)/rb_a;
        const int64_t nb_e  = (n_embd + rb - 1)/rb;
        const bool clamp = l.act == llama_moe_cache::ACT_SWIGLU_CLAMP;
        const bool oai   = l.act == llama_moe_cache::ACT_SWIGLU_OAI;
        const float limit = l.limit;
        // expert biases: gate and up here; down only for a prefill job (its pairs' router weights are zeroed on the device),
        // in decode the device adds w*b_down for the CPU's pairs itself (they read the zero slot, then the bias is added)
        const float * bgb = l.b_gate_up ? (const float *) l.b_gate_up->data : l.b_gate ? (const float *) l.b_gate->data : nullptr;
        const float * bub = l.b_gate_up ? (const float *) l.b_gate_up->data + n_ff : l.b_up ? (const float *) l.b_up->data : nullptr;
        const int64_t bgu_e = l.b_gate_up ? 2*n_ff : n_ff;  // bias stride per expert
        const float * bdb = l.pf_job && l.b_down ? (const float *) l.b_down->data : nullptr;

        // LLAMA_MOE_CPU_HOT_TEST=1 (measurement): the phases run twice, the second time with the experts' weights hot in the
        // caches; the second run's output is the one kept, both times are summed (an upper bound for pre-reading the experts)
        static const bool hot_test = getenv("LLAMA_MOE_CPU_HOT_TEST") != nullptr;
        std::vector<float> out_save;
        if (hot_test && !l.pf_job) {
            out_save = l.out;
        }
        int64_t tp2 = 0, tp3 = 0, tp4 = 0;
        for (int rep = 0; rep < (hot_test && !l.pf_job ? 2 : 1); ++rep) {
        const int64_t tr0 = ggml_time_us();
        if (rep == 1) {
            l.out = out_save;
        }
        // A: gate and up rows, then the activation, per (expert, row block)
        const int64_t n_items_a = (int64_t) kexperts.size()*nb_ff;
        pool_run([&](int) {
            float g[256*8], u[256*8];
            for (int64_t it; (it = pool_next.fetch_add(1, std::memory_order_relaxed)) < n_items_a; ) {
                const kexpert & ke = kexperts[it/nb_ff];
                const int64_t r0 = (it % nb_ff)*rb_a, r1 = std::min(r0 + rb_a, n_ff);
                const char * bg = (const char *) expert_data(l, l_il, wg, ke.e);
                const char * bu = (const char *) expert_data(l, l_il, wu, ke.e);
                for (int32_t p = ke.p0; p < ke.p1; p += 8) {
                    const int32_t pn = std::min<int32_t>(8, ke.p1 - p);
                    if (dot4 && (r1 - r0) % 4 == 0) {
                        for (int32_t q = 0; q < pn; ++q) {
                            const q2_act * xp = kxp.data() + kpair_tok[p + q]*nbx;
                            for (int64_t r = r0; r < r1; r += 4) {
                                q2_dot4(n_embd, (const q2_block *) (bg + r*wg->nb[1]), wg->nb[1], xp, &g[q*rb_a + r - r0]);
                                q2_dot4(n_embd, (const q2_block *) (bu + (r + u_off)*wu->nb[1]), wu->nb[1], xp, &u[q*rb_a + r - r0]);
                            }
                        }
                    } else if (mt_gu && pn >= mt_min_gu) {
                        const void * ys[8];
                        float og[8], ou[8];
                        for (int32_t q = 0; q < pn; ++q) {
                            ys[q] = kxq.data() + kpair_tok[p + q]*xq_row;
                        }
                        for (int64_t r = r0; r < r1; ++r) {
                            mt_gu(n_embd, bg + r*wg->nb[1], ys, pn, og);
                            mt_gu(n_embd, bu + (r + u_off)*wu->nb[1], ys, pn, ou);
                            for (int32_t q = 0; q < pn; ++q) {
                                g[q*rb_a + r - r0] = og[q];
                                u[q*rb_a + r - r0] = ou[q];
                            }
                        }
                    } else
                    for (int64_t r = r0; r < r1; ++r) {
                        for (int32_t q = 0; q < pn; ++q) {
                            if (fast) {
                                const q2_act * xp = kxp.data() + kpair_tok[p + q]*nbx;
                                g[q*rb_a + r - r0] = q2_dot(n_embd, (const q2_block *) (bg + r*wg->nb[1]), xp);
                                u[q*rb_a + r - r0] = q2_dot(n_embd, (const q2_block *) (bu + (r + u_off)*wu->nb[1]), xp);
                                continue;
                            }
                            const void * xq = kxq.data() + kpair_tok[p + q]*xq_row;
                            tr_gu->vec_dot((int) n_embd, &g[q*rb_a + r - r0], 0, bg + r*wg->nb[1], 0, xq, 0, 1);
                            tr_gu->vec_dot((int) n_embd, &u[q*rb_a + r - r0], 0, bu + (r + u_off)*wu->nb[1], 0, xq, 0, 1);
                        }
                    }
                    for (int32_t q = 0; q < pn; ++q) {
                        float * a = kact.data() + (p + q)*n_ff;
                        for (int64_t r = r0; r < r1; ++r) {
                            float gv = g[q*rb_a + r - r0], uv = u[q*rb_a + r - r0];
                            if (bgb) {
                                gv += bgb[ke.e*bgu_e + r];
                            }
                            if (bub) {
                                uv += bub[ke.e*bgu_e + r];
                            }
                            if (oai) {
                                gv = std::min(gv, limit);
                                uv = std::clamp(uv, -limit, limit);
                                a[r] = gv/(1.0f + expf(1.702f*(-gv)))*(uv + 1.0f);
                                continue;
                            }
                            if (clamp) {
                                gv = std::min(gv, limit);
                                uv = std::clamp(uv, -limit, limit);
                            }
                            a[r] = gv/(1.0f + expf(-gv))*uv;
                        }
                    }
                }
            }
        });
        tp2 = ggml_time_us();
        // A2: the activations quantized for the down projection
        pool_run([&](int) {
            for (int64_t p; (p = pool_next.fetch_add(1, std::memory_order_relaxed)) < n_pairs; ) {
                tr_aq->from_float(kact.data() + p*n_ff, kaq.data() + p*aq_row, n_ff);
                if (fast) {
                    q2_prepare((const q8_block *) (kaq.data() + p*aq_row), n_ff, kap.data() + p*nbf);
                }
            }
        });
        tp3 = ggml_time_us();
        if (rb_pe > 0) {
            // B (per expert): y[p, r] for each pair of the item's expert, then the weighted sums per output row block
            kdy.resize(n_pairs*n_embd);
            const int64_t nb_pe = (n_embd + rb_pe - 1)/rb_pe;
            const int64_t n_items_b = (int64_t) kexperts.size()*nb_pe;
            pool_run([&](int) {
                for (int64_t it; (it = pool_next.fetch_add(1, std::memory_order_relaxed)) < n_items_b; ) {
                    const kexpert & ke = kexperts[it/nb_pe];
                    const int64_t r0 = (it % nb_pe)*rb_pe, r1 = std::min(r0 + rb_pe, n_embd);
                    const char * bd = (const char *) expert_data(l, l_il, wd, ke.e);
                    if (dot4 && (r1 - r0) % 4 == 0) {
                        for (int32_t p = ke.p0; p < ke.p1; ++p) {
                            for (int64_t r = r0; r < r1; r += 4) {
                                q2_dot4(n_ff, (const q2_block *) (bd + r*wd->nb[1]), wd->nb[1], kap.data() + p*nbf, kdy.data() + p*n_embd + r);
                            }
                        }
                        continue;
                    }
                    if (mt_d && ke.p1 - ke.p0 >= mt_min_d) {
                        for (int32_t p = ke.p0; p < ke.p1; p += 8) {
                            const int32_t pn = std::min<int32_t>(8, ke.p1 - p);
                            const void * ys[8];
                            float y[8];
                            for (int32_t q = 0; q < pn; ++q) {
                                ys[q] = kaq.data() + (p + q)*aq_row;
                            }
                            for (int64_t r = r0; r < r1; ++r) {
                                mt_d(n_ff, bd + r*wd->nb[1], ys, pn, y);
                                for (int32_t q = 0; q < pn; ++q) {
                                    kdy[(p + q)*n_embd + r] = y[q];
                                }
                            }
                        }
                        continue;
                    }
                    for (int64_t r = r0; r < r1; ++r) {
                        for (int32_t p = ke.p0; p < ke.p1; ++p) {
                            float * y = kdy.data() + p*n_embd + r;
                            if (fast) {
                                *y = q2_dot(n_ff, (const q2_block *) (bd + r*wd->nb[1]), kap.data() + p*nbf);
                            } else {
                                tr_d->vec_dot((int) n_ff, y, 0, bd + r*wd->nb[1], 0, kaq.data() + p*aq_row, 0, 1);
                            }
                        }
                    }
                }
            });
            pool_run([&](int) {
                for (int64_t it; (it = pool_next.fetch_add(1, std::memory_order_relaxed)) < nb_e; ) {
                    const int64_t r0 = it*rb, r1 = std::min(r0 + rb, n_embd);
                    for (const kexpert & ke : kexperts) {
                        for (int32_t p = ke.p0; p < ke.p1; ++p) {
                            float * o = l.out.data() + kpair_tok[p]*n_embd;
                            const float * y = kdy.data() + p*n_embd;
                            const float wp = l.w[kpair_wi[p]];
                            for (int64_t r = r0; r < r1; ++r) {
                                o[r] += wp*y[r];
                            }
                        }
                    }
                }
            });
        } else
        // B: down rows per row block of the output, all pairs, so that no two threads add to the same output
        pool_run([&](int) {
            for (int64_t it; (it = pool_next.fetch_add(1, std::memory_order_relaxed)) < nb_e; ) {
                const int64_t r0 = it*rb, r1 = std::min(r0 + rb, n_embd);
                for (const kexpert & ke : kexperts) {
                    const char * bd = (const char *) expert_data(l, l_il, wd, ke.e);
                    if (dot4 && (r1 - r0) % 4 == 0) {
                        // per output (token, row) the pairs still add in pair order
                        for (int32_t p = ke.p0; p < ke.p1; ++p) {
                            float* o = l.out.data() + kpair_tok[p]*n_embd;
                            const float wp = l.w[kpair_wi[p]];
                            for (int64_t r = r0; r < r1; r += 4) {
                                float y4[4];
                                q2_dot4(n_ff, (const q2_block *) (bd + r*wd->nb[1]), wd->nb[1], kap.data() + p*nbf, y4);
                                for (int j = 0; j < 4; ++j) {
                                    o[r + j] += wp*y4[j];
                                }
                            }
                        }
                        continue;
                    }
                    if (mt_d && ke.p1 - ke.p0 >= mt_min_d) {
                        // per output (token, row) the pairs still add in pair order
                        for (int32_t p = ke.p0; p < ke.p1; p += 8) {
                            const int32_t pn = std::min<int32_t>(8, ke.p1 - p);
                            const void * ys[8];
                            float y[8];
                            for (int32_t q = 0; q < pn; ++q) {
                                ys[q] = kaq.data() + (p + q)*aq_row;
                            }
                            for (int64_t r = r0; r < r1; ++r) {
                                mt_d(n_ff, bd + r*wd->nb[1], ys, pn, y);
                                for (int32_t q = 0; q < pn; ++q) {
                                    l.out[kpair_tok[p + q]*n_embd + r] += l.w[kpair_wi[p + q]]*y[q];
                                }
                            }
                        }
                        continue;
                    }
                    for (int64_t r = r0; r < r1; ++r) {
                        for (int32_t p = ke.p0; p < ke.p1; ++p) {
                            float y;
                            if (fast) {
                                y = q2_dot(n_ff, (const q2_block *) (bd + r*wd->nb[1]), kap.data() + p*nbf);
                            } else {
                                tr_d->vec_dot((int) n_ff, &y, 0, bd + r*wd->nb[1], 0, kaq.data() + p*aq_row, 0, 1);
                            }
                            l.out[kpair_tok[p]*n_embd + r] += l.w[kpair_wi[p]]*y;
                        }
                    }
                }
            }
        });
        if (bdb) {
            // prefill job: + w*b_down per pair, per output row block (no two threads add to the same output)
            pool_run([&](int) {
                for (int64_t it; (it = pool_next.fetch_add(1, std::memory_order_relaxed)) < nb_e; ) {
                    const int64_t r0 = it*rb, r1 = std::min(r0 + rb, n_embd);
                    for (const kexpert & ke : kexperts) {
                        const float * b = bdb + ke.e*n_embd;
                        for (int32_t p = ke.p0; p < ke.p1; ++p) {
                            float * o = l.out.data() + kpair_tok[p]*n_embd;
                            const float wp = l.w[kpair_wi[p]];
                            for (int64_t r = r0; r < r1; ++r) {
                                o[r] += wp*b[r];
                            }
                        }
                    }
                }
            });
        }
        if (hot_test && !l.pf_job) {
            t_hot[rep] += ggml_time_us() - tr0;
            n_hot += rep;
        }
        }
        if (l.pf_job) {
            return;
        }
        n_cpu_pairs += n_pairs;
        n_ph_kexp += kexperts.size();
        for (const kexpert & ke : kexperts) {
            n_ph_mt += mt_gu && ke.p1 - ke.p0 >= mt_min_gu;
        }
        tp4 = ggml_time_us();
        t_ph[0] += tp0 - l.t_start; t_ph[1] += tp1 - tp0; t_ph[2] += tp2 - tp1; t_ph[3] += tp3 - tp2; t_ph[4] += tp4 - tp3;
        n_ph++;
    }

    // worker thread: the weighted sum of the experts of layer l that the CPU computes, into l.out [n_embd, n_tokens]
    void compute_cpu(layer & l) {
        const int64_t n_tokens = l.x.size()/n_embd;
        const int64_t n_used   = l.w.size()/std::max<int64_t>(n_tokens, 1);

        l.out.assign(n_embd*n_tokens, 0.0f);
        if (l.cpu_experts.empty()) {
            l.t_job = ggml_time_us() - l.t_start;
            return;
        }
        if (l.up_wait) {
            // LLAMA_MOE_CPU_AFTER_UP: start after the layer's uploads are done
            const int64_t tw = ggml_time_us();
            int st;
            while ((st = up_state.load(std::memory_order_acquire)) == 0) {
            }
            if (st == 1) {
                ggml_backend_event_synchronize(up_ev);
            }
            t_after_up_us += ggml_time_us() - tw;
            n_after_up++;
        }
        const int64_t t0 = ggml_time_us();

        if (use_kernel) {
            compute_cpu_kernel(l, n_tokens, n_used);
            l.t_job   = ggml_time_us() - l.t_start;
            if (l.pf_job) {
                t_pf_job_us += l.t_job;
            } else {
                t_cpu_us += ggml_time_us() - t0;
            }
            return;
        }

        // the (token, expert) pairs of the experts the CPU computes
        const int32_t zero = groups[l.ig].n_slots;
        pair_tok.clear();
        pair_wi.clear();
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t k = 0; k < n_used; ++k) {
                if (l.map_dev[l.ids[t*n_used + k]] == zero) {
                    pair_tok.push_back((int32_t) t);
                    pair_wi.push_back((int32_t) (t*n_used + k));
                }
            }
        }
        const int64_t n_pairs = pair_tok.size();

        cpu_graph & cg = get_cpu_graph(l, n_pairs);
        auto point = [](ggml_tensor * node, ggml_tensor * w) {
            if (node) {
                GGML_ASSERT(w && w->type == node->src[0]->type && ggml_are_same_shape(w, node->src[0]));
                node->src[0] = w;
            }
        };
        point(cg.mm_gate_up, l.gate_up);
        point(cg.mm_up,      l.up);
        point(cg.mm_gate,    l.gate);
        point(cg.mm_down,    l.down);
        ggml_tensor * x = cg.x, * sel = cg.sel, * y = cg.y;
        ggml_cgraph * gf = cg.gf;

        float   * xd = (float   *) x->data;
        int32_t * sd = (int32_t *) sel->data;
        for (int64_t p = 0; p < n_pairs; ++p) {
            memcpy(xd + p*n_embd, l.x.data() + (int64_t) pair_tok[p]*n_embd, n_embd*sizeof(float));
            sd[p] = l.ids[pair_wi[p]];
        }
        ggml_backend_graph_compute(cpu_backend.get(), gf);

        const float * yd = (const float *) y->data;
        for (int64_t p = 0; p < n_pairs; ++p) {
            const float w = l.w[pair_wi[p]];
            float * o = l.out.data() + (int64_t) pair_tok[p]*n_embd;
            const float * yp = yd + p*n_embd;
            for (int64_t i = 0; i < n_embd; ++i) {
                o[i] += w*yp[i];
            }
        }

        n_cpu_pairs += n_pairs;
        t_cpu_us    += ggml_time_us() - t0;
        l.t_job      = ggml_time_us() - l.t_start;
    }

    // hybrid mode: register the current graph's tensors of a cached layer
    ggml_tensor * set_hybrid_layer(int32_t il, ggml_tensor * cur, ggml_tensor * weights, llama_moe_cache::act_type act, float limit) {
        if (!hybrid || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0 || layers[il].cpu_out == nullptr) {
            return nullptr;
        }
        layer & l = layers[il];
        if (cur == nullptr || act == llama_moe_cache::ACT_NONE || l.down == nullptr || !(l.gate_up || (l.up && l.gate)) ||
                !cpu_can(l, act)) {
            l.x_dev = nullptr;
            return nullptr;
        }
        l.x_dev = cur;
        l.w_dev = weights;
        l.act   = act;
        l.limit = limit;
        return l.cpu_out;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (fetch) {
            const double nu = (double) std::max<size_t>(n_fetch_ubatches, 1);
            LLAMA_LOG_INFO("llama_moe_cache: GPU fetch: %zu decode ubatches, %zu experts read in place, %zu admitted; per ubatch: before the graph %.1f us, after it %.1f us, helper %.1f us\n",
                n_fetch_ubatches, n_fetch_inplace, n_fetch_admitted, t_fetch_us/nu, t_fetch_end_us/nu, t_fetch_helper_us/nu);
        }
        for (const group & g : groups) {
            if (g.lru.freq_ld > 0.0) {
                LLAMA_LOG_INFO("llama_moe_cache: frequency cache (decay %.4f per step): %zu admitted misses kept on the CPU (colder than every resident)\n",
                    std::exp(-g.lru.freq_ld), g.lru.n_freq_rejected);
            }
        }
        if (hybrid) {
            LLAMA_LOG_INFO("llama_moe_cache: hybrid: %zu missed experts (%zu token-expert pairs) computed by the CPU, %.1f us per pair\n",
                n_cpu_experts, n_cpu_pairs, n_cpu_pairs ? t_cpu_us/n_cpu_pairs : 0.0);
            if (tail_rank > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: tail skip (rank >= %d): %zu missed experts skipped\n", tail_rank, n_tail_skip);
            }
            if (hybrid_auto) {
                LLAMA_LOG_INFO("llama_moe_cache: hybrid auto share: %.2f at the end, %zu changes%s\n", hybrid_share, n_share_changes,
                    share_rule ? (" (rule, CPU " + std::to_string(rule_R_last).substr(0, 5) + " GB/s, link " + std::to_string(link_gbs).substr(0, 5) + " GB/s)").c_str() : "");
                LLAMA_LOG_INFO("llama_moe_cache: hybrid gate: %zu probes, %zu decided off, %zu ubatches with the CPU path, %zu without, ends %s\n",
                    n_gate_probes, n_gate_off, n_ubatch_on, n_ubatch_off, gate_on ? "on" : "off");
                if (share_tune && share_rule) {
                    LLAMA_LOG_INFO("llama_moe_cache: hybrid share tuner: %zu trials, %zu moves, offset %+.1f at the end\n", n_tune_trials, n_tune_moves, share_off);
                }
            }
            if (late_k > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: late pass: %zu calls, %zu experts uploaded on the second stream, %zu kept early (more than %d per token)\n",
                    n_late_calls, n_late, n_late_early, late_k);
            }
            if (flag_mode) {
                LLAMA_LOG_INFO("llama_moe_cache: hybrid flag mode: the GPU waited %.1f us per layer call for the CPU, %zu timeouts\n",
                    t_flag_wait_us/std::max<size_t>(n_flag_waits, 1), n_flag_timeouts);
            }
            if (guest) {
                LLAMA_LOG_INFO("llama_moe_cache: hybrid: the main thread joined %zu CPU job phases\n", n_guest_items);
            }
            if (n_hot > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: hot test: CPU job phases %.1f us cold, %.1f us hot (%zu jobs)\n", t_hot[0]/n_hot, t_hot[1]/n_hot, n_hot);
            }
            if (n_ph > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: hybrid CPU job phases (us per job): wake %.1f, prep %.1f, gate/up %.1f, act quant %.1f, down %.1f, post %.1f (%zu jobs)\n",
                    t_ph[0]/n_ph, t_ph[1]/n_ph, t_ph[2]/n_ph, t_ph[3]/n_ph, t_ph[4]/n_ph, t_ph[5]/n_ph, n_ph);
                LLAMA_LOG_INFO("llama_moe_cache: hybrid CPU jobs: %.2f tokens per expert, %.1f%% of experts by the multi-token i-quant kernels\n",
                    n_ph_kexp ? (double) n_cpu_pairs/n_ph_kexp : 0.0, n_ph_kexp ? 100.0*n_ph_mt/n_ph_kexp : 0.0);
            }
            if (doorbell) {
                const double nd = std::max<size_t>(n_db_calls, 1);
                if (db_async > 0) {
                    LLAMA_LOG_INFO("llama_moe_cache: doorbell background admission: %zu claimed, %zu published, %zu dropped, %.0f us copying per call\n",
                        n_as_claimed, n_as_published, n_as_stale, n_db_calls ? t_as_copy_us/n_db_calls : 0.0);
                }
                LLAMA_LOG_INFO("llama_moe_cache: doorbell: %zu calls, host %.1f us per call, GPU waited %.1f us per call for the host, %zu timeouts\n",
                    n_db_calls, t_db_us/nd, t_db_gpu_wait_us/nd, n_db_timeouts);
            }
            const double n = std::max<size_t>(n_hybrid_calls, 1);
            LLAMA_LOG_INFO("llama_moe_cache: hybrid per layer call (us): sync %.1f, plan+uploads %.1f, until cpu_out %.1f, wait for CPU %.1f, CPU job %.1f (%zu calls)\n",
                t_sync_us/n, t_upload_us/n, t_gap_us/n, t_wait_us/n, t_cpu_us/n, n_hybrid_calls);
            LLAMA_LOG_INFO("llama_moe_cache: plan+uploads split (us): plan %.1f, read-ahead and checks %.1f, wait for the CPU worker and lookahead %.1f, copies %.1f (%.2f MiB per call, %.2f GB/s)\n",
                t_up_plan_us/n, t_up_ensure_us/n, t_up_wait_us/n, t_up_copy_us/n, up_bytes/n/1048576.0, up_bytes/std::max(t_up_copy_us, 1.0)/1e3);
            if (up_order) {
                LLAMA_LOG_INFO("llama_moe_cache: upload order: %zu pieces with pages not in RAM copied last (%.2f per layer call)\n", n_up_late, n_up_late/n);
            }
            if (up_probe) {
                LLAMA_LOG_INFO("llama_moe_cache: upload probe: %.1f%% of %zu uploaded pages not in RAM when the copy started\n",
                    100.0*up_pages_out/std::max<size_t>(up_pages, 1), up_pages);
            }
        }
        if (willneed) {
            LLAMA_LOG_INFO("llama_moe_cache: willneed: %zu layer calls, %zu ranges, %.1f us per call\n",
                n_wn_calls, n_wn_ranges, t_wn_us/std::max<size_t>(n_wn_calls, 1));
        }
        if (cold_on) {
            LLAMA_LOG_INFO("llama_moe_cache: cold: %zu uploaded expert tensors marked cold\n", n_cold);
        }
        if (n_shadow_calls > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: shadow: %zu passes, %zu missed experts computed from RAM, %zu left out and read ahead, %zu cut by rank\n",
                n_shadow_calls, n_shadow_kept, n_shadow_dropped, n_shadow_cut);
            if (shadow_gpu_mode) {
                LLAMA_LOG_INFO("llama_moe_cache: GPU-only shadow: %zu uncached experts found in RAM (computed by the real pass), %zu read-ahead batches, %zu real-pass touches skipped while one ran\n",
                    n_shadow_ram, arena ? arena->n_bulk : 0, arena ? arena->n_bulk_skip : 0);
            }
        }
        if (arena) {
            const moe_arena & a = *arena;
            const double nr = std::max<size_t>(a.n_reads, 1);
            LLAMA_LOG_INFO("llama_moe_cache: arena: %zu calls, %zu hits, %zu reads (%.1f%% hit), %.1f GiB read, %.1f us per call, %.0f us and %.2f MiB per read, %zu not placed, %zu failed\n",
                a.n_calls, a.n_hits, a.n_reads, 100.0*a.n_hits/std::max<size_t>(a.n_hits + a.n_reads, 1), a.bytes/1073741824.0,
                a.t_us/std::max<size_t>(a.n_calls, 1), a.t_us/nr, a.bytes/nr/1048576.0, a.n_full, a.n_fail);
            if (a.n_cold_calls > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: arena: %zu demoted after their upload to VRAM; async reads %d: %zu waits for a part, %.1f us per layer call\n",
                    a.n_demoted, (int) a.async_reads, a.n_ready_waits.load(), a.t_ready_us.load()/(double) std::max<size_t>(a.n_calls, 1));
                LLAMA_LOG_INFO("llama_moe_cache: arena exclusive %d: %zu released after their upload, %zu VRAM victims copied back (%.1f GiB), %zu still here, %zu without a slot; verified %zu, %zu differ; early %zu (%.1f GiB), %zu GPU waits on the copy-backs\n",
                    (int) arena_excl, a.n_released, a.n_wb, wb_bytes/1073741824.0, a.n_wb_here, a.n_wb_full, a.n_verified, a.n_verify_bad,
                    n_wb_early, wb_early_bytes/1073741824.0, n_wb_gpu_waits);
                LLAMA_LOG_INFO("llama_moe_cache: arena prompt share: %zu calls, %zu hits, %zu reads (%.1f%% hit); decode: %zu hits, %zu reads (%.1f%% hit)\n",
                    a.n_cold_calls, a.n_cold_hits, a.n_cold_reads, 100.0*a.n_cold_hits/std::max<size_t>(a.n_cold_hits + a.n_cold_reads, 1),
                    a.n_hits - a.n_cold_hits, a.n_reads - a.n_cold_reads,
                    100.0*(a.n_hits - a.n_cold_hits)/std::max<size_t>(a.n_hits - a.n_cold_hits + a.n_reads - a.n_cold_reads, 1));
            }
            if (a.timing && a.tb_n > 0) {
                const double tb = std::max(a.tb_bytes, 1.0);
                LLAMA_LOG_INFO("llama_moe_cache: fault batches: %zu, %.0f us each; %zu with SSD reads: %.0f us and %.2f MiB each, %.2f GB/s inside them; bytes read at < 1 / 1-2 / 2-3 / >= 3 GB/s: %.0f / %.0f / %.0f / %.0f%%\n",
                    a.tb_n, a.tb_us/a.tb_n, a.tb_n_read, a.tb_us_read/std::max<size_t>(a.tb_n_read, 1), a.tb_bytes/std::max<size_t>(a.tb_n_read, 1)/1048576.0,
                    a.tb_bytes/std::max(a.tb_us_read, 1.0)/1e3, 100*a.tb_bytes_by_rate[0]/tb, 100*a.tb_bytes_by_rate[1]/tb, 100*a.tb_bytes_by_rate[2]/tb, 100*a.tb_bytes_by_rate[3]/tb);
            }
            if (a.n_pred > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: fault prediction: %zu experts touched ahead, %zu of them missed by the next layer (%.1f%%)\n",
                    a.n_pred, a.n_pred_used, 100.0*a.n_pred_used/a.n_pred);
            }
            if (fault_auto) {
                LLAMA_LOG_INFO("llama_moe_cache: fault threads on for %.0f%% of %.0f s, %zu switches\n",
                    100.0*t_fault_on_us/std::max(t_fault_us, 1.0), t_fault_us/1e6, n_fault_switch);
            }
        }
        if (preread) {
            LLAMA_LOG_INFO("llama_moe_cache: pre-read: %zu jobs, %zu experts, %.1f GiB read, %.1f us per job, %zu stopped by the next job\n",
                n_pr_jobs, n_pr_experts, pr_bytes/1073741824.0, t_pr_us/std::max<size_t>(n_pr_jobs, 1), n_pr_aborted);
        }
        if (la_backend) {
            LLAMA_LOG_INFO("llama_moe_cache: lookahead: %zu jobs, %zu experts prefetched, %zu used (%.1f%%), %zu stopped at the running layer, %zu cancelled, "
                "%.1f us per job, callbacks waited %.1f us in all, %zu stream waits\n", n_la_jobs, n_la_issued, n_la_useful, 100.0*n_la_useful/std::max<size_t>(n_la_issued, 1),
                n_la_skipped, n_la_cancelled, t_la_us/std::max<size_t>(n_la_jobs, 1), t_la_wait_us, n_la_joins);
            if (la_all) {
                LLAMA_LOG_INFO("llama_moe_cache: lookahead in decode without the CPU path (LLAMA_MOE_LOOKA_ALL): %zu of the jobs\n", n_la_all_jobs);
            }
        }
        if (cpu_after_up) {
            LLAMA_LOG_INFO("llama_moe_cache: CPU jobs after the uploads (LLAMA_MOE_CPU_AFTER_UP): %zu jobs waited %.1f us each\n", n_after_up, t_after_up_us/std::max<size_t>(n_after_up, 1));
        }
        if (pf && n_pf_layers > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: prefill share: %zu layer calls, %zu of %zu missed experts on the CPU (%zu pairs), CPU %.0f us per expert, GPU %.0f us per upload (last), "
                "per layer: plan %.0f us, CPU job %.0f us, GPU waited %.0f us, %zu mismatches, %zu experts read ahead by the fault threads\n",
                n_pf_layers, n_pf_cpu, n_pf_stream, n_pf_pairs, pf_c_us, pf_g_us, t_pf_plan_us/n_pf_layers, t_pf_job_us/n_pf_layers, t_pf_wait_us/n_pf_layers, n_pf_mismatch, n_pf_touch);
        }
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        if (lend_ig >= 0) {
            LLAMA_LOG_INFO("llama_moe_cache: slot lending: lent %zu times to prompt batches, %zu experts evicted (%zu copied back to the arena), %.1f ms per release + restore\n",
                n_flex_release, n_flex_evicted, n_flex_back, n_flex_release ? flex_us/1000.0/n_flex_release : 0.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
        const std::map<ggml_backend_dev_t, size_t> & dev_budget, size_t flex_size) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, dev_budget, flex_size)) {
}

bool llama_moe_cache::flex_ok(ggml_backend_buffer_type_t buft) {
    auto on = [](const char * v) { const char * e = getenv(v); return e && atoi(e) != 0; };
    const char * f = getenv("LLAMA_MOE_FLEX");
    if (f && atoi(f) == 0) {
        return false;
    }
    const char * hl = getenv("LLAMA_MOE_HOT_LIST");
    if (on("LLAMA_MOE_DOORBELL") || on("LLAMA_MOE_SHADOW") || on("LLAMA_MOE_GPU_FETCH") || on("LLAMA_MOE_LATE") ||
        on("LLAMA_MOE_LOOKA") || on("LLAMA_MOE_PREREAD") || (hl && *hl) || getenv("LLAMA_MOE_VRAM_FREQ")) {
        return false;
    }
    // the banks need a VMM buffer (ggml-cuda); without one the cache is sized for the full ubatch's compute buffers, as before
    ggml_backend_dev_t dev = buft ? ggml_backend_buft_get_device(buft) : nullptr;
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto fchunk = reg ? (size_t (*)(ggml_backend_buffer_type_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_vmm_chunk") : nullptr;
    return fchunk && fchunk(buft) > 0;
}

size_t llama_moe_cache::flex_size() const { return pimpl->lend_ig >= 0 ? pimpl->flex_bytes : 0; }
bool   llama_moe_cache::flex_in() const   { return pimpl->lend_ig >= 0 && !pimpl->lend_out; }
void   llama_moe_cache::flex_release()    { pimpl->flex_release(); }
void   llama_moe_cache::flex_restore()    { pimpl->flex_restore(); }

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph) || (pimpl->hybrid && pimpl->copy_out(backend, src, dst));
}

bool llama_moe_cache::hybrid_configured() const {
    return pimpl->hybrid;
}

bool llama_moe_cache::lookahead_all() const {
    return pimpl->la_all && pimpl->la_backend != nullptr;
}

bool llama_moe_cache::hybrid_enabled() const {
    return pimpl->hybrid && (!pimpl->hybrid_auto || pimpl->path_on || pimpl->store_gate_path()) && !pimpl->shadow_gpu && !pimpl->fetch;
}

bool llama_moe_cache::set_shadow(bool on) {
    impl & p = *pimpl;
    // LLAMA_MOE_SHADOW_FORCE=1 (debugging): also while the fault threads are off
    static const bool force = getenv("LLAMA_MOE_SHADOW_FORCE") && atoi(getenv("LLAMA_MOE_SHADOW_FORCE")) != 0;
    const bool ok = p.hybrid && p.arena && p.arena->touch_only && p.arena->no_wait && (force || !p.fault_auto || p.fault_on);
    if (p.shadow_gpu_mode) {
        // the graph changes both ways (device slot maps, no hybrid path), so it is rebuilt
        if (on && ok) {
            p.shadow_gpu_begin();
            p.shadow_gpu   = true;
            p.gate_changed = true;
            p.gate_on      = true; // the real passes keep the CPU path (the gate is skipped once shadows run)
            p.path_on      = true;
            p.n_shadow_calls++;
            return true;
        }
        if (!on && p.shadow_gpu) {
            p.shadow_gpu_end();
            p.shadow_gpu   = false;
            p.gate_changed = true;
        }
        return false;
    }
    p.shadow = on && ok;
    if (p.shadow) {
        p.n_shadow_calls++;
        if (!p.gate_on || !p.path_on) {
            p.gate_on      = true;
            p.path_on      = true;
            p.gate_changed = true;
        }
    }
    return p.shadow;
}

ggml_tensor * llama_moe_cache::fetch_slot_map(int32_t il, int64_t n_tokens) const {
    return pimpl->fetch_slot_map(il, n_tokens);
}

void llama_moe_cache::fetch_register_ids(int32_t il, ggml_tensor * ids) const {
    if (il >= 0 && il < (int32_t) pimpl->layers.size()) {
        pimpl->layers[il].fetch_ids = ids;
    }
}

void llama_moe_cache::fetch_begin(int64_t n_tokens) {
    pimpl->fetch_begin(n_tokens);
}

void llama_moe_cache::fetch_end() {
    pimpl->fetch_end();
}

ggml_tensor * llama_moe_cache::shadow_slot_map(int32_t il) const {
    return pimpl->shadow_slot_map(il);
}

void llama_moe_cache::shadow_register_ids(int32_t il, ggml_tensor * ids) const {
    if (il >= 0 && il < (int32_t) pimpl->layers.size()) {
        pimpl->layers[il].shadow_ids = ids;
    }
}

bool llama_moe_cache::take_graph_change() const {
    const bool res = pimpl->gate_changed;
    pimpl->gate_changed = false;
    return res;
}

ggml_tensor * llama_moe_cache::set_hybrid_layer(int32_t il, ggml_tensor * cur, ggml_tensor * weights, act_type act, float limit) const {
    return pimpl->set_hybrid_layer(il, cur, weights, act, limit);
}

ggml_tensor * llama_moe_cache::doorbell_slots(ggml_context * ctx, int32_t il, ggml_tensor * ids, ggml_tensor * x, ggml_tensor * w, int64_t n_tokens, int64_t n_used) const {
    return pimpl->doorbell_slots(ctx, il, ids, x, w, n_tokens, n_used);
}

int llama_moe_cache::late_k(int32_t il, int64_t n_tokens) const {
    const auto & p = *pimpl;
    if (p.late_k <= 0 || n_tokens > 8 || il < 0 || il >= (int32_t) p.layers.size() || p.layers[il].l_slots == nullptr) {
        return 0;
    }
    return p.late_k;
}

ggml_tensor * llama_moe_cache::late_barrier(ggml_context * ctx, int32_t il, ggml_tensor * x) const {
    const auto & p = *pimpl;
    const auto & l = p.layers[il];
    pimpl->layers[il].late_graph = true;
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    if (l.db_out != nullptr) {
        // doorbell: the op uploads the late experts on a second stream; this join waits for them
        l.db_out->op_params[3] = 1;
        return ggml_host_add(ctx, x, nullptr, (uint32_t *) l.db->data);
    }
    return ggml_host_add(ctx, x, (const float *) p.lt_zero[p.groups[l.ig].id]->data, (uint32_t *) l.l_flag->data);
}

ggml_tensor * llama_moe_cache::late_slots(ggml_context * ctx, int32_t il, int64_t n_tokens) const {
    const auto & p = *pimpl;
    const auto & l = p.layers[il];
    if (l.db_out != nullptr) {
        const int64_t off = ((ggml_nelements(l.db_out) - GGML_MOE_DB_SCRATCH(p.db_max_fill, p.late_k) + 1) & ~1) + GGML_MOE_DB_LATE_OFF(p.db_max_fill);
        return ggml_view_2d(ctx, l.db_out, p.late_k, n_tokens, p.late_k*sizeof(int32_t), off*sizeof(int32_t));
    }
    ggml_tensor * t = l.l_slots;
    return ggml_view_2d(ctx, t, p.late_k, n_tokens, p.late_k*sizeof(int32_t), 0);
}

ggml_tensor * llama_moe_cache::late_weights(ggml_context * ctx, int32_t il, int64_t n_tokens, ggml_tensor * w) const {
    const auto & p = *pimpl;
    const auto & l = p.layers[il];
    if (l.db_out != nullptr) {
        // the ranks follow the late slots [8*K]: the late experts' router weights by rank, [1, K, n_tokens]
        const int64_t off = ((ggml_nelements(l.db_out) - GGML_MOE_DB_SCRATCH(p.db_max_fill, p.late_k) + 1) & ~1) + GGML_MOE_DB_LATE_OFF(p.db_max_fill) + 8*p.late_k;
        ggml_tensor * r = ggml_view_2d(ctx, l.db_out, p.late_k, n_tokens, p.late_k*sizeof(int32_t), off*sizeof(int32_t));
        return ggml_get_rows(ctx, ggml_reshape_3d(ctx, w, 1, ggml_nelements(w)/n_tokens, n_tokens), r);
    }
    ggml_tensor * t = l.l_w;
    return ggml_view_3d(ctx, t, 1, p.late_k, n_tokens, sizeof(float), p.late_k*sizeof(float), 0);
}

ggml_tensor * llama_moe_cache::hybrid_add(ggml_context * ctx, int32_t il, ggml_tensor * moe_out) const {
    return pimpl->hybrid_add(ctx, il, moe_out);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last, int64_t e0, const void ** src) {
    return pimpl->copy_experts(backend, w, dst, e, last, e0, src);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf || d.buf_size > 0) {
            res[d.buft] += d.buf_size;   // with slot lending, d.buf_size includes the banks' VMM buffer
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}

bool llama_moe_cache::prefill_enabled(int64_t n_tokens) const {
    return pimpl->prefill_enabled(n_tokens);
}

void llama_moe_cache::disable_prefill_share() {
    if (pimpl->pf) {
        LLAMA_LOG_INFO("llama_moe_cache: no hybrid prefill share without op offload\n");
    }
    pimpl->pf = false;
}

ggml_tensor * llama_moe_cache::set_prefill_layer(int32_t il, ggml_tensor * x, ggml_tensor * w, const ggml_tensor * ids, act_type act, float limit) const {
    return pimpl->set_prefill_layer(il, x, w, ids, act, limit);
}

ggml_tensor * llama_moe_cache::prefill_add(ggml_context * ctx, int32_t il, ggml_tensor * moe_out) const {
    return pimpl->prefill_add(ctx, il, moe_out);
}

void llama_moe_cache::prefill_arena(const ggml_tensor * w, const std::vector<bool> & used) {
    pimpl->prefill_arena(w, used);
}

const uint8_t * llama_moe_cache::arena_expert(const ggml_tensor * w, int64_t e) const {
    return pimpl->arena_expert(w, e);
}

bool llama_moe_cache::prefill_plan(ggml_backend_t backend, const ggml_tensor * w, const ggml_tensor * ids, std::vector<int32_t> & ids_data, std::vector<bool> & used) {
    return pimpl->prefill_plan(backend, w, ids, ids_data, used);
}

ggml_tensor * llama_moe_cache::lookahead_router(int32_t il, ggml_tensor ** bias, int * k) const {
    return pimpl->lookahead_router(il, bias, k);
}

void llama_moe_cache::set_lookahead_layer(int32_t il, ggml_tensor * pred) const {
    pimpl->set_lookahead_layer(il, pred);
}

// large-batch expert chunks

llama_moe_chunks::llama_moe_chunks(const llama_model & model, const std::vector<ggml_backend_t> & backends, int32_t n, bool any_backend) {
    ggml_init_params params = {
        /*.mem_size   =*/ model.layers.size()*4*n*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx.reset(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to create the MoE chunk context");
    }
    // a probe of the large-batch op: the backend must accept a range's MUL_MAT_ID (out-of-range ids, GGML_MUL_MAT_ID_RANGED_PARAM)
    ggml_init_params probe_params = {
        /*.mem_size   =*/ 8*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr probe_ctx(ggml_init(probe_params));
    const int64_t n_used = 8; // any count: the support depends on the type and the batch
    auto supported = [&](ggml_backend_t backend, ggml_tensor * view) {
        ggml_tensor * b   = ggml_new_tensor_3d(probe_ctx.get(), GGML_TYPE_F32, view->ne[0], n_used, 32);
        ggml_tensor * ids = ggml_new_tensor_2d(probe_ctx.get(), GGML_TYPE_I32, n_used, 32);
        ggml_tensor * op  = ggml_mul_mat_id(probe_ctx.get(), view, b, ids);
        op->op_params[GGML_MUL_MAT_ID_RANGED_PARAM] = 1;
        const bool ok = ggml_backend_supports_op(backend, op);
        ggml_reset(probe_ctx.get());
        return ok;
    };
    size_t n_w = 0, n_skip = 0;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto it_be = std::find_if(backends.begin(), backends.end(), [&](ggml_backend_t b) { return ggml_backend_get_device(b) == model.dev_layer(il); });
        if (it_be == backends.end() || ggml_backend_dev_type(model.dev_layer(il)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        // only backends that check the mark: the others would write the out-of-range pairs
        const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(model.dev_layer(il)));
        if (strcmp(reg, "CUDA") != 0 && !(any_backend && strcmp(reg, "ROCm") == 0)) {
            continue;
        }
        for (ggml_tensor * w : llama_moe_cache_layer_experts(model.layers[il])) {
            // no data: a context of a model loaded without allocation (the -fit estimate), views need the weights' memory;
            // quantized only: CUDA multiplies those with MMQ, which skips out-of-range ids (f16/bf16 experts take other paths)
            if (!llama_moe_cache_is_host_weight(w) || w->data == nullptr || !ggml_is_quantized(w->type) || w->ne[2] < n) {
                continue;
            }
            std::vector<ggml_tensor *> chunks;
            for (int32_t k = 0; k < n; ++k) {
                const int64_t e0 = w->ne[2]*k/n, e1 = w->ne[2]*(k + 1)/n;
                ggml_tensor * t = ggml_view_3d(ctx.get(), w, w->ne[0], w->ne[1], e1 - e0, w->nb[1], w->nb[2], e0*w->nb[2]);
                ggml_format_name(t, "%s.c%d", w->name, k);
                ggml_backend_view_init(t);
                chunks.push_back(t);
            }
            if (!supported(*it_be, chunks[0])) {
                n_skip++;
                continue;
            }
            entry & en = views[w];
            en.backend = *it_be;
            for (int32_t k = 0; k < n; ++k) {
                src[chunks[k]] = { w, w->ne[2]*k/n };
            }
            en.chunks = std::move(chunks);
            n_w++;
        }
    }
    if (n_w > 0 || n_skip > 0) {
        LLAMA_LOG_INFO("llama_moe_chunks: large batches multiply %zu host expert tensors in %d expert ranges (%zu not supported)\n", n_w, n, n_skip);
    }
}

const std::vector<ggml_tensor *> & llama_moe_chunks::get(const ggml_tensor * w, ggml_backend_t * backend) const {
    static const std::vector<ggml_tensor *> none;
    const auto it = views.find(w);
    if (it == views.end()) {
        return none;
    }
    *backend = it->second.backend;
    return it->second.chunks;
}

const ggml_tensor * llama_moe_chunks::chunk_of(const ggml_tensor * view, int64_t * e0) const {
    const auto it = src.find(view);
    if (it == src.end()) {
        return nullptr;
    }
    *e0 = it->second.second;
    return it->second.first;
}
