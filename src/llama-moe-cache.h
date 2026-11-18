#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <unordered_map>
#include <string>
#include <memory>
#include <vector>

struct llama_model;

// the RAM this process can fill (Linux): MemTotal, or a smaller memory limit of its cgroup: v2 memory.max of its cgroup or one
// above it, including the namespace root a container sees as "0::/" (Docker, Podman), or v1 memory.limit_in_bytes; 0: unknown.
// Inline: the server uses it too. root: a prefix for /proc and /sys (tests)
inline size_t llama_moe_ram_limit(const std::string & root = "") {
    size_t lim = 0;
#ifdef __linux__
    std::ifstream mi(root + "/proc/meminfo");
    std::string k;
    size_t v = 0;
    while (mi >> k >> v) {
        if (k == "MemTotal:") {
            lim = v << 10;
            break;
        }
        std::getline(mi, k);
    }
    auto cap = [&lim](const std::string & file) {
        std::ifstream f(file);
        std::string s;
        if (f >> s && s != "max" && !s.empty() && s.find_first_not_of("0123456789") == std::string::npos) {
            const size_t m = std::stoull(s);
            if (m > 0 && m < ((size_t) 1 << 62)) { // v1 writes ~2^63 for no limit
                lim = lim > 0 ? std::min(lim, m) : m;
            }
        }
    };
    std::ifstream cg(root + "/proc/self/cgroup");
    std::string line;
    while (std::getline(cg, line)) {
        const size_t c1 = line.find(':');
        const size_t c2 = c1 == std::string::npos ? std::string::npos : line.find(':', c1 + 1);
        if (c2 == std::string::npos) {
            continue;
        }
        const std::string ctrl = line.substr(c1 + 1, c2 - c1 - 1);
        std::string path = line.substr(c2 + 1);
        const bool v2 = line.rfind("0::", 0) == 0;
        const bool v1 = !v2 && (ctrl == "memory" || ctrl.find("memory,") == 0 || ctrl.find(",memory") != std::string::npos);
        if (!v2 && !v1) {
            continue;
        }
        const std::string base = root + (v2 ? "/sys/fs/cgroup" : "/sys/fs/cgroup/memory");
        const std::string file = v2 ? "/memory.max" : "/memory.limit_in_bytes";
        // this cgroup and every one above it, then the mount root (a namespaced container's own cgroup)
        for (; !path.empty() && path != "/"; path = path.substr(0, path.rfind('/'))) {
            cap(base + path + file);
        }
        cap(base + file);
    }
#endif
    return lim;
}

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
// (or, in hybrid mode, part of them, the CPU computes the others)
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // dev_budget: with the default split (no tensor split), the split weights by device instead of their free memory
    // flex_size: of size, the bytes of slots (single GPU, see flex_release) that a prompt's larger compute buffers borrow
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
        const std::map<ggml_backend_dev_t, size_t> & dev_budget = {}, size_t flex_size = 0);
    ~llama_moe_cache();

    // slot lending (LLAMA_MOE_FLEX, single GPU with VMM): the banks live in a VMM buffer and the largest group lends the physical
    // memory under its last slots, as much as the difference between the compute buffers of a prompt ubatch and of a small one.
    // Before a prompt batch they are released (their experts evicted, copied back to the arena in exclusive store mode) so the
    // larger compute buffers fit; after it, mapped again at the same addresses (empty: refilled by misses). flex_ok: the
    // environment and the device allow it (no doorbell, shadow, GPU fetch, late pass, lookahead, hot list or VRAM_FREQ)
    static bool flex_ok(ggml_backend_buffer_type_t buft);
    size_t flex_size() const;   // 0: no flex group
    bool   flex_in() const;     // allocated (not lent out)
    void   flex_release();
    void   flex_restore();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of
    // w's experts from e0 on (a range, see llama_moe_chunks); returns the number of experts copied, 0 if expert e is not in the cache
    // src: if not nullptr, nothing is copied; *src is set to the device address of the first expert (the caller copies)
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last, int64_t e0 = 0, const void ** src = nullptr);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // hybrid mode (LLAMA_MOE_CPU=<share of the misses uploaded, 0..1>): the CPU computes the other missed experts while the GPU
    // computes the cached ones, and the GPU adds the CPU's output (cpu_out) to the MoE output
    // ACT_SWIGLU_OAI: gpt-oss, min(g, limit) * sigmoid(1.702 min(g, limit)) * (clamp(u, -limit, limit) + 1)
    enum act_type { ACT_NONE = -1, ACT_SWIGLU = 0, ACT_SWIGLU_CLAMP = 1, ACT_SWIGLU_OAI = 2 };

    // hybrid mode is configured (LLAMA_MOE_CPU is set)
    bool hybrid_configured() const;

    // the graph being built should compute the missed experts on the CPU: configured, and in auto mode not switched off by
    // the A/B gate (which compares ubatch times with the CPU path on and off)
    bool hybrid_enabled() const;

    // true once after the gate switched the CPU path on or off: the previous graph must not be reused
    bool take_graph_change() const;

    // LLAMA_MOE_SHADOW (slipstream for models read from the SSD): while on, decode ubatches neither upload nor evict experts,
    // the CPU computes the missed experts that are in RAM, and the ones still on the SSD are left out (their contribution is
    // dropped) and read by the fault threads for the real pass that follows. Returns whether the shadow is on: only when the
    // fault threads are on (many page faults)
    bool set_shadow(bool on);
    // GPU-only shadow (LLAMA_MOE_SHADOW=2): the device slot map of a layer while a shadow graph is built (nullptr otherwise),
    // and the selected experts of that graph, read after the pass
    ggml_tensor * shadow_slot_map(int32_t il) const;
    // LLAMA_MOE_GPU_FETCH: the device slot map of layer il for a decode graph of n_tokens (nullptr: the usual split), the
    // graph's selected experts (an output), and the per-ubatch step (before the graph is built or reused)
    ggml_tensor * fetch_slot_map(int32_t il, int64_t n_tokens) const;
    void fetch_register_ids(int32_t il, ggml_tensor * ids) const;
    void fetch_begin(int64_t n_tokens);
    void fetch_end(); // after the graph is launched
    void shadow_register_ids(int32_t il, ggml_tensor * ids) const;

    // called while building the graph of a cached layer: cur [n_embd, n_tokens] and weights [1, n_expert_used, n_tokens] are the
    // contiguous expert input and router weights, computed before the slot lookup; returns the host tensor to add to the MoE
    // output (a view of it for n_tokens), nullptr if the layer is not computed in hybrid mode
    ggml_tensor * set_hybrid_layer(int32_t il, ggml_tensor * cur, ggml_tensor * weights, act_type act, float limit) const;

    // moe_out [n_embd, n_tokens] plus the output of the experts the CPU computes, for a layer set_hybrid_layer returned a tensor for:
    // with pinned host memory the GPU waits for a flag the CPU sets, so the graph is launched before the CPU is done
    // (LLAMA_MOE_CPU_COPY=1, or pageable memory: cpu_out is copied as a host weight once the CPU is done)
    ggml_tensor * hybrid_add(ggml_context * ctx, int32_t il, ggml_tensor * moe_out) const;

    // late pass (LLAMA_MOE_LATE=K, hybrid decode): the uploaded missed experts go on a second stream
    // while the GPU computes the cached ones (the first pass reads the zero slot for them), then a pass of K slots per token
    // computes them once the copies are in. late_k(il, n_tokens): K, 0 if not used for this layer and size. late_barrier: x
    // after the copies of the layer's last upload are in (a device flag the copy stream writes). late_slots: I32 [K, n_tokens]
    // (the zero slot for padding), late_weights: F32 [1, K, n_tokens] (0 for padding)
    int late_k(int32_t il, int64_t n_tokens) const;
    ggml_tensor * late_barrier(ggml_context * ctx, int32_t il, ggml_tensor * x) const;
    ggml_tensor * late_slots(ggml_context * ctx, int32_t il, int64_t n_tokens) const;
    // w: the router weights [1, n_used, n_tokens] (doorbell mode reads the late experts' weights from it by rank)
    ggml_tensor * late_weights(ggml_context * ctx, int32_t il, int64_t n_tokens, ggml_tensor * w) const;

    // doorbell mode (LLAMA_MOE_DOORBELL=1): the slots of the selected experts ids [n_used*n_tokens] for up to 8 tokens, from a device op
    // that hands them (and in hybrid mode x and w, see set_hybrid_layer) to a host thread; nullptr if not used for this layer and size
    ggml_tensor * doorbell_slots(ggml_context * ctx, int32_t il, ggml_tensor * ids, ggml_tensor * x, ggml_tensor * w, int64_t n_tokens, int64_t n_used) const;

    // lookahead prefetch (LLAMA_MOE_LOOKA=<experts per layer>): the router of layer il+1 and its selection bias (may be nullptr),
    // which the graph applies to layer il's expert input; k = predicted experts per token. nullptr: layer il does not predict
    ggml_tensor * lookahead_router(int32_t il, ggml_tensor ** bias, int * k) const;
    // LLAMA_MOE_LOOKA_ALL=1 with the lookahead on: predict and prefetch in decode ubatches without the CPU path too
    bool lookahead_all() const;

    // registers the predicted experts [k, n_tokens] (I32, contiguous, computed before layer il's slot lookup), nullptr clears
    void set_lookahead_layer(int32_t il, ggml_tensor * pred) const;

    // prefill share (LLAMA_MOE_PREFILL_CPU, hybrid flag mode): ubatches of n_tokens on the large-batch path hand the CPU part of
    // the missed experts
    bool prefill_enabled(int64_t n_tokens) const;

    // the prefill share runs from the scheduler's copy callback of the large-batch weight upload: without op offload
    // (--no-op-offload) there is no upload, so the context turns the share off
    void disable_prefill_share();

    // called while building a large-batch graph for each cached layer: x [n_embd, n_tokens] and w [1, n_expert_used, n_tokens] are
    // the contiguous expert input and router weights (w the one the expert outputs are multiplied by), ids the selected experts
    // the MUL_MAT_IDs read; x nullptr clears the layer. Returns non-null if the layer takes the share: then the graph must add it
    // with prefill_add
    ggml_tensor * set_prefill_layer(int32_t il, ggml_tensor * x, ggml_tensor * w, const ggml_tensor * ids, act_type act, float limit) const;

    // moe_out [n_embd, n_tokens] plus the CPU's experts (the GPU waits for a flag, see ggml_host_add)
    ggml_tensor * prefill_add(ggml_context * ctx, int32_t il, ggml_tensor * moe_out) const;

    // large-batch copy callback, once per ubatch and layer after the selected experts were read (ids_data, with ids' strides):
    // takes the CPU's experts out of used and rewrites ids and the router weights on the device; false if the layer has no share
    bool prefill_plan(ggml_backend_t backend, const ggml_tensor * w, const ggml_tensor * ids, std::vector<int32_t> & ids_data, std::vector<bool> & used);
    // LLAMA_MOE_ARENA_MIB=auto (store mode): read a prompt layer's used experts the VRAM cache lacks into the arena, then
    // the prompt's copies take them from there (arena_expert, nullptr: from the mapping)
    void prefill_arena(const ggml_tensor * w, const std::vector<bool> & used);
    const uint8_t * arena_expert(const ggml_tensor * w, int64_t e) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;

// large-batch expert chunks (4 by default on CUDA, LLAMA_MOE_CHUNKS=<n>, 0 = off): a large batch multiplies the host experts of a GPU layer in n expert ranges,
// one MUL_MAT_ID each with the ids of the other ranges out of range, so the scheduler's copy of the weights for the op offload
// is 1/n of the expert tensor; the compute buffer shrinks by about (n-1)/n of the largest one, the MoE cache gets the rest
class llama_moe_chunks {
public:
    llama_moe_chunks(const llama_model & model, const std::vector<ggml_backend_t> & backends, int32_t n, bool any_backend);

    // the views of w's expert ranges, empty if w is not chunked; backend: the device of w's layer, which computes them
    const std::vector<ggml_tensor *> & get(const ggml_tensor * w, ggml_backend_t * backend) const;

    // the host experts of a range view and its first expert, nullptr if view is not one
    const ggml_tensor * chunk_of(const ggml_tensor * view, int64_t * e0) const;

private:
    struct entry {
        ggml_backend_t backend = nullptr;
        std::vector<ggml_tensor *> chunks;
    };
    ggml_context_ptr ctx;
    std::unordered_map<const ggml_tensor *, entry> views;
    std::unordered_map<const ggml_tensor *, std::pair<const ggml_tensor *, int64_t>> src; // view -> host experts, first expert
};

using llama_moe_chunks_ptr = std::unique_ptr<llama_moe_chunks>;
