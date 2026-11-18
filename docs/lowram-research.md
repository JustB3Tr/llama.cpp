# Low-RAM MoE research and experiment notes

Snapshot: 2026-11-17. This is an experimental fork, not a claim that all models or modalities run correctly.

## Model identity

* [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) is the earlier text model, 284B total / 13B active per its model card. The local two-shard Q2_K_S GGUF totals 98,592,510,368 bytes; its metadata says `deepseek4`, 43 blocks, 256 experts and 6 experts used. These describe this file, not every V4 variant.
* [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) is a different multimodal 552B-backbone model, not a renamed V4 Flash. A GGUF conversion and the matching architecture and vision path must be checked separately. [tandem.cpp](https://github.com/ewhacc/tandem.cpp) reports V4.1 benchmarks using a particular 139 GiB expert quant; this is not a prediction for an RTX 4060.
* [GLM-5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is a different multimodal 320B / 18B-active model. [Upstream added its text architecture](https://github.com/ggml-org/llama.cpp/pull/27773); this fork's actual model, quant, text and vision compatibility remain untested.

The pinned source registers distinct `deepseek4`, `deepseek41`, and `glm5-next` architectures and model implementations. Registration alone does not establish that a particular GGUF, its tokenizer, multimodal projector, or every execution path works correctly.

## What exists already

This branch is based on llama.cpp `f5222f25c18da062ab2b8a2dbbfbf4feae398566` plus the locally installed tandem.cpp patch. That patch contains GPU expert caching, host expert arenas, async read paths, fault-triggered prefetch and optional router lookahead. Preserve the upstream tandem license and attribution. The exact locally installed snapshot is used here; do not assume it equals the latest GitHub version. Existing tandem code has counters for cache hits, reads, CPU work, read-ahead and faults when statistics are enabled. Check logs for activation, rather than inferring it from allocated GPU memory.

At inspection time the local tandem repository HEAD was `64a6b186f650bca7638b7c5baccb118587d1f715`, with a modified patch and related files. Remote tandem HEAD was `54d03ac6424912741501785de757dcb559e29667`; remote llama.cpp HEAD was `401563f10ec60754e688cd3165c1343cf4be9642`. This branch is pinned to the local tandem base, not the current upstream tip. Porting later upstream changes requires review and renewed tests.

The launcher in `tools/lowram_tandem_run.py` is adapted from tandem's launcher. The original stopped scanning GGUF metadata at the first tokenizer key. The local GGUF puts expert metadata later, causing the original to see an architecture without an expert count and treat it as dense. This copy reads the entire header, warns on missing expert counts, offers explicit GPU layer count, and warns about cache starvation on 8 GiB cards. Unknown architecture or missing metadata does not establish model compatibility. It still chooses heuristic settings; validate generated commands.

The fork also decouples automatic RAM store selection from the fault-thread auto setting. This is a small behavioral correction in the tandem patch and has not yet been shown to improve inference. Inference, memory usage, and correctness need A/B testing before relying on it.

## Engineering choices and constraints

* **Capacity:** Swap is SSD-backed, not extra physical RAM. Linux page cache uses reclaimable RAM; `available` includes reclaimable cache. RAM, GPU allocations, KV state, staging, OS overhead and the host page cache compete for capacity. Do not add page cache and process RSS as though they never overlap.
* **Expert-aware storage:** GGUF can mmap weights. A dedicated store or selective `pread` may avoid reading unwanted pages and duplicate VRAM/RAM entries, but can add copying, double caching, fragmentation, metadata and startup costs. Direct I/O requires alignment, bypasses the kernel cache and needs its own eviction and error handling. Compare bytes per generated token and major faults, not only file size.
* **Caching/pinning:** Keep useful routed experts near compute and consider RAM/VRAM exclusivity; pinned host memory is not free and can create OS pressure. Cache hit rate must be measured by layer and bytes as well as count. The [MoE-Infinity paper](https://arxiv.org/abs/2401.14361) motivates trace-aware expert reuse; its measured speedups are not this PC's results.
* **Prefetch and async I/O:** Request exactly selected experts as soon as routing permits, overlap reads and CPU/GPU work, then test bounded predictive lookahead only where hits exceed wasted I/O and eviction. A wrong prediction can slow demand reads. Queue depth, request size and thread count need measurements on the actual SSD and filesystem. Prefetch hides latency, it does not create bandwidth.
* **Compression:** Already-quantized expert tensors may compress poorly with generic lossless codecs. Per-tensor quantization and low-rank approximations can reduce reads but can change model outputs and quality. Measure perplexity, task accuracy, exact token comparisons where appropriate, and decompression/compute cost. Do not silently skip experts.
* **Filesystem:** [Microsoft recommends](https://learn.microsoft.com/en-us/windows/wsl/filesystems) Linux files in WSL's own filesystem for Linux workloads. `/mnt/d` versus ext4-on-VHDX must be benchmarked on the same physical SSD with available free capacity; a copy of a 92 GiB model consumes another 92 GiB and is not automatically faster. Windows-native llama.cpp is a separate baseline, with different cache and runtime behavior.
* **Throughput bound:** Approximate storage contribution is uncached bytes per token divided by sustained effective read bytes/second, plus unhidden latency and compute. Device active time and sequential SSD specifications do not establish actual request size or bandwidth. The earlier 78 MB/s reading and 72 MB/s writing had unverified attribution; competing writes, WSL and page cache may matter. No speedup is promised.

## Reproducible protocol

Use the same GGUF shards, prompt, output length, seed, context, threads, GPU layers, quant, exact routing, GPU memory headroom, WSL RAM cap, and machine load. Run at least three independent trials, alternate order, distinguish cold-ish from warm cache, and record startup and first-token latency, prefill and decode tok/s, read bytes/token, GPU occupancy, peak RAM, page faults, and cache activation/hit counters. A prompt cache must be disabled or held constant. Do not drop system caches globally or move/delete the user's model as part of a benchmark. Process I/O counters in WSL are not necessarily physical disk I/O: collect host disk data separately. Confirm output token/quality equivalence when changing execution.

1. Record a Windows-native baseline and a WSL baseline separately, then compare only under equivalent conditions. The earlier 1.5–1.8 versus 0.32 tok/s results are historical and uncontrolled.
2. On the same WSL binary and model, compare `-ngl 99`, then smaller values such as 10, 20 and 30 with `-cmoe --moe-cache-mib auto`; inspect whether a cache actually activates and the resulting CPU/GPU split. These layer counts are candidates, not recommendations or verified optima.
3. Only after finding a viable cache, compare automatic store on/off and fault settings with the same cache budget and order. Validate the fork's independent setting fix with explicit `LLAMA_MOE_FAULT_AUTO=0` or `LLAMA_MOE_FAULT_THREADS`, and check both logs and output correctness.
4. Test prefetch off/on and bounded settings, retaining cache/traffic counters. Test mmap against an expert store and ext4 copy only if capacity allows. Record no-hit and adverse results.
5. Evaluate V4.1 and GLM-5.3 separately using documented model files, supported conversions, smoke tests and quality checks. No weights are downloaded by this project.

`tools/lowram_bench.py` starts a loopback-only server, writes raw logs and JSON, requests a fixed prompt and deterministic sampling, records Linux process read_bytes deltas and terminates its own server. Example:

```sh
python3 tools/lowram_bench.py --server /path/to/llama-server --model /path/to/first-shard.gguf --out /tmp/lowram-run-01 --tokens 32 --repeats 2 -- -ngl 10 -cmoe --moe-cache-mib auto -c 1024 -np 1 --cache-ram 0
```

Do not use simultaneous compilation or other heavy workloads during performance comparisons. It does not flush OS cache, measure Windows physical I/O, guarantee a cold start, or ensure identical outputs between different engines.
