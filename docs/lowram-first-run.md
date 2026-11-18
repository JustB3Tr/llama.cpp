# First hardware observations

2026-11-17, RTX 4060 8 GiB, i7-11700K, WSL2 Linux 6.6.114.1, ~39 GiB WSL RAM. DeepSeek V4 Flash Q2_K_S, two GGUF shards totaling 98,592,510,368 bytes, read through `/mnt/d`. Existing local tandem server built from llama.cpp f5222f25c plus the local tandem patch was used, not a newly compiled binary from this branch. Each row is one request with an 11-token prompt, 12 requested output tokens, context 1024, prompt cache disabled, greedy temperature, and 8 CPU threads. Both used `-cmoe --moe-cache-mib auto`.

| GPU layers | Startup | Prompt | Decode | Process read_bytes during request | Observation |
|---:|---:|---:|---:|---:|---|
| 99 | 69.1 s | 0.261 tok/s | 0.436 tok/s | 20,080,005,120 | Explicit warning: fewer than 256 MiB available, MoE cache disabled |
| 10 | 46.1 s | 0.289 tok/s | 0.407 tok/s | 19,790,413,824 | No cache-disabled warning, but logging was insufficient to prove cache activation |

This is a diagnostic pair, **not a valid speedup claim**: the second run followed the first without controlling the OS page cache; one tiny request per configuration is noisy; GPU layer placement changes computation; process read bytes include prompt and decode and are not verified physical device bytes. The generated text differed, so token equivalence has not been demonstrated. A normal chat prompt, longer generation, activation logs, and alternating repetitions are needed. Previous user-reported Windows and WSL results used different setups and cannot be treated as controlled comparison.

The server reported `failed to fit params ... n_gpu_layers already set by user`; both layer settings were explicit. At 99 layers it also warned that the MoE cache was disabled, even though VRAM was nearly full. Thus VRAM occupancy alone did not prove an expert cache existed. The warning establishes this run's state only; it does not establish the earlier run's state. At 10 layers the absence of the warning is not positive proof of a working cache.

Raw `result.json` and `server.log` for both runs were retained outside the repository on the test machine in `C:\projects\llama-lowram-bench`. They include local file paths. Subsequent trials should raise log verbosity and/or use tandem's `LLAMA_MOE_STATS_FILE` to confirm the cache and measure hit/read counters, then test exactness with identical prompts and deterministic settings. Record competing Windows disk activity separately. Startup and decode of the changed fork remain untested until its build and separate inference validation complete.
