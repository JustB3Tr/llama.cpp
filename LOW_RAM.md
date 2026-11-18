# Low-RAM MoE experiments

This branch carries a local tandem.cpp patch on top of a pinned llama.cpp base, plus a launcher fix, an independent RAM-store selection fix, a benchmark runner, and initial hardware tests. It is experimental; see [research and compatibility](docs/lowram-research.md) and [measured results](docs/lowram-first-run.md).

On the test PC, a clean WSL CUDA build was copied to `/mnt/c/projects/llama-lowram-bin`. Use the fork launcher to inspect a proposed command without starting a server:

```sh
cd /mnt/c/projects/llama-lowram
LD_LIBRARY_PATH=/mnt/c/projects/llama-lowram-bin python3 tools/lowram_tandem_run.py \
  --server-bin /mnt/c/projects/llama-lowram-bin/llama-server \
  --model /path/to/first-shard.gguf --gpu-layers 25 --ctx 1024 --dry-run --no-calibrate
```

Remove `--dry-run` only when you intend to start the server. Layer 25 is a tested candidate, not a proven optimum. The targeted behavior tested here additionally used `LLAMA_MOE_FAULT_AUTO=0`; otherwise tandem's default automatic fault behavior applies. Expect a large RAM arena when the store is activated. Verify logs and consider other workloads before running. Do not publish a performance claim from the initial short, ordered trials.

Run focused tests with `python3 -m unittest discover -s tests -p 'test_lowram_*.py' -v`. For build settings and the limits of the current tests, see the linked notes.
