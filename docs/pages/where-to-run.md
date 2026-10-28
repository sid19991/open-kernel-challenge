# Where to run kernels

Fetched / checked: 2026-10-28.

## Short answer

**Time and correctness-test on the official OKC H200 cloud**, not on this Cursor VM and not on a MacBook.

This machine is a CPU-only Ubuntu box (4 cores, 16 GB RAM, no NVIDIA device, no Docker, no PyTorch, no nvcc). The challenge’s baseline kernels import `sglang` / `sgl_kernel` (cuBLAS, Marlin W4A16, FA3) and must be CUDA-graph capturable on **one NVIDIA H200 SXM** inside `lmsysorg/sglang:dev-muse-spark-open`. Scores from any other GPU, or from CPU, do not transfer.

The judge is already the cloud: `okc check` (unscored, 20/day) and `okc submit` (scored, 10/day), 112 H200s in the kernel pool when last polled, backlog 0.

## What this VM is good for

- Editing `starter/*/submission.py`
- Reading `TASK.md`, `reference.py`, `baseline.py`, and the pinned SGLang tree
- Running `python3 scripts/static_check.py` (syntax, `kernel_fn`/`prepare`, graph-breaker greps)
- Shipping a file to the H200 judge with `okc`

It is **not** a place to install the 24 GB judge image, pull NVFP4 weights, or iterate on CUDA occupancy.

## What is not a substitute

| Machine | Why not |
|---|---|
| This Cursor Cloud Agent | No GPU, no CUDA, no Docker |
| Siddharth’s MacBook Air (self-hosted worker) | darwin, not Hopper; no H200 |
| A random CUDA GPU (4090, L40, etc.) | Different memory bandwidth / Tensor Core generation; eager timing also lies vs CUDA graphs |
| CPU PyTorch | BF16 GEMMs and NVFP4 Marlin will not match; graph rules do not apply |

A local Hopper box (H100/H200) with the official image can be used for **iteration**, still confirm on `okc check` before treating a number as real.

## Workflow

1. Pick a problem under `starter/` (start with `muse_bf16_proj` or `muse_moe_w4a16`; they are smaller than the full layer).
2. Edit `submission.py`. Keep the first line `# okc: problem=...`.
3. `python3 scripts/static_check.py starter/<problem>/submission.py`
4. A person runs `okc login` once (rules + GitHub device flow). An agent cannot accept the rules. Then, in a sandbox, export `OKC_TOKEN` from `okc api token`.
5. `okc check starter/<problem>/submission.py --json` — same tests and H200 timing, off the board.
6. After a pass, read `starter/<problem>/profiles/submission.<job>.txt` for where time went.
7. `okc submit starter/<problem>/submission.py` when you want a scored run.

Serving track (opens 5 Nov, noon PT) is the same idea on a whole 8×H200 node — even less suitable for this VM.
