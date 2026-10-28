# Muse Spark Open Kernel Challenge — local materials

Offline copy of everything needed to start [openkernelchallenge.dev](https://openkernelchallenge.dev/) as of **2026-10-28**. This is not an app. It is the site, CLI, starter kernels, pinned SGLang, API dumps, papers, licenses, and indexes of assets that are gated or too large to store here.

Fetched from the public site, `/api/v1`, `okc` 0.7.2, and the official git remotes. See [MANIFEST.md](MANIFEST.md) for every URL, fetch date, and what is missing.

## How to start (kernels)

The kernel track is open through **4 Nov 2026, 11:59 pm PT**. You need an NVIDIA GPU to iterate locally; official judging is **one H200 SXM** inside `lmsysorg/sglang:dev-muse-spark-open`.

1. Use the bundled CLI (Linux amd64 already unpacked):

   ```bash
   chmod +x cli/0.7.2/okc-linux-amd64
   export PATH="$PWD/cli/0.7.2:$PATH"
   # or: gunzip -k cli/0.7.2/okc-linux-amd64.gz && mv cli/0.7.2/okc-linux-amd64 ~/.okc/bin/okc
   ```

   Other platforms: gunzip `cli/0.7.2/okc-{linux,darwin}-{amd64,arm64}.gz` (checksums in `cli/0.7.2/SHA256SUMS`). Official installer: `cli/install.sh`.

2. Open a problem under `starter/`. Each directory is exactly what `okc init <problem>` writes:

   | Directory | What it computes |
   |---|---|
   | `starter/muse_bf16_proj/` | Dense BF16 projections + FP32 router |
   | `starter/muse_attention_decode/` | Gated attention decode over a paged FP8 KV cache |
   | `starter/muse_moe_w4a16/` | Top-8 of 256 NVFP4 experts (W4A16) |
   | `starter/muse_decode_layer/` | Whole decoder layer, decode shapes (TP8/EP8) |
   | `starter/muse_prefill_layer/` | Whole decoder layer, prefill shapes |

   In each: `submission.py` is your kernel (keep the `# okc: problem=...` first line), `baseline.py` is SGLang’s, `reference.py` / `checks.py` are the judge’s spec, `TASK.md` is the problem, `AGENTS.md` is for a coding agent. NVFP4 problems also have `nvfp4.py`.

3. A person must run `okc login` in a terminal (accept rules + GitHub device flow) before submit. Agents must not accept the rules for anyone. Then:

   ```bash
   okc check muse_moe_w4a16/submission.py      # unscored, own daily cap
   okc submit muse_moe_w4a16/submission.py     # scored
   okc leaderboard muse_moe_w4a16
   ```

   Caps (today, from the API): 10 scored + 20 check runs per problem per day, reset 00:00 UTC.

The judge image has PyTorch, Triton, FlashInfer, DeepGEMM, TileLang, CuTe DSL, CUTLASS. No network in the job. Kernels must be CUDA-graph capturable. Full restrictions: `docs/pages/rules.md`, `docs/llms.txt`, each `TASK.md`.

## How to start (serving)

Opens **5 Nov 2026, noon PT**. Prize money is paid on this board only.

```bash
# already done: okc serve init → serving/
# edit serving/sglang (only python/sglang source + JSON, not model_loader/)
okc serve diff                    # writes serving/patch.diff
okc serve check serving/patch.diff
okc serve submit serving/patch.diff
```

Pinned SGLang is `serving/sglang` at commit `a55ab7d5` (“Support Muse Spark Open”). Patch rules: `serving/RULES.md`. Launch flags: `serving/serving-api.json`.

## Tree

```
README.md                 this file
MANIFEST.md               source URLs, dates, missing/gated/too-large
cli/                      install.sh + okc 0.7.2 binaries and SHA256SUMS
starter/                  okc init output for all five kernel problems
serving/                  okc serve init: RULES.md, patch.diff, pinned sglang, docker metadata
docs/
  llms.txt                official agent dump (rules, CLI, HTTP API)
  terms.pdf               Paradigm official contest terms
  api/                    public /api/v1 JSON + okc --json dumps
  pages/                  homepage, rules, leaderboard, per-problem copy
  cli/                    okc --help for every command
  external/               Meta / Paradigm / Hugging Face / SGLang cookbook pages
site/                     SPA shell, JS/CSS bundle, favicons, opengraph image
reference/                gpu-mode/reference-kernels and kernelbot (credited judge starting point)
papers/                   eval methodology HTML + Muse Spark safety/preparedness PDF
licenses/                 Apache-2.0, Muse Model License, SGLang, GPU MODE
data/hf/                  Hugging Face weights indexes (not the weights)
```

## API (no token)

Base: `https://openkernelchallenge.dev/api/v1` — snapshots in `docs/api/`.

Public: `GET /problems`, `/problems/{name}` (includes starter files), `/leaderboard`, `/leaderboard/{name}`, `/queue`, `/rules`, `/serving`, `/serving/leaderboard`.

Auth-walled (need `okc login`): `/me`, `/jobs`, `/tokens`, `POST` submissions, `POST /rules/accept`. Documented in `docs/llms.txt` and `docs/cli/okc-api.txt`.

## What this tree is not

- Not a clone of competitor submissions.
- Not the H200 judge, not the 8×H200 serving node, not the hidden serving conversations.
- Not the Muse Spark Open weight shards (~167 GB NVFP4 / ~1 TB BF16) or the ~23.9 GB Docker image — indexes and digests only. See MANIFEST.md.
EOF