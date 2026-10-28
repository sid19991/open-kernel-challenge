# Muse Spark Open Kernel Challenge

Source: https://openkernelchallenge.dev/
Fetched: 2026-10-28T14:09:04Z

Make Meta’s Muse Spark Open run faster. Start by optimizing its GPU kernels, then speed up how SGLang serves it. The fastest win $100,000 in prizes.

## Install the CLI (macOS, Linux)

```
curl -fsSL https://openkernelchallenge.dev/install.sh | sh
```

A local copy is at `cli/install.sh`. Binaries for 0.7.2 are under `cli/0.7.2/`.

## Kernel Challenges

Oct 22, 11:00 am – Nov 4, 11:59 pm PT

Problems:

- muse_bf16_proj
- muse_attention_decode
- muse_moe_w4a16
- muse_decode_layer
- muse_prefill_layer

## How Kernels Are Judged

1. Submission — write `kernel_fn` in the submission file. Start from the PyTorch starter and use whatever SGLang uses (Triton, CUTLASS, FlashInfer, DeepGEMM) or write your own CUDA.
2. Correctness — tested on random unseen inputs; output must match a reference within the problem’s tolerance.
3. Timing — your kernel and SGLang’s take turns on the same H200 with the same inputs, each run as a CUDA graph.
4. Ranking — timed shapes come from SGLang serving agentic coding conversations, from one user up to 32. Score is speedup vs SGLang.

## Serving Track

Nov 5, noon – Nov 19, 11:59 pm PT

Each submission uses a whole 8×H200 node. Top kernel competitors get queue priority. Prize money is paid on the serving board.

1. Submission — modify SGLang and submit the diff.
2. Workload — hidden multi-turn agentic coding conversations at 1, 4, 16 and 32 concurrent users.
3. Correctness — agree with stock SGLang nearly as well as it agrees with itself.
4. Score — throughput (tokens/s per GPU) and interactivity (tokens/s per user) at each load; how far the curve sits above SGLang’s, averaged across loads.

## Prizes

Top contestants may be given the opportunity to interview for a role at Meta.

1. First — $50,000 and a signed H200
2. Second — $30,000
3. Third — $10,000
4. Fourth — $3,000
5. Fifth — $2,000
6. Sixth to tenth — $1,000 each

The kernel leaderboards carry no prizes.

## SPA note

The website is a single-page app (`site/index.html` + `site/assets/`). Routes such as `/faq`, `/timeline`, `/scoring`, `/team`, `/licenses`, `/datasets`, `/papers`, `/submission` return the same HTML shell; they are not separate documents. Content for rules, problems, leaderboard, serving, and API lives in the JS bundle, `llms.txt`, and `/api/v1/*`.
