---
pipeline_tag: text-generation
tags:
- speculative-decoding
- dflash
- sglang
- safetensors
base_model: meta-models/Muse-Spark-open-816B-A42B-NVFP4
inference: false
---

# Muse-Spark-Open-DFlash2

## Overview

A DFlash2 speculator for
[Muse Spark Open](https://huggingface.co/meta-models/Muse-Spark-open-816B-A42B-NVFP4),
designed to accelerate inference through speculative decoding with SGLang.
The draft proposes multiple tokens in parallel, which the target model verifies.

This repository contains the BF16 draft weights and configuration. The target
model supplies the embeddings, LM head, tokenizer, and native chat template.
The draft is used alongside the target and is not a standalone language model.

## Model Specifications

- **Base model:** `meta-models/Muse-Spark-open-816B-A42B-NVFP4`
- **Format:** Single-file BF16 Safetensors
- **Draft:** 5 Qwen3-style GQA layers, hidden size 8192, 64 query heads / 8 KV heads
- **Attention:** Sliding-window attention with a 2,048-token window
- **Verification width:** 8 tokens (1 current token + 7 draft tokens)
- **Selector:** Rank 256, top-k 16
- **Auxiliary target layers:** `[1, 16, 30, 44, 59]`
- **Configured context length:** 1,048,576 tokens, aligned with the target
- **Target weights:** Embedding and unembedding weights are loaded from the target

The configured context limit is distinct from the attention window. The config
alignment does not establish draft acceptance quality across the full 1M context.

## Evaluation Results

`acc_len` is the per-request acceptance length `L = C / V`, where `C` is the
number of completion tokens and `V` is the number of verification steps, averaged
equally across successful requests. Completion-token-weighted AL is
`sum(C * L) / sum(C)`; pooled AL is `sum(C) / sum(V)`.

| Dataset | Samples | acc_len | Completion-token-weighted AL | Pooled AL |
|---|---:|---:|---:|---:|
| GSM8K | 1,319 | **5.370116** | 5.133438 | 5.067487 |
| AIME26 | 240 (30 × 8) | **4.194966** | 3.942215 | 3.891917 |
| GPQA Diamond | 198 | **3.879590** | 3.614902 | 3.587121 |
| MindCube | 1,050 | **3.253494** | 3.213286 | 3.206133 |
| Video-MME short | 900 | **3.743335** | 3.393189 | 3.340400 |

Measured with real DFlash8 and **block verification enabled**, checkpoint-default
sampling (`temperature=1.0`, `top_p=1.0`, `top_k=64`), and `xhigh` reasoning.
The run used two independent ARM64 GB300 replicas, each with TP4/EP4, FP8 target
KV cache, and BF16 draft KV cache. Suites ran concurrently with 16–64 workers per
suite; these results are not from a single-request-at-a-time evaluation.

All 3,707 samples completed without truncation. Acceptance statistics include
two successful SDK retries; the two initial failed HTTP attempts had no model
counters and are excluded from AL. Video-MME short used 96 frames per sample.

## Serving with SGLang

The following example launches one TP4/EP4 replica with block verification:

```bash
python -m sglang.launch_server \
  --model-path meta-models/Muse-Spark-open-816B-A42B-NVFP4 \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_trtllm \
  --tp-size 4 \
  --ep-size 4 \
  --attention-backend trtllm_mha \
  --page-size 128 \
  --mem-fraction-static 0.8 \
  --reasoning-parser muse \
  --tool-call-parser muse \
  --enable-layernorm-sp \
  --enable-megamoe-for-layernorm-sp \
  --kv-cache-dtype fp8_e4m3 \
  --enable-tf32-matmul \
  --sampling-defaults model \
  --max-running-requests 128 \
  --context-length 1048576 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path RadixArk/Muse-Spark-Open-DFlash2 \
  --speculative-draft-attention-backend fa4 \
  --speculative-num-draft-tokens 8 \
  --speculative-use-block-verification \
  --host 127.0.0.1 \
  --port 30000
```

Use the target's native chat template and `xhigh` reasoning for the evaluation
settings above. 
