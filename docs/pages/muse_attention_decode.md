# Gated attention decode

Source: https://openkernelchallenge.dev/problems/muse_attention_decode
Fetched: 2026-10-28T14:09:04Z

Attention

## Summary

One decode step or 8-token verify block of grouped-query attention over a paged FP8 cache: QK-norm, a 2,048-token sliding window with RoPE or full attention over the whole context, and a gated, normalized output.

## Computation

```
# per request b and block token j, at position p = positions[t]
q, k, v, gate = split(qkv[t])   # the fused projection output
q = rmsnorm(q) * 5.473;  k = rmsnorm(k)
q, k = rope(q, p), rope(k, p)   # sliding layers
cache[cache_loc[t]] = fp8(k), fp8(v)
a = softmax(q @ K.T / 8) @ V    # up to p: window or all
out = rmsnorm(a) * sigmoid(gate)
```

## Returns

bfloat16 [B, L, Hq·64]

## Shape notes

- B is batch and L is q_len: 1 for a decode step, or 8 for a speculative verify block (causal inside it). Each GPU has 16 query heads and 2 key/value heads (TP8).
- Each load times a sliding-window layer (the last 2,048 tokens, with RoPE) and a full-attention layer (the whole context, no RoPE). Contexts are around 80K tokens.
- The inputs are what SGLang’s attention receives: the fused projection output, plus the step’s positions, cache rows and FA3 lengths, which SGLang builds once per step.
- `scheduler_metadata` is FA3's tile schedule for the step, built from its lengths as SGLang does: set on full-attention decode steps, None on sliding layers and verify blocks. Kernels that don't use FA3 can ignore it.
- The cache is FP8 E4M3. Its page size is 1 in the scored cases and 64 in most tests, so read config["page_size"]. Its scales are 1 in the scored cases and 1/64 in some tests.
- Decode loads are SGLang serving with speculative decoding off; verify loads are its default recipe, which checks each request's 8-token draft in one step.

## Correctness

[
  "The output: within 2.5\u00d7 the reference's own error when its queries are rounded to FP8 E4M3, plus 1.5%, kept between 3% and 15%, against an FP32 reference.",
  "The new cache entries: within 8% of the reference's. Your output is checked against the reference attending to your own new entries.",
  "Some tests put keys at the window's edges.",
  "Each head's keys share a direction, so attention spreads over many keys. On full layers the low-attention keys' values share a direction too."
]

## SGLang baseline

- time: 3766.3 µs per decode step
- GPU: H200

## Scored shapes (from API `benchmarks`)

```json
[
  {
    "batch": 1,
    "ctx": 80000,
    "heads": "tp8",
    "layer": "sliding",
    "page": 1,
    "dist": "soft",
    "load": "decode 1",
    "weight": 46,
    "seed": 601
  },
  {
    "batch": 1,
    "ctx": 80000,
    "heads": "tp8",
    "layer": "full",
    "page": 1,
    "dist": "coherent",
    "load": "decode 1",
    "weight": 16,
    "seed": 602
  },
  {
    "batch": 16,
    "ctx": 80000,
    "heads": "tp8",
    "layer": "sliding",
    "page": 1,
    "spread": "agentic",
    "dist": "soft",
    "load": "decode 16",
    "weight": 46,
    "seed": 603
  },
  {
    "batch": 16,
    "ctx": 80000,
    "heads": "tp8",
    "layer": "full",
    "page": 1,
    "spread": "agentic",
    "dist": "coherent",
    "load": "decode 16",
    "weight": 16,
    "seed": 604
  },
  {
    "batch": 32,
    "ctx": 80000,
    "heads": "tp8",
    "layer": "sliding",
    "page": 1,
    "spread": "agentic",
    "dist": "soft",
    "load": "decode 32",
    "weight": 46,
    "seed": 605
  },
  {
    "batch": 32,
    "ctx": 80000,
    "heads": "tp8",
    "layer": "full",
    "page": 1,
    "spread": "agentic",
    "dist": "coherent",
    "load": "decode 32",
    "weight": 16,
    "seed": 606
  },
  {
    "batch": 1,
    "ctx": 80000,
    "heads": "tp8",
    "layer": "sliding",
    "page": 1,
    "q_len": 8,
    "dist": "soft",
    "load": "verify 1",
    "weight": 46,
    "seed": 607
  },
  {
    "batch": 1,
    "ctx": 80000,
    "heads": "tp8",
    "layer": "full",
    "page": 1,
    "q_len": 8,
    "dist": "coherent",
    "load": "verify 1",
    "weight": 16,
    "seed": 608
  },
  {
    "batch": 16,
    "ctx": 80000,
    "heads": "tp8",
    "layer": "sliding",
    "page": 1,
    "q_len": 8,
    "spread": "agentic",
    "dist": "soft",
    "load": "verify 16",
    "weight": 46,
    "seed": 609
  },
  {
    "batch": 16,
    "ctx": 80000,
    "heads": "tp8",
    "layer": "full",
    "page": 1,
    "q_len": 8,
    "spread": "agentic",
    "dist": "coherent",
    "load": "verify 16",
    "weight": 16,
    "seed": 610
  },
  {
    "batch": 32,
    "ctx": 80000,
    "heads": "tp8",
    "layer": "sliding",
    "page": 1,
    "q_len": 8,
    "spread": "agentic",
    "dist": "soft",
    "load": "verify 32",
    "weight": 46,
    "seed": 611
  },
  {
    "batch": 32,
    "ctx": 80000,
    "heads": "tp8",
    "layer": "full",
    "page": 1,
    "q_len": 8,
    "spread": "agentic",
    "dist": "coherent",
    "load": "verify 32",
    "weight": 16,
    "seed": 612
  }
]
```
