# Prefill layer

Source: https://openkernelchallenge.dev/problems/muse_prefill_layer
Fetched: 2026-10-28T14:09:04Z

Prefill

## Summary

The whole decoder layer at the prefill shapes of an agentic coding turn: 753 new tokens on an 80K-token conversation, four conversations at once, and the first turn’s 16K-token chunks. Most of the time goes to the routed experts, as one large grouped NVFP4 GEMM.

## Computation

```
# x: attention input [T, 8192] BF16   s: residual stream FP32
q, k, v, g = split(x @ qkv_gate.T);  QK-norm, RoPE, write fp8(k), fp8(v)
a = (rmsnorm(attention(q)) * sigmoid(g)) @ o.T     # all-reduce: identity
s = α1 * s + β1 * rmsnorm(a);  h = rmsnorm(s) * (1 + w)
experts = top8(sigmoid(0.177 * h @ router.T) + bias)
y = local experts(rmsnorm(h @ pooler.T) * (1 + w))  # all-reduce: identity
f = y @ unpooler.T   # T ≤ 8: [y @ unpooler[:1024].T, 0, …, 0]
s = α2 * s + β2 * rmsnorm(f * (1 + w))
return rmsnorm(s) * (1 + w), s, experts
```

## Signature

```
kernel_fn(data: Inputs, state)
```

## Returns

(bfloat16 [T, 8192], float32 [T, 8192], int32 [T, 8])

## Shape notes

- Each case is one prefill forward: B requests, each adding L new tokens (causal within the block) after seq_lens[b] cached ones, written into the paged FP8 cache.
- Inputs, outputs, bounds and the routing check are muse_decode_layer’s. The cache’s page size is 1, except one test with 64-token pages, so read config["page_size"].
- Prefill isn’t part of a decode step, so this problem has its own board and doesn’t count toward the overall board.

## SGLang baseline

- time: 387403.9 µs per prefill load
- GPU: H200

## Scored shapes (from API `benchmarks`)

```json
[
  {
    "batch": 1,
    "ctx": 80000,
    "layer": "sliding",
    "q_len": 753,
    "page": 1,
    "dist": "soft",
    "load": "turn 1",
    "weight": 46,
    "repeats": 6,
    "seed": 1001
  },
  {
    "batch": 1,
    "ctx": 80000,
    "layer": "full",
    "q_len": 753,
    "page": 1,
    "dist": "soft",
    "load": "turn 1",
    "weight": 16,
    "repeats": 6,
    "seed": 1002
  },
  {
    "batch": 4,
    "ctx": 80000,
    "layer": "sliding",
    "q_len": 753,
    "page": 1,
    "dist": "soft",
    "load": "turn 4",
    "weight": 46,
    "repeats": 6,
    "seed": 1003
  },
  {
    "batch": 4,
    "ctx": 80000,
    "layer": "full",
    "q_len": 753,
    "page": 1,
    "dist": "soft",
    "load": "turn 4",
    "weight": 16,
    "repeats": 6,
    "seed": 1004
  },
  {
    "batch": 1,
    "ctx": 0,
    "layer": "full",
    "q_len": 16384,
    "page": 1,
    "dist": "soft",
    "load": "first turn",
    "weight": 16,
    "repeats": 6,
    "seed": 1006
  },
  {
    "batch": 1,
    "ctx": 32768,
    "layer": "sliding",
    "q_len": 16384,
    "page": 1,
    "dist": "soft",
    "load": "first turn",
    "weight": 138,
    "repeats": 6,
    "seed": 1007
  },
  {
    "batch": 1,
    "ctx": 32768,
    "layer": "full",
    "q_len": 16384,
    "page": 1,
    "dist": "soft",
    "load": "first turn",
    "weight": 16,
    "repeats": 6,
    "seed": 1008
  },
  {
    "batch": 1,
    "ctx": 65536,
    "layer": "full",
    "q_len": 16384,
    "page": 1,
    "dist": "soft",
    "load": "first turn",
    "weight": 16,
    "repeats": 6,
    "seed": 1010
  }
]
```
