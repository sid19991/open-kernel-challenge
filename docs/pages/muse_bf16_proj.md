# Dense projections

Source: https://openkernelchallenge.dev/problems/muse_bf16_proj
Fetched: 2026-10-28T14:09:04Z

Proj

## Summary

The attention and MoE-block projections and the router, as one GPU runs them serving Muse Spark Open on 8×H200. They are BF16 (the router FP32).

## Computation

```
# x: [m, k] BF16   W: [n, k] = weights[layer]
# qkv_gate, o, pooler, unpooler: BF16; router: FP32
out = x @ W.T  # [m, n], BF16 (router: FP32)
```

## Signature

```
kernel_fn(data: Inputs, state)
```

## Returns

bfloat16 / float32 [m, n]

## Shape notes

- m is the tokens per step: 1, 16 and 32 for decode, and 8, 128 and 256 for 8-token verify blocks. The weights arrive once, in prepare(), untimed.
- Decode loads are SGLang serving with speculative decoding off; verify loads are its default recipe, which checks each request's 8-token draft in one step.

## Correctness

[
  "The output, against an FP64 reference: at most 1%, and 0.01% for the router, whose outputs choose the experts. Splitting the router's FP32 weights into a BF16 hi + lo pair with FP32 accumulation, as SGLang does above 8 tokens, passes at any m; BF16 router weights alone don't.",
  "Use the weights at their own precision (the router's hi + lo split aside). About one row in eight of x has a single nonzero entry, a power of two, so its output is a column of the weights, exact with the original weights and slightly off with the router's hi + lo split. Those rows must be within 0.1%."
]

## SGLang baseline

- time: 4830.8 µs per decode step
- GPU: H200

## Scored shapes (from API `benchmarks`)

```json
[
  {
    "m": 1,
    "layer": "qkv_gate",
    "load": "decode 1",
    "weight": 62,
    "seed": 201
  },
  {
    "m": 1,
    "layer": "o",
    "load": "decode 1",
    "weight": 62,
    "seed": 202
  },
  {
    "m": 1,
    "layer": "pooler",
    "load": "decode 1",
    "weight": 62,
    "seed": 203
  },
  {
    "m": 1,
    "layer": "unpooler",
    "load": "decode 1",
    "weight": 62,
    "seed": 204
  },
  {
    "m": 1,
    "layer": "router",
    "load": "decode 1",
    "weight": 62,
    "seed": 205
  },
  {
    "m": 16,
    "layer": "qkv_gate",
    "load": "decode 16",
    "weight": 62,
    "seed": 206
  },
  {
    "m": 16,
    "layer": "o",
    "load": "decode 16",
    "weight": 62,
    "seed": 207
  },
  {
    "m": 16,
    "layer": "pooler",
    "load": "decode 16",
    "weight": 62,
    "seed": 208
  },
  {
    "m": 16,
    "layer": "unpooler",
    "load": "decode 16",
    "weight": 62,
    "seed": 209
  },
  {
    "m": 16,
    "layer": "router",
    "load": "decode 16",
    "weight": 62,
    "seed": 210
  },
  {
    "m": 32,
    "layer": "qkv_gate",
    "load": "decode 32",
    "weight": 62,
    "seed": 211
  },
  {
    "m": 32,
    "layer": "o",
    "load": "decode 32",
    "weight": 62,
    "seed": 212
  },
  {
    "m": 32,
    "layer": "pooler",
    "load": "decode 32",
    "weight": 62,
    "seed": 213
  },
  {
    "m": 32,
    "layer": "unpooler",
    "load": "decode 32",
    "weight": 62,
    "seed": 214
  },
  {
    "m": 32,
    "layer": "router",
    "load": "decode 32",
    "weight": 62,
    "seed": 215
  },
  {
    "m": 8,
    "layer": "qkv_gate",
    "load": "verify 1",
    "weight": 62,
    "seed": 216
  },
  {
    "m": 8,
    "layer": "o",
    "load": "verify 1",
    "weight": 62,
    "seed": 217
  },
  {
    "m": 8,
    "layer": "pooler",
    "load": "verify 1",
    "weight": 62,
    "seed": 218
  },
  {
    "m": 8,
    "layer": "unpooler",
    "load": "verify 1",
    "weight": 62,
    "seed": 219
  },
  {
    "m": 8,
    "layer": "router",
    "load": "verify 1",
    "weight": 62,
    "seed": 220
  },
  {
    "m": 128,
    "layer": "qkv_gate",
    "load": "verify 16",
    "weight": 62,
    "seed": 221
  },
  {
    "m": 128,
    "layer": "o",
    "load": "verify 16",
    "weight": 62,
    "seed": 222
  },
  {
    "m": 128,
    "layer": "pooler",
    "load": "verify 16",
    "weight": 62,
    "seed": 223
  },
  {
    "m": 128,
    "layer": "unpooler",
    "load": "verify 16",
    "weight": 62,
    "seed": 224
  },
  {
    "m": 128,
    "layer": "router",
    "load": "verify 16",
    "weight": 62,
    "seed": 225
  },
  {
    "m": 256,
    "layer": "qkv_gate",
    "load": "verify 32",
    "weight": 62,
    "seed": 226
  },
  {
    "m": 256,
    "layer": "o",
    "load": "verify 32",
    "weight": 62,
    "seed": 227
  },
  {
    "m": 256,
    "layer": "pooler",
    "load": "verify 32",
    "weight": 62,
    "seed": 228
  },
  {
    "m": 256,
    "layer": "unpooler",
    "load": "verify 32",
    "weight": 62,
    "seed": 229
  },
  {
    "m": 256,
    "layer": "router",
    "load": "verify 32",
    "weight": 62,
    "seed": 230
  }
]
```
