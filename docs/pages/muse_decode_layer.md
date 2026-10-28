# Whole decoder layer

Source: https://openkernelchallenge.dev/problems/muse_decode_layer
Fetched: 2026-10-28T14:09:04Z

Layer

## Summary

Everything one GPU does for one layer of Muse Spark Open under TP8/EP8: the projections, attention, the router and experts, and every norm and residual in between.

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

- T is the tokens per step, batch × q_len (B × L). Each load times a sliding-window and a full-attention layer, at contexts around 80K tokens.
- Communication is left out: each all-reduce and all-gather is the identity on this GPU’s partial result, as if the other seven GPUs contributed nothing.
- `scheduler_metadata` is FA3's tile schedule for the step, built from its lengths as SGLang does: set on full-attention decode steps, None on sliding layers and verify blocks. Kernels that don't use FA3 can ignore it.
- Routing is skewed, as in the real model; a few experts per layer take most tokens.
- Decode loads are SGLang serving with speculative decoding off; verify loads are its default recipe, which checks each request's 8-token draft in one step.

## Correctness

[
  "The residual update (output stream \u2212 \u03b11\u00b7\u03b12 \u00b7 input stream) and the next attention input, against an FP32 reference: within 2.5\u00d7 the reference's own error when its queries are rounded to FP8, which comes to 3\u201315% depending on the shape.",
  "The new cache entries: within 8% of the reference's."
]

## Routing

[
  "Each expert you pick must score (sigmoid + router_bias) within 0.05 of the reference's 8th best.",
  "Each of the reference's top 8 on this GPU that beats its 9th best by more than 0.03 (0.04 above 256 tokens) must be among yours.",
  "This GPU must get at least the reference's routes minus 8% and 4, and its distinct experts minus 15% and 2.",
  "The MoE output is then checked with the experts you picked."
]

## SGLang baseline

- time: 22620.3 µs per decode step
- GPU: H200

## Scored shapes (from API `benchmarks`)

```json
[
  {
    "batch": 1,
    "ctx": 80000,
    "layer": "sliding",
    "page": 1,
    "dist": "soft",
    "load": "decode 1",
    "weight": 46,
    "repeats": 60,
    "seed": 801
  },
  {
    "batch": 1,
    "ctx": 80000,
    "layer": "full",
    "page": 1,
    "dist": "soft",
    "load": "decode 1",
    "weight": 16,
    "repeats": 60,
    "seed": 802
  },
  {
    "batch": 16,
    "ctx": 80000,
    "layer": "sliding",
    "page": 1,
    "spread": "agentic",
    "dist": "soft",
    "load": "decode 16",
    "weight": 46,
    "seed": 803
  },
  {
    "batch": 16,
    "ctx": 80000,
    "layer": "full",
    "page": 1,
    "spread": "agentic",
    "dist": "soft",
    "load": "decode 16",
    "weight": 16,
    "seed": 804
  },
  {
    "batch": 32,
    "ctx": 80000,
    "layer": "sliding",
    "page": 1,
    "spread": "agentic",
    "dist": "soft",
    "load": "decode 32",
    "weight": 46,
    "seed": 805
  },
  {
    "batch": 32,
    "ctx": 80000,
    "layer": "full",
    "page": 1,
    "spread": "agentic",
    "dist": "soft",
    "load": "decode 32",
    "weight": 16,
    "seed": 806
  },
  {
    "batch": 1,
    "ctx": 80000,
    "layer": "sliding",
    "page": 1,
    "q_len": 8,
    "dist": "soft",
    "load": "verify 1",
    "weight": 46,
    "seed": 807
  },
  {
    "batch": 1,
    "ctx": 80000,
    "layer": "full",
    "page": 1,
    "q_len": 8,
    "dist": "soft",
    "load": "verify 1",
    "weight": 16,
    "seed": 808
  },
  {
    "batch": 16,
    "ctx": 80000,
    "layer": "sliding",
    "page": 1,
    "q_len": 8,
    "spread": "agentic",
    "dist": "soft",
    "load": "verify 16",
    "weight": 46,
    "seed": 809
  },
  {
    "batch": 16,
    "ctx": 80000,
    "layer": "full",
    "page": 1,
    "q_len": 8,
    "spread": "agentic",
    "dist": "soft",
    "load": "verify 16",
    "weight": 16,
    "seed": 810
  },
  {
    "batch": 32,
    "ctx": 80000,
    "layer": "sliding",
    "page": 1,
    "q_len": 8,
    "spread": "agentic",
    "dist": "soft",
    "load": "verify 32",
    "weight": 46,
    "seed": 811
  },
  {
    "batch": 32,
    "ctx": 80000,
    "layer": "full",
    "page": 1,
    "q_len": 8,
    "spread": "agentic",
    "dist": "soft",
    "load": "verify 32",
    "weight": 16,
    "seed": 812
  }
]
```
