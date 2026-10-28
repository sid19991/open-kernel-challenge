# Routed experts

Source: https://openkernelchallenge.dev/problems/muse_moe_w4a16
Fetched: 2026-10-28T14:09:04Z

MoE

## Summary

Top-8 of 256 experts at width 4096: SwiGLU, NVFP4 weights, and a norm on each expert’s output before it is weighted and summed.

## Computation

```
# per token: its top 8 of 256 experts; only local ones here
for t, (e, g) in routes if e is local:
    h = x[t] @ dequant(w13[e]).T
    gate, up = split(h)
    y = (silu(gate) * up) @ dequant(w2[e]).T
    out[t] += g * rmsnorm(y * (1 + post_norm_w))
```

## Signature

```
kernel_fn(data: Inputs, state)
```

## Returns

bfloat16 [M, H]

## Shape notes

- bs (M in the shapes) is the tokens per step: 1, 16 and 32 for decode, and 8, 128 and 256 for 8-token verify blocks. The weights arrive once, in prepare(), untimed.
- This GPU holds 32 of the 256 experts, as under 8-way expert parallelism. Every token arrives, and only its routes to local experts are computed.
- Routings follow real routers’ skewed expert popularity. A step waits for the busiest GPU, so each routing makes this GPU the busiest.
- Decode loads are SGLang serving with speculative decoding off; verify loads are its default recipe, which checks each request's 8-token draft in one step.

## Correctness

[
  "The output, against an FP32 reference: at most 1.5%."
]

## SGLang baseline

- time: 13448.5 µs per decode step
- GPU: H200

## Scored shapes (from API `benchmarks`)

```json
[
  {
    "bs": 1,
    "sharding": "ep8",
    "load": "decode 1",
    "weight": 60,
    "seed": 401
  },
  {
    "bs": 16,
    "sharding": "ep8",
    "load": "decode 16",
    "weight": 60,
    "seed": 402
  },
  {
    "bs": 32,
    "sharding": "ep8",
    "load": "decode 32",
    "weight": 60,
    "seed": 403
  },
  {
    "bs": 8,
    "sharding": "ep8",
    "load": "verify 1",
    "weight": 60,
    "seed": 404
  },
  {
    "bs": 128,
    "sharding": "ep8",
    "load": "verify 16",
    "weight": 60,
    "seed": 405
  },
  {
    "bs": 256,
    "sharding": "ep8",
    "load": "verify 32",
    "weight": 60,
    "seed": 406
  }
]
```
