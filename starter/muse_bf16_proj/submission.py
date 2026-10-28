# okc: problem=muse_bf16_proj
"""BF16 projections + FP32 router for H200 CUDA-graph replay.

prepare() packs weights once (untimed). kernel_fn is captured as a CUDA graph:
branching on shapes/config only, no host syncs.

Official H200 check 84e26dd270dc4051: 4509.4 µs (1.071x vs SGLang).
- M=1 BF16: Hopper streaming GEMV (~1.08-1.14x)
- skinny N<=1024, M<=32: tiny_gemm_bf16
- other BF16: torch.mm on a contiguous [K, N] weight (NN)
- router M<=8: MuseSpark v1.2 router_gemm (~2.16x at M=8)
- router M>8: BF16 hi+lo pair, FP32 accumulate
"""

from typing import TypedDict

import torch
import torch.nn.functional as F


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


ROUTER_FP32_MAX_TOKENS = 8
UNPOOL_SLICE = 1024
SLICED_MAX_TOKENS = 8
BF16_LAYERS = ("qkv_gate", "o", "pooler", "unpooler")
WARMUP_MS = (1, 5, 7, 8, 9, 16, 32, 33, 128, 130, 256, 512, 1000)
TINY_N_MAX = 1024
TINY_M_MAX = 32

_hopper_gemv = None
_use_hopper_gemv = None
_tiny_gemm = None
_can_use_tiny = None
_router_gemm = None
_router_gemm_ok = None


def _init_kernels() -> None:
    global _hopper_gemv, _use_hopper_gemv
    global _tiny_gemm, _can_use_tiny
    global _router_gemm, _router_gemm_ok
    if _hopper_gemv is not None:
        return
    try:
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import (
            hopper_bf16_gemv,
            use_hopper_bf16_gemv,
        )

        _hopper_gemv = hopper_bf16_gemv
        _use_hopper_gemv = use_hopper_bf16_gemv
    except Exception:
        _hopper_gemv = False
        _use_hopper_gemv = lambda *args, **kwargs: False
    try:
        from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16

        _tiny_gemm = tiny_gemm_bf16
        _can_use_tiny = can_use_tiny_gemm
    except Exception:
        _tiny_gemm = False
        _can_use_tiny = lambda *args, **kwargs: False
    try:
        from sglang.kernels.ops.muse_spark_v12.router_gemm import (
            router_gemm,
            router_gemm_supported,
        )

        _router_gemm = router_gemm
        _router_gemm_ok = router_gemm_supported
    except Exception:
        _router_gemm = False
        _router_gemm_ok = lambda *args, **kwargs: False


def _bf16_w(state, layer: str, n: int) -> torch.Tensor:
    if layer == "unpooler" and n == UNPOOL_SLICE:
        return state["unpooler_slice"]
    w = state[layer]
    if w.shape[0] != n:
        return w[:n]
    return w


def _bf16_wt(state, layer: str, n: int) -> torch.Tensor:
    if layer == "unpooler" and n == UNPOOL_SLICE:
        return state["unpooler_slice_t"]
    wt = state[layer + "_t"]
    if wt.shape[1] != n:
        return wt[:, :n]
    return wt


def _use_tiny(n: int, k: int, m: int) -> bool:
    return bool(
        _tiny_gemm
        and 1 < m <= TINY_M_MAX
        and n <= TINY_N_MAX
        and _can_use_tiny(n, k, TINY_M_MAX)
    )


def _bf16_forward(x: torch.Tensor, w: torch.Tensor, wt: torch.Tensor, m: int) -> torch.Tensor:
    n, k = w.shape
    if m == 1 and _hopper_gemv and _use_hopper_gemv(1, n, k):
        return _hopper_gemv(x, w)
    if _use_tiny(n, k, m):
        return _tiny_gemm(x, w, max_m=TINY_M_MAX)
    return torch.mm(x, wt)


def _router_forward(x: torch.Tensor, state, m: int) -> torch.Tensor:
    if m <= ROUTER_FP32_MAX_TOKENS:
        xf = x.float()
        wt = state["router_t"]
        if _router_gemm and _router_gemm_ok(xf, wt):
            return _router_gemm(xf, wt)
        return F.linear(xf, state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def prepare(weights: Weights):
    """Contiguous packs, SGLang kernel warmup. Untimed, up to 300s."""
    _init_kernels()
    state = {}
    for name, w in weights.items():
        wc = w.contiguous()
        state[name] = wc
        if name != "router":
            state[name + "_t"] = wc.t().contiguous()
    router = state["router"]
    state["router_t"] = router.t().contiguous()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    unp = state["unpooler"]
    state["unpooler_slice"] = unp[:UNPOOL_SLICE].contiguous()
    state["unpooler_slice_t"] = state["unpooler_slice"].t().contiguous()

    device = router.device
    for m in WARMUP_MS:
        for layer in BF16_LAYERS:
            n = UNPOOL_SLICE if (layer == "unpooler" and m <= SLICED_MAX_TOKENS) else state[layer].shape[0]
            w = _bf16_w(state, layer, n)
            wt = _bf16_wt(state, layer, n)
            x = torch.zeros((m, w.shape[1]), dtype=torch.bfloat16, device=device)
            _bf16_forward(x, w, wt, m)
        rx = torch.zeros((m, router.shape[1]), dtype=torch.bfloat16, device=device)
        _router_forward(rx, state, m)
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    """Return x @ W[:n].T: bf16, or float32 for the router."""
    x = data["x"]
    cfg = data["config"]
    layer = cfg["layer"]
    m = cfg["m"]
    n = cfg["n"]

    if layer == "router":
        return _router_forward(x, state, m)

    w = _bf16_w(state, layer, n)
    wt = _bf16_wt(state, layer, n)
    return _bf16_forward(x, w, wt, m)
