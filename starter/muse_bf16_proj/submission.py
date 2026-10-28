# okc: problem=muse_bf16_proj
"""BF16 projections + FP32 router for H200 CUDA-graph replay.

prepare() packs weights once (untimed). kernel_fn is captured as a CUDA graph:
branching on shapes/config only, no host syncs.

Official H200 check 84e26dd270dc4051: 4509.4 µs (1.071x vs SGLang).
This revision floors the two largest gaps:
- router: BF16 x, one FP32 weight pass, in-kernel split-K reduce (M<=32);
  fused hi+lo tensor-core GEMM for M>32 (one launch, no extra reduce)
- pooler M=2..16: SGLang dsv3 fused-A GEMM (cp.async + MMA)
"""

from typing import Optional, TypedDict

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
ROUTER_CUDA_MAX_M = 32
ROUTER_K = 8192
ROUTER_SPLIT_K = 32
ROUTER_N_TILE = 64
FUSED_A_MAX_M = 16

_hopper_gemv = None
_use_hopper_gemv = None
_tiny_gemm = None
_can_use_tiny = None
_router_gemm = None
_router_gemm_ok = None
_fused_a = None
_router_mod = None
_hilo_kernel = None


_ROUTER_CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kHidden = 8192;
constexpr int kSplitK = 32;
constexpr int kNTile = 64;
constexpr int kThreads = 256;
constexpr int kMaxGroup = 8;
constexpr int kColGroups = kNTile / 4;
constexpr int kPhases = kThreads / kColGroups;
constexpr int kRows = kHidden / kSplitK;
constexpr int kIters = kRows / kPhases;
constexpr int kRedPhases = kPhases / 2;

struct Params {
  const __nv_bfloat16* __restrict__ x;
  const float* __restrict__ w;
  float* __restrict__ out;
  float* __restrict__ partials;
  int* __restrict__ counters;
  int num_tokens;
  int64_t k;
  int64_t n;
};

__global__ void __launch_bounds__(kThreads)
router_kernel(const Params params) {
  const int split = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int group = threadIdx.x % kColGroups;
  const int phase = threadIdx.x / kColGroups;
  const int n = n_tile * kNTile + group * 4;
  const int64_t k0 = static_cast<int64_t>(split) * kRows;
  const int T = params.num_tokens;

  __shared__ __align__(16) float x_s[kMaxGroup * kRows];
  __shared__ __align__(16) float4 red[kRedPhases * kMaxGroup * kColGroups];
  __shared__ int is_last;

  float4 wv[kIters];
  const float* w_col = params.w + k0 * params.n + n;
#pragma unroll
  for (int i = 0; i < kIters; ++i) {
    wv[i] = __ldg(reinterpret_cast<const float4*>(
        w_col + static_cast<int64_t>(phase + i * kPhases) * params.n));
  }

  for (int t0 = 0; t0 < T; t0 += kMaxGroup) {
    for (int i = threadIdx.x; i < kMaxGroup * kRows; i += kThreads) {
      const int t = i / kRows;
      const int kk = i - t * kRows;
      const int tt = t0 + t;
      float v = 0.f;
      if (tt < T) {
        v = __bfloat162float(params.x[static_cast<int64_t>(tt) * params.k + k0 + kk]);
      }
      x_s[t * kRows + kk] = v;
    }
    __syncthreads();

    float4 acc[kMaxGroup];
#pragma unroll
    for (int t = 0; t < kMaxGroup; ++t) acc[t] = make_float4(0.f, 0.f, 0.f, 0.f);

#pragma unroll
    for (int i = 0; i < kIters; ++i) {
      const int kk = phase + i * kPhases;
#pragma unroll
      for (int t = 0; t < kMaxGroup; ++t) {
        if (t0 + t < T) {
          const float xv = x_s[t * kRows + kk];
          acc[t].x = fmaf(xv, wv[i].x, acc[t].x);
          acc[t].y = fmaf(xv, wv[i].y, acc[t].y);
          acc[t].z = fmaf(xv, wv[i].z, acc[t].z);
          acc[t].w = fmaf(xv, wv[i].w, acc[t].w);
        }
      }
    }

    if (phase >= kRedPhases) {
#pragma unroll
      for (int t = 0; t < kMaxGroup; ++t) {
        if (t0 + t < T) {
          red[((phase - kRedPhases) * kMaxGroup + t) * kColGroups + group] = acc[t];
        }
      }
    }
    __syncthreads();
    if (phase < kRedPhases) {
#pragma unroll
      for (int t = 0; t < kMaxGroup; ++t) {
        if (t0 + t < T) {
          const float4 o = red[(phase * kMaxGroup + t) * kColGroups + group];
          acc[t].x += o.x;
          acc[t].y += o.y;
          acc[t].z += o.z;
          acc[t].w += o.w;
          red[(phase * kMaxGroup + t) * kColGroups + group] = acc[t];
        }
      }
    }
    __syncthreads();

    float* partial = params.partials + static_cast<int64_t>(split) * T * params.n;
    if (phase == 0) {
#pragma unroll
      for (int t = 0; t < kMaxGroup; ++t) {
        if (t0 + t < T) {
          float4 s = red[t * kColGroups + group];
#pragma unroll
          for (int p = 1; p < kRedPhases; ++p) {
            const float4 o = red[(p * kMaxGroup + t) * kColGroups + group];
            s.x += o.x;
            s.y += o.y;
            s.z += o.z;
            s.w += o.w;
          }
          *reinterpret_cast<float4*>(partial + static_cast<int64_t>(t0 + t) * params.n + n) = s;
        }
      }
    }
    __syncthreads();
  }

  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int prev = atomicAdd(params.counters + n_tile, 1);
    is_last = (prev == kSplitK - 1);
    if (is_last) params.counters[n_tile] = 0;
  }
  __syncthreads();
  if (!is_last) return;
  __threadfence();

  for (int i = threadIdx.x; i < T * kNTile; i += kThreads) {
    const int t = i / kNTile;
    const int c = i - t * kNTile;
    const int64_t idx = static_cast<int64_t>(t) * params.n + n_tile * kNTile + c;
    float s = 0.0f;
#pragma unroll 8
    for (int sp = 0; sp < kSplitK; ++sp) {
      s += __ldcg(params.partials + static_cast<int64_t>(sp) * T * params.n + idx);
    }
    params.out[idx] = s;
  }
}

}  // namespace

void router_bf16xfp32(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor out,
    torch::Tensor partials,
    torch::Tensor counters) {
  const int T = static_cast<int>(x.size(0));
  const auto n = w.size(1);
  const int n_tiles = static_cast<int>(n / kNTile);
  Params params{
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      w.data_ptr<float>(),
      out.data_ptr<float>(),
      partials.data_ptr<float>(),
      counters.data_ptr<int>(),
      T,
      x.size(1),
      n,
  };
  router_kernel<<<dim3(kSplitK, n_tiles), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(params);
}
"""


def _compile_router():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="muse_router_bf16xfp32_v1",
        cpp_sources="""
#include <torch/extension.h>
void router_bf16xfp32(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
""",
        cuda_sources=_ROUTER_CU,
        functions=["router_bf16xfp32"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        with_cuda=True,
        verbose=False,
    )


def _init_hilo_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _router_hilo(
        x_ptr,
        hi_ptr,
        lo_ptr,
        out_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_hin,
        stride_lon,
        stride_om,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            hi = tl.load(
                hi_ptr + offs_n[:, None] * stride_hin + offs_k[None, :],
                mask=mask_n[:, None],
                other=0.0,
            )
            lo = tl.load(
                lo_ptr + offs_n[:, None] * stride_lon + offs_k[None, :],
                mask=mask_n[:, None],
                other=0.0,
            )
            acc += tl.dot(x, tl.trans(hi), out_dtype=tl.float32)
            acc += tl.dot(x, tl.trans(lo), out_dtype=tl.float32)
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    def run(x: torch.Tensor, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        n = hi.shape[0]
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        block_m = 16
        block_n = 16
        block_k = 64
        grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))
        _router_hilo[grid](
            x,
            hi,
            lo,
            out,
            m,
            n,
            k,
            x.stride(0),
            hi.stride(0),
            lo.stride(0),
            out.stride(0),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=4,
        )
        return out

    return run


def _init_kernels() -> None:
    global _hopper_gemv, _use_hopper_gemv
    global _tiny_gemm, _can_use_tiny
    global _router_gemm, _router_gemm_ok
    global _fused_a, _router_mod, _hilo_kernel
    if _hopper_gemv is not None:
        return
    try:
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import (
            hopper_bf16_gemv,
            use_hopper_bf16_gemv,
        )

        _hopper_gemv = hopper_bf16_gemv
        _use_hopper_gemv = use_hopper_bf16_gemv
    except Exception as e:
        print("hopper_bf16_gemv unavailable:", e)
        _hopper_gemv = False
        _use_hopper_gemv = lambda *args, **kwargs: False
    try:
        from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16

        _tiny_gemm = tiny_gemm_bf16
        _can_use_tiny = can_use_tiny_gemm
    except Exception as e:
        print("tiny_gemm unavailable:", e)
        _tiny_gemm = False
        _can_use_tiny = lambda *args, **kwargs: False
    try:
        from sglang.kernels.ops.muse_spark_v12.router_gemm import (
            router_gemm,
            router_gemm_supported,
        )

        _router_gemm = router_gemm
        _router_gemm_ok = router_gemm_supported
    except Exception as e:
        print("router_gemm unavailable:", e)
        _router_gemm = False
        _router_gemm_ok = lambda *args, **kwargs: False
    try:
        from sglang.kernels.ops.gemm.dsv3_fused_a_gemm import dsv3_fused_a_gemm

        _fused_a = dsv3_fused_a_gemm
    except Exception:
        try:
            from sglang.kernels.ops.gemm.fused_a_gemm import dsv3_fused_a_gemm

            _fused_a = dsv3_fused_a_gemm
        except Exception as e:
            print("dsv3_fused_a_gemm unavailable:", e)
            _fused_a = False
    try:
        _router_mod = _compile_router()
        print("router CUDA kernel ready")
    except Exception as e:
        print("router CUDA compile failed:", e)
        _router_mod = False
    try:
        _hilo_kernel = _init_hilo_kernel()
        print("router hi+lo triton ready")
    except Exception as e:
        print("router hi+lo triton failed:", e)
        _hilo_kernel = False


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


def _use_fused_a(n: int, k: int, m: int) -> bool:
    return bool(
        _fused_a
        and 1 < m <= FUSED_A_MAX_M
        and n == 4096
        and k == 8192
        and n % 16 == 0
        and k % 256 == 0
    )


def _bf16_forward(x: torch.Tensor, w: torch.Tensor, wt: torch.Tensor, m: int) -> torch.Tensor:
    n, k = w.shape
    if m == 1 and _hopper_gemv and _use_hopper_gemv(1, n, k):
        return _hopper_gemv(x, w)
    if _use_fused_a(n, k, m):
        return _fused_a(x, w.T)
    if _use_tiny(n, k, m):
        return _tiny_gemm(x, w, max_m=TINY_M_MAX)
    return torch.mm(x, wt)


def _router_cuda(x: torch.Tensor, state, m: int) -> Optional[torch.Tensor]:
    if not _router_mod:
        return None
    n = state["router"].shape[0]
    k = state["router"].shape[1]
    if k != ROUTER_K or n % ROUTER_N_TILE != 0 or m < 1 or m > ROUTER_CUDA_MAX_M:
        return None
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _router_mod.router_bf16xfp32(
        x, state["router_t"], out, state["router_partials"], state["router_counters"]
    )
    return out


def _router_forward(x: torch.Tensor, state, m: int) -> torch.Tensor:
    y = _router_cuda(x, state, m)
    if y is not None:
        return y
    if m <= ROUTER_FP32_MAX_TOKENS:
        xf = x.float()
        wt = state["router_t"]
        if _router_gemm and _router_gemm_ok(xf, wt):
            return _router_gemm(xf, wt)
        return F.linear(xf, state["router"])
    if _hilo_kernel is not False and _hilo_kernel is not None:
        return _hilo_kernel(x, state["router_hi"], state["router_lo"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def prepare(weights: Weights):
    """Contiguous packs, SGLang / CUDA / Triton warmup. Untimed, up to 300s."""
    global _fused_a, _router_mod, _hilo_kernel
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
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    unp = state["unpooler"]
    state["unpooler_slice"] = unp[:UNPOOL_SLICE].contiguous()
    state["unpooler_slice_t"] = state["unpooler_slice"].t().contiguous()

    n_exp = router.shape[0]
    state["router_partials"] = torch.empty(
        ROUTER_SPLIT_K * ROUTER_CUDA_MAX_M * n_exp, dtype=torch.float32, device=router.device
    )
    state["router_counters"] = torch.zeros(
        max(1, n_exp // ROUTER_N_TILE), dtype=torch.int32, device=router.device
    )

    device = router.device
    for m in WARMUP_MS:
        for layer in BF16_LAYERS:
            n = UNPOOL_SLICE if (layer == "unpooler" and m <= SLICED_MAX_TOKENS) else state[layer].shape[0]
            w = _bf16_w(state, layer, n)
            wt = _bf16_wt(state, layer, n)
            x = torch.zeros((m, w.shape[1]), dtype=torch.bfloat16, device=device)
            try:
                _bf16_forward(x, w, wt, m)
            except Exception as e:
                print(f"warmup bf16 {layer} m={m} failed:", e)
                if layer == "pooler" and _fused_a:
                    _fused_a = False
                    print("disabled fused_a after warmup failure")
                    _bf16_forward(x, w, wt, m)
        rx = torch.zeros((m, router.shape[1]), dtype=torch.bfloat16, device=device)
        try:
            _router_forward(rx, state, m)
        except Exception as e:
            print(f"warmup router m={m} failed:", e)
            if m <= ROUTER_CUDA_MAX_M:
                _router_mod = False
            else:
                _hilo_kernel = False
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
