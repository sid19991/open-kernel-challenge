# okc: problem=muse_bf16_proj
"""BF16 projections + FP32 router for H200 CUDA-graph replay.

prepare() packs weights once (untimed). kernel_fn is captured as a CUDA graph:
branching on shapes/config only, no host syncs.

Router (floor target ~1.8 µs): BF16 x streamed against FP32 W in one kernel,
in-kernel split-K reduce, no host cast. Used for M<=16. Above that, hi+lo mm.
Pooler (floor target ~14 µs): Hopper-style streaming GEMV with x in smem
(kRows=2) for M<=8. Larger M stays on NN torch.mm.
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
ROUTER_CUDA_MAX_M = 16
ROUTER_K = 8192
ROUTER_SPLIT_K = 32
ROUTER_N_TILE = 64
POOLER_N = 4096
POOLER_K = 8192
POOLER_CUDA_MAX_M = 8

_hopper_gemv = None
_use_hopper_gemv = None
_tiny_gemm = None
_can_use_tiny = None
_router_gemm = None
_router_gemm_ok = None
_cuda_mod = None
_pooler_cuda_ok = True


_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
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

struct RouterParams {
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
router_kernel(const RouterParams params) {
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

__device__ __forceinline__ float dot8(const float4 wv, const float4 xv) {
  const __nv_bfloat16* w2 = reinterpret_cast<const __nv_bfloat16*>(&wv);
  const __nv_bfloat16* x2 = reinterpret_cast<const __nv_bfloat16*>(&xv);
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc = fmaf(__bfloat162float(w2[i]), __bfloat162float(x2[i]), acc);
  }
  return acc;
}

template <int MaxM, int N, int K, int kRowC, int kUnroll, int kNumWarps>
__global__ void __launch_bounds__(kNumWarps * 32)
pooler_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                   const __nv_bfloat16* __restrict__ w,
                   __nv_bfloat16* __restrict__ out,
                   int M) {
  extern __shared__ __nv_bfloat16 sx[];
  constexpr int kVec = 8;
  const int tid = threadIdx.x;
#pragma unroll
  for (int m = 0; m < MaxM; ++m) {
    if (m < M) {
      const __nv_bfloat16* xr = x + static_cast<int64_t>(m) * K;
      __nv_bfloat16* sr = sx + static_cast<int64_t>(m) * K;
      for (int i = tid * kVec; i < K; i += kNumWarps * 32 * kVec) {
        *reinterpret_cast<float4*>(sr + i) = *reinterpret_cast<const float4*>(xr + i);
      }
    }
  }
  __syncthreads();

  const int warp = tid / 32;
  const int lane = tid % 32;
  const int r0 = (blockIdx.x * kNumWarps + warp) * kRowC;
  if (r0 >= N) return;

  float acc[MaxM][kRowC];
#pragma unroll
  for (int m = 0; m < MaxM; ++m)
#pragma unroll
    for (int r = 0; r < kRowC; ++r) acc[m][r] = 0.f;

  constexpr int kStep = 32 * kVec * kUnroll;
  for (int k = lane * kVec * kUnroll; k < K; k += kStep) {
    float4 xv[MaxM][kUnroll];
#pragma unroll
    for (int m = 0; m < MaxM; ++m) {
      if (m < M) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          xv[m][u] = *reinterpret_cast<const float4*>(sx + static_cast<int64_t>(m) * K + k + u * kVec);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < kRowC; ++r) {
      if (r0 + r < N) {
        const __nv_bfloat16* wr = w + static_cast<size_t>(r0 + r) * K + k;
        float4 wv[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          wv[u] = __ldcs(reinterpret_cast<const float4*>(wr + u * kVec));
        }
#pragma unroll
        for (int m = 0; m < MaxM; ++m) {
          if (m < M) {
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) acc[m][r] += dot8(wv[u], xv[m][u]);
          }
        }
      }
    }
  }

#pragma unroll
  for (int m = 0; m < MaxM; ++m) {
    if (m < M) {
#pragma unroll
      for (int r = 0; r < kRowC; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          acc[m][r] += __shfl_down_sync(0xffffffff, acc[m][r], off);
        }
        if (lane == 0 && r0 + r < N) {
          out[static_cast<int64_t>(m) * N + r0 + r] = __float2bfloat16(acc[m][r]);
        }
      }
    }
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
  RouterParams params{
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

void pooler_bf16(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
  const int M = static_cast<int>(x.size(0));
  constexpr int N = 4096;
  constexpr int K = 8192;
  constexpr int kRowC = 2;
  constexpr int kUnroll = 2;
  constexpr int kNumWarps = 8;
  constexpr int kRowsPerBlock = kRowC * kNumWarps;
  constexpr int kBlocks = (N + kRowsPerBlock - 1) / kRowsPerBlock;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M == 1) {
    auto kn = pooler_gemv_kernel<1, N, K, kRowC, kUnroll, kNumWarps>;
    kn<<<kBlocks, kNumWarps * 32, 1 * K * 2, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        M);
  } else {
    auto kn = pooler_gemv_kernel<8, N, K, kRowC, kUnroll, kNumWarps>;
    cudaFuncSetAttribute(kn, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * K * 2);
    kn<<<kBlocks, kNumWarps * 32, 8 * K * 2, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        M);
  }
}
"""


def _compile_cuda():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="muse_router_pooler_v2",
        cpp_sources="""
#include <torch/extension.h>
void router_bf16xfp32(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void pooler_bf16(torch::Tensor, torch::Tensor, torch::Tensor);
""",
        cuda_sources=_CUDA,
        functions=["router_bf16xfp32", "pooler_bf16"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        with_cuda=True,
        verbose=False,
    )


def _init_kernels() -> None:
    global _hopper_gemv, _use_hopper_gemv
    global _tiny_gemm, _can_use_tiny
    global _router_gemm, _router_gemm_ok
    global _cuda_mod
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
        _cuda_mod = _compile_cuda()
        print("router+pooler CUDA ready")
    except Exception as e:
        print("CUDA compile failed:", e)
        _cuda_mod = False


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


def _pooler_cuda(x: torch.Tensor, w: torch.Tensor, m: int) -> Optional[torch.Tensor]:
    if not _cuda_mod or not _pooler_cuda_ok or m < 1 or m > POOLER_CUDA_MAX_M:
        return None
    if w.shape[0] != POOLER_N or w.shape[1] != POOLER_K:
        return None
    out = torch.empty((m, POOLER_N), dtype=torch.bfloat16, device=x.device)
    _cuda_mod.pooler_bf16(x, w, out)
    return out


def _bf16_forward(x: torch.Tensor, w: torch.Tensor, wt: torch.Tensor, m: int) -> torch.Tensor:
    n, k = w.shape
    if m == 1 and _hopper_gemv and _use_hopper_gemv(1, n, k):
        return _hopper_gemv(x, w)
    if _use_tiny(n, k, m):
        return _tiny_gemm(x, w, max_m=TINY_M_MAX)
    return torch.mm(x, wt)


def _router_cuda(x: torch.Tensor, state, m: int) -> Optional[torch.Tensor]:
    if not _cuda_mod:
        return None
    n = state["router"].shape[0]
    k = state["router"].shape[1]
    if k != ROUTER_K or n % ROUTER_N_TILE != 0 or m < 1 or m > ROUTER_CUDA_MAX_M:
        return None
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _cuda_mod.router_bf16xfp32(
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
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def prepare(weights: Weights):
    """Contiguous packs, CUDA / SGLang warmup. Untimed, up to 300s."""
    global _cuda_mod, _pooler_cuda_ok
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
                if layer == "pooler" and _pooler_cuda_ok:
                    _pooler_cuda_ok = False
                    print("disabled pooler CUDA after warmup failure")
                    _bf16_forward(x, w, wt, m)
        rx = torch.zeros((m, router.shape[1]), dtype=torch.bfloat16, device=device)
        try:
            _router_forward(rx, state, m)
        except Exception as e:
            print(f"warmup router m={m} failed:", e)
            _cuda_mod = False
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
