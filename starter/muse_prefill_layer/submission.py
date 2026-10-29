# okc: problem=muse_prefill_layer
"""SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache
                                   (cast, then the store_cache JIT kernel at the step's rows)
  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers
  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);
                                   it also writes the FP32 router input up to 8 tokens
  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs
                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a
                                   side stream, overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two humming indexed W4A16 GEMMs on the
                                   NVFP4 weights (SGLang's humming MoE runner kernels, in place of
                                   moe_wna16_marlin_gemm), silu_and_mul, Triton post-expert norm
                                   weighted sum; then the all-reduce (identity here)
  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then
                                   the all-gather (identity here: zeros in the other columns);
                                   above that, the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang:
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."""

from typing import TypedDict

import json
import time

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from humming.config import GemmType
from humming.layer import HummingInputSchema, HummingMethod
from humming.schema import BaseWeightSchema
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
HUMMING_COMPUTE = json.dumps({"use_f16_accum": False, "gemm_type": GemmType.INDEXED.value})
# One humming tile for every prefill shape (best in-layer on H200 from 753 to 16K tokens). Block M
# tiles one expert's rows, and the busiest GPU's experts are uneven, so stream-K stays on.
HUMMING_BLOCK_M = 128
HUMMING_TILE = json.dumps([[0, 1 << 30, {
    "block_shape": [HUMMING_BLOCK_M, 256, 64], "warp_shape": [HUMMING_BLOCK_M, 32, 64],
    "use_stream_k": True, "use_f16_accum": False, "num_stages": 4}]])
# moe_w4a16.cu's tiles, by cfg index: (warps, 16-row blocks per warp, 8-token groups); the block of
# slots (moe_align's block size) is 8 * groups.
CUDA_CFGS = ((8, 2, 4), (8, 2, 8), (8, 4, 4), (8, 4, 8), (4, 4, 8), (4, 2, 2))
MOE_CU = r"""
// NVFP4 W4A16 MoE GEMM for H200: C[slot, n] = sum_k X[row(slot), k] * W[e, n, k] over one expert's
// moe_align block of slots. Weights are repacked (losslessly) so that one 32-bit word holds one lane's
// eight E2M1 codes of an m16n8k16 A fragment; scales stay E4M3. Tensor cores see W as the A operand
// (16 output features) and the tokens as B (8 per MMA), accumulating in FP32.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define KDIM 4096
#define KCH (KDIM / 64)
// Activation rows in shared memory: 64 bf16 of a K chunk plus 16 pad, so the 8-byte B-fragment
// loads of a half warp (4 rows x 4 lanes) cover all 32 banks once.
#define AS 80

__device__ __forceinline__ void cp_async16(void* dst, const void* src, int bytes) {
  unsigned s = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}

// Two E4M3 scales (rows g, g+8) as bf16x2 equal to sf * 2^118 (exact: E4M3 fits FP16, all normal).
__device__ __forceinline__ uint32_t scale_bf16x2(uint32_t s16) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)s16, __NV_E4M3);
  uint32_t v = (uint32_t)h.x | ((uint32_t)h.y << 16);
  return ((v >> 3) & 0x0FFF0FFFu) + 0x73007300u;
}

// Pair j (nibbles at bits 4j and 16+4j) becomes a bf16x2 equal to the E2M1 values * 2^-126 (code 1,
// 0.5, is a bf16 subnormal); times the scale that is value * sf * 2^-8, exact in bf16.
__device__ __forceinline__ void decode(uint32_t w, uint32_t sb, uint32_t a[4]) {
  const uint32_t M = 0x01C001C0u, S = 0x80008000u;
  uint32_t slo = __byte_perm(sb, 0, 0x1010), shi = __byte_perm(sb, 0, 0x3232);
  a[0] = bmul2(((w << 6) & M) | ((w << 12) & S), slo);
  a[1] = bmul2(((w << 2) & M) | ((w << 8) & S), shi);
  a[2] = bmul2(((w >> 2) & M) | ((w << 4) & S), slo);
  a[3] = bmul2(((w >> 6) & M) | (w & S), shi);
}

__device__ __forceinline__ void mma(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int WARPS, int WM, int NT, int STAGES>
struct Cfg {
  static constexpr int BT = NT * 8, NCTA = WARPS * 16 * WM, THREADS = WARPS * 32;
  static constexpr int ACT = BT * AS * 2, WB = WARPS * WM * 512, SB = WARPS * WM * 64;
  static constexpr int STAGE = ACT + WB + SB, OS = NCTA + 8;
  static constexpr int PIPE = STAGES * STAGE, EPI = BT * OS * 2;
  static constexpr int SMEM = PIPE > EPI ? PIPE : EPI;
};

template <int WARPS, int WM, int NT, int STAGES>
__global__ void __launch_bounds__(WARPS * 32) moe_w4a16_kernel(
    const __nv_bfloat16* __restrict__ X, int row_div, const uint4* __restrict__ Wq,
    const uint2* __restrict__ Sq, const float* __restrict__ gs, const int* __restrict__ sorted_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ n_padded, __nv_bfloat16* __restrict__ C,
    int N, int numel) {
  using P = Cfg<WARPS, WM, NT, STAGES>;
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ int s_slot[P::BT];
  const int blk = blockIdx.x;
  if (blk * P::BT >= *n_padded) return;
  const int e = expert_ids[blk];
  if (e < 0) return;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  for (int r = tid; r < P::BT; r += P::THREADS) s_slot[r] = sorted_ids[blk * P::BT + r];
  __syncthreads();
  const int RB = N / 16;
  const int rb0 = blockIdx.y * (P::NCTA / 16) + warp * WM;
  const size_t wrow = (size_t)e * RB + rb0;

  auto load_stage = [&](int kc, int st) {
    unsigned char* base = smem + st * P::STAGE;
    __nv_bfloat16* act = (__nv_bfloat16*)base;
#pragma unroll
    for (int p = tid; p < P::BT * 8; p += P::THREADS) {
      int r = p >> 3, c = p & 7, slot = s_slot[r];
      bool v = (unsigned)slot < (unsigned)numel;
      const __nv_bfloat16* src = X + (size_t)(v ? slot / row_div : 0) * KDIM + kc * 64 + c * 8;
      cp_async16(act + r * AS + c * 8, src, v ? 16 : 0);
    }
    uint4* w = (uint4*)(base + P::ACT) + warp * WM * 32;
#pragma unroll
    for (int m = 0; m < WM; ++m) cp_async16(w + m * 32 + lane, Wq + ((wrow + m) * KCH + kc) * 32 + lane, 16);
    unsigned char* s = base + P::ACT + P::WB + warp * WM * 64;
    if (lane < WM * 4) {
      int m = lane >> 2, q = lane & 3;
      cp_async16(s + m * 64 + q * 16, (const unsigned char*)(Sq + ((wrow + m) * KCH + kc) * 8) + q * 16, 16);
    }
  };

  float acc[WM][NT][4];
#pragma unroll
  for (int m = 0; m < WM; ++m)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[m][n][i] = 0.f;

#pragma unroll
  for (int st = 0; st < STAGES - 1; ++st) {
    load_stage(st, st);
    cp_async_commit();
  }

#pragma unroll 1
  for (int kc = 0; kc < KCH; ++kc) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    int nk = kc + STAGES - 1;
    if (nk < KCH) load_stage(nk, nk % STAGES);
    cp_async_commit();
    const unsigned char* base = smem + (kc % STAGES) * P::STAGE;
    const __nv_bfloat16* act = (const __nv_bfloat16*)base;
    const uint4* w = (const uint4*)(base + P::ACT) + warp * WM * 32;
    const uint2* s = (const uint2*)(base + P::ACT + P::WB) + warp * WM * 8;
    uint4 wr[WM];
    uint2 sr[WM];
#pragma unroll
    for (int m = 0; m < WM; ++m) {
      wr[m] = w[m * 32 + lane];
      sr[m] = s[m * 8 + g];
    }
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t b[NT][2];
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        uint2 v = *(const uint2*)(act + (n * 8 + g) * AS + ks * 16 + 4 * t);
        b[n][0] = v.x;
        b[n][1] = v.y;
      }
#pragma unroll
      for (int m = 0; m < WM; ++m) {
        uint32_t word = ks == 0 ? wr[m].x : ks == 1 ? wr[m].y : ks == 2 ? wr[m].z : wr[m].w;
        uint32_t s16 = ((ks < 2 ? sr[m].x : sr[m].y) >> ((ks & 1) * 16)) & 0xFFFFu;
        uint32_t a[4];
        decode(word, scale_bf16x2(s16), a);
#pragma unroll
        for (int n = 0; n < NT; ++n) mma(acc[m][n], a, b[n][0], b[n][1]);
      }
    }
  }

  cp_async_wait<0>();
  __syncthreads();
  __nv_bfloat16* out = (__nv_bfloat16*)smem;
  const float sc = gs[e] * 256.f;
#pragma unroll
  for (int m = 0; m < WM; ++m)
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      int row = (warp * WM + m) * 16 + g, tok = n * 8 + 2 * t;
      out[tok * P::OS + row] = __float2bfloat16_rn(acc[m][n][0] * sc);
      out[(tok + 1) * P::OS + row] = __float2bfloat16_rn(acc[m][n][1] * sc);
      out[tok * P::OS + row + 8] = __float2bfloat16_rn(acc[m][n][2] * sc);
      out[(tok + 1) * P::OS + row + 8] = __float2bfloat16_rn(acc[m][n][3] * sc);
    }
  __syncthreads();
  constexpr int VPR = P::NCTA / 8;
  for (int p = tid; p < P::BT * VPR; p += P::THREADS) {
    int r = p / VPR, c = p % VPR, slot = s_slot[r];
    if ((unsigned)slot < (unsigned)numel)
      *(uint4*)(C + (size_t)slot * N + blockIdx.y * P::NCTA + c * 8) = *(const uint4*)(out + r * P::OS + c * 8);
  }
}

template <int WARPS, int WM, int NT, int STAGES>
static int launch(const void* X, int row_div, const void* W, const void* S, const float* gs, const int* sorted,
                  const int* experts, const int* npad, void* C, int N, int numel, int nblocks, cudaStream_t stream) {
  using P = Cfg<WARPS, WM, NT, STAGES>;
  auto kern = moe_w4a16_kernel<WARPS, WM, NT, STAGES>;
  static bool init = false;
  if (!init) {
    cudaError_t err = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, P::SMEM);
    if (err != cudaSuccess) return (int)err;
    init = true;
  }
  if (N % P::NCTA) return -1;
  dim3 grid(nblocks, N / P::NCTA);
  kern<<<grid, P::THREADS, P::SMEM, stream>>>((const __nv_bfloat16*)X, row_div, (const uint4*)W, (const uint2*)S,
                                               gs, sorted, experts, npad, (__nv_bfloat16*)C, N, numel);
  return (int)cudaGetLastError();
}

int64_t moe_gemm(int64_t x, int64_t row_div, int64_t w, int64_t s, int64_t gs_, int64_t sorted_, int64_t experts_,
                 int64_t npad_, int64_t c, int64_t N, int64_t numel, int64_t nblocks, int64_t cfg, int64_t stream) {
  const void *X = (const void*)x, *W = (const void*)w, *S = (const void*)s;
  const float* gs = (const float*)gs_;
  const int *sorted = (const int*)sorted_, *experts = (const int*)experts_, *npad = (const int*)npad_;
  void* C = (void*)c;
  cudaStream_t st = (cudaStream_t)stream;
  int rd = (int)row_div, n = (int)N, m = (int)numel, nb = (int)nblocks;
  switch (cfg) {
    case 0: return launch<8, 2, 4, 4>(X, rd, W, S, gs, sorted, experts, npad, C, n, m, nb, st);
    case 1: return launch<8, 2, 8, 4>(X, rd, W, S, gs, sorted, experts, npad, C, n, m, nb, st);
    case 2: return launch<8, 4, 4, 4>(X, rd, W, S, gs, sorted, experts, npad, C, n, m, nb, st);
    case 3: return launch<8, 4, 8, 3>(X, rd, W, S, gs, sorted, experts, npad, C, n, m, nb, st);
    case 4: return launch<4, 4, 8, 4>(X, rd, W, S, gs, sorted, experts, npad, C, n, m, nb, st);
    case 5: return launch<4, 2, 2, 6>(X, rd, W, S, gs, sorted, experts, npad, C, n, m, nb, st);
  }
  return -2;
}
"""
MOE_CPP = """#include <cstdint>
int64_t moe_gemm(int64_t x, int64_t row_div, int64_t w, int64_t s, int64_t gs, int64_t sorted_ids, int64_t expert_ids,
                 int64_t n_padded, int64_t c, int64_t n, int64_t numel, int64_t nblocks, int64_t cfg, int64_t stream);
"""
try:
    # At import, so the judge can build it ahead of the GPU run; on failure the experts stay on humming.
    _MOE_LIB = load_inline(name="okc_moe_w4a16", cpp_sources=MOE_CPP, cuda_sources=MOE_CU, functions=["moe_gemm"],
                           extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"])
    _MOE_ERR = None
except Exception as exc:  # noqa: BLE001
    _MOE_LIB, _MOE_ERR = None, exc
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _humming_layer(weights):
    """The local experts as a humming W4A16 MoE layer: the ModelOpt NVFP4 tensors as they are
    (E2M1 codes, E4M3 16-element block scales, FP32 tensor scale), in humming's layout."""
    layer = nn.Module()
    for name in ("w13", "w2"):
        layer.register_parameter(f"{name}_weight", _param(weights[name].clone()))
        layer.register_parameter(f"{name}_weight_scale", _param(weights[f"{name}_sf"].clone()))
        layer.register_parameter(f"{name}_weight_scale_2", _param(weights[f"{name}_gs"].float().clone()))
    schema = BaseWeightSchema.from_config({"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16})
    for sub, (shape_n, shape_k) in {"w13": (2 * D_EXPERT, D_EXPERT), "w2": (D_EXPERT, D_EXPERT)}.items():
        tensors = {k.removeprefix(sub + "_"): v for k, v in layer.state_dict().items() if k.startswith(sub + "_")}
        stacks = [shape_n // 2] * 2 if sub == "w13" else [shape_n]
        sub_schema, tensors = schema.convert_humming(tensors=tensors, shape_n_stacks=stacks, shape_k_stacks=[shape_k],
                                                     num_experts=LOCAL_EXPERTS, param_dtype=torch.bfloat16)
        for name, _ in list(layer.named_parameters()):
            if name.startswith(sub + "_"):
                delattr(layer, name)
        for name, tensor in tensors.items():
            setattr(layer, f"{sub}_{name}", nn.Parameter(tensor, requires_grad=False))
        HummingMethod.prepare_layer_meta(layer=layer, shape_n=shape_n, shape_k=shape_k, pad_n_to_multiple=256,
                                         pad_k_to_multiple=128, input_schema=HummingInputSchema(),
                                         weight_schema=sub_schema, has_bias=False, num_experts=LOCAL_EXPERTS,
                                         torch_dtype=torch.bfloat16, sublayer_name=sub)
        HummingMethod.transform_humming_layer(layer, sublayer_name=sub)
    layer.register_buffer("locks", torch.zeros(1024, dtype=torch.int32, device=weights["w13"].device))
    return layer


class _CudaMoE:
    """The NVFP4 W4A16 GEMMs of MOE_CU (load_inline) on a lossless
    repack of the ModelOpt tensors: per (expert, 16 rows, 64 k) one 32-bit word per lane holding its
    eight E2M1 codes of an m16n8k16 A fragment, and the E4M3 scales of rows g, g+8 side by side."""

    def __init__(self, lib, weights):
        self.lib = lib
        self.w, self.s, self.gs = {}, {}, {}
        for sub in ("w13", "w2"):
            w, sf = weights[sub], weights[f"{sub}_sf"]
            sf_f = sf.float()
            if not bool(torch.isfinite(sf_f).all()) or bool((sf_f < 0).any()):
                raise ValueError(f"{sub}: negative or non-finite block scales")
            zero = sf_f == 0
            if bool(zero.any()):
                nonzero = ((w.view(*sf.shape, 8) & 0x77) != 0).any(-1)
                if bool((zero & nonzero).any()):
                    raise ValueError(f"{sub}: zero block scale over nonzero codes")
            self.w[sub], self.s[sub] = _pack_nvfp4(w, sf)
            self.gs[sub] = weights[f"{sub}_gs"].float().contiguous()

    def gemm(self, x, row_div, sub, sorted_ids, expert_ids, n_padded, out, cfg):
        err = self.lib.moe_gemm(x.data_ptr(), row_div, self.w[sub].data_ptr(), self.s[sub].data_ptr(),
                                self.gs[sub].data_ptr(), sorted_ids.data_ptr(), expert_ids.data_ptr(),
                                n_padded.data_ptr(), out.data_ptr(), out.shape[1], out.shape[0], expert_ids.numel(),
                                cfg, torch.cuda.current_stream().cuda_stream)
        if err:
            raise RuntimeError(f"moe_gemm cfg {cfg} failed: {err}")


def _pack_nvfp4(w, sf):
    experts, n, kh = w.shape
    k = 2 * kh
    words = torch.empty((experts, n // 16, k // 64, 32, 4), dtype=torch.int32, device=w.device)
    scales = torch.empty((experts, n // 16, k // 64, 8, 8), dtype=torch.uint8, device=w.device)
    for e in range(experts):
        b = w[e]
        codes = torch.stack((b & 15, b >> 4), dim=-1).view(n // 16, 2, 8, k // 64, 4, 4, 2, 2)
        # (row block, g / g+8, g, k chunk, k step, t, k pair, k in pair) -> lane-major words whose
        # nibble 4 * (k in pair) + 2 * (k pair) + (g / g+8) feeds A register (k pair, g / g+8).
        codes = codes.permute(0, 3, 2, 5, 4, 7, 6, 1).reshape(-1, 2)
        packed = (codes[:, 0] | (codes[:, 1] << 4)).view(n // 16, k // 64, 32, 4, 4)
        words[e] = packed.view(torch.int32).view(n // 16, k // 64, 32, 4)
        s = sf[e].view(torch.uint8).view(n // 16, 2, 8, k // 64, 4)
        scales[e] = s.permute(0, 3, 2, 4, 1).reshape(n // 16, k // 64, 8, 8)
    return words, scales


def _tclass(tokens):
    return 0 if tokens < 1500 else 1 if tokens < 8000 else 2


def _cuda_time(fn, iters):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def _tune_cuda(state, device):
    """Checks every CUDA tile against humming and times both on skewed synthetic routing (this GPU's
    experts drawn ~1.4x as often as the mean, unevenly), per token class; returns the class -> (w13
    tile, w2 tile) choice, None for humming."""
    cu, layer = state["cuda"], state["humming"]
    gen = torch.Generator().manual_seed(0)
    probs = torch.ones(N_EXPERTS)
    probs[:LOCAL_EXPERTS] = 1.5 * (0.4 + 1.2 * torch.rand(LOCAL_EXPERTS, generator=gen))
    dgen = torch.Generator(device=device).manual_seed(0)
    choice = {}
    for cls, tokens in enumerate((753, 3012, 16384)):
        ids = torch.multinomial(probs.expand(tokens, -1), TOP_K, generator=gen).to(device)
        local = torch.where(ids < LOCAL_EXPERTS, ids, -1).to(torch.int32)
        valid = local.view(-1) >= 0
        x = torch.randn((tokens, D_EXPERT), generator=dgen, device=device).to(torch.bfloat16)
        inter = torch.randn((tokens * TOP_K, D_EXPERT), generator=dgen, device=device).to(torch.bfloat16)
        sorted_ids, expert_ids, n_padded = moe_align_block_size(local, HUMMING_BLOCK_M, LOCAL_EXPERTS,
                                                                ignore_invalid_expert=True)
        common = dict(sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=n_padded,
                      compute_config=HUMMING_COMPUTE, valid_shape_m=local.numel(), tuning_config=HUMMING_TILE)
        refs, best = {}, {}
        iters = 10 if tokens < 8000 else 4
        for sub, inp, top_k, n in (("w13", x, TOP_K, 2 * D_EXPERT), ("w2", inter, 1, D_EXPERT)):
            ref = torch.zeros((tokens * TOP_K, n), device=device, dtype=torch.bfloat16)
            run = lambda: HummingMethod.forward_layer(layer=layer, inputs=inp, input_scale=None, outputs=ref,
                                                      sublayer_name=sub, top_k=top_k, **common)
            t_ref = _cuda_time(run, iters)
            refs[sub] = ref
            best[sub] = (t_ref * 0.98, None)
            msg = [f"humming={t_ref:.0f}us"]
            for cfg, (_, _, nt) in enumerate(CUDA_CFGS):
                try:
                    s_ids, e_ids, n_pad = moe_align_block_size(local, nt * 8, LOCAL_EXPERTS,
                                                               ignore_invalid_expert=True)
                    out = torch.zeros_like(ref)
                    fn = lambda: cu.gemm(inp, TOP_K if sub == "w13" else 1, sub, s_ids, e_ids, n_pad, out, cfg)
                    fn()
                    torch.cuda.synchronize()
                    r, o = ref[valid].float(), out[valid].float()
                    err = ((o - r).abs().max() / r.abs().max().clamp_min(1e-30)).item()
                    mean_err = ((o - r).abs().mean() / r.abs().mean().clamp_min(1e-30)).item()
                    t = _cuda_time(fn, iters)
                    ok = err < 1e-2 and mean_err < 2e-3
                    msg.append(f"cfg{cfg}={t:.0f}us err={err:.1e}/{mean_err:.1e}{'' if ok else ' BAD'}")
                    if ok and t < best[sub][0]:
                        best[sub] = (t, cfg)
                    del out
                except Exception as exc:
                    msg.append(f"cfg{cfg} failed {exc!r}")
            print(f"[cuda] T={tokens} {sub} " + " ".join(msg), flush=True)
        choice[cls] = (best["w13"][1], best["w2"][1])
        print(f"[cuda] T={tokens} choice={choice[cls]}", flush=True)
        del refs, x, inter
    torch.cuda.empty_cache()
    return choice


def _experts(x, topk_weights, local_ids, state):
    """The local experts (ids 0-31; -1 elsewhere) with W4A16 GEMMs on the NVFP4 weights (the CUDA
    kernel where it measured faster in prepare, humming's indexed GEMM otherwise), then
    MuseSpark1xFp4Experts' post-expert-norm weighted sum."""
    num_tokens = x.shape[0]
    tiles = state["cuda_choice"].get(_tclass(num_tokens), (None, None))
    aligned = {}

    def align(block):
        if block not in aligned:
            aligned[block] = moe_align_block_size(local_ids, block, LOCAL_EXPERTS, ignore_invalid_expert=True)
        return aligned[block]

    def gemm(sub, inp, top_k, out, cfg):
        if cfg is None:
            sorted_ids, expert_ids, n_padded = align(HUMMING_BLOCK_M)
            HummingMethod.forward_layer(layer=state["humming"], inputs=inp, input_scale=None, outputs=out,
                                        sublayer_name=sub, top_k=top_k, sorted_ids=sorted_ids, expert_ids=expert_ids,
                                        num_tokens_padded=n_padded, compute_config=HUMMING_COMPUTE,
                                        valid_shape_m=local_ids.numel(), tuning_config=HUMMING_TILE)
        else:
            state["cuda"].gemm(inp, top_k, sub, *align(CUDA_CFGS[cfg][2] * 8), out, cfg)

    h = torch.empty((num_tokens * TOP_K, 2 * D_EXPERT), device=x.device, dtype=x.dtype)
    gemm("w13", x, TOP_K, h, tiles[0])
    inter = torch.empty((num_tokens * TOP_K, D_EXPERT), device=x.device, dtype=x.dtype)
    silu_and_mul(h, inter, expert_ids=local_ids.view(-1), expert_step=1)
    y = torch.empty((num_tokens * TOP_K, D_EXPERT), device=x.device, dtype=x.dtype)
    gemm("w2", inter, 1, y, tiles[1])
    return fused_post_expert_norm_weighted_sum(
        y.view(num_tokens, TOP_K, D_EXPERT), topk_weights.to(torch.float32).contiguous(),
        norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids)


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    state["humming"] = _humming_layer(weights)
    state["cuda_choice"] = {}
    try:
        start = time.time()
        if _MOE_LIB is None:
            raise RuntimeError(f"build failed: {str(_MOE_ERR)[-3000:]}")
        state["cuda"] = _CudaMoE(_MOE_LIB, weights)
        print(f"[cuda] packed in {time.time() - start:.0f}s", flush=True)
        state["cuda_choice"] = _tune_cuda(state, device)
        print(f"[cuda] tuned in {time.time() - start:.0f}s", flush=True)
    except Exception as exc:
        print(f"[cuda] disabled: {exc!r}"[:4000], flush=True)
    del freqs
    torch.cuda.synchronize()
    return state



def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank (-1 elsewhere).
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    local_ids = torch.where(local_ids < LOCAL_EXPERTS, local_ids, -1).to(torch.int32)
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
