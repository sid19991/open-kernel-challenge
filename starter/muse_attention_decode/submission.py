# okc: problem=muse_attention_decode
"""Paged FP8 GQA for H200 CUDA-graph replay.

FA3 is the SGLang baseline and already strong on Hopper, but page_size=1 is a
random 64-byte gather per token. This kernel keeps FusedQKNorm / store_cache /
rms_norm_gate and, on page_size=1, runs a specialized split-K GQA kernel that
indexes the page table directly (no CSR) and reuses each KV load across the
verify block's L queries. page_size=64 and compile failure fall back to FA3.
"""

from typing import TypedDict

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from torch.utils.cpp_extension import load_inline

from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
H200_SMS = 132
ROPE_POSITIONS = 1 << 21

_NORMS = {}
_ROPE = {}
_WS = {}
_CUDA = None


def _qk_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in _NORMS:
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _NORMS[key]


def _cos_sin(theta, d, device):
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def _buf(tag, shape, dtype, device):
    n = 1
    for s in shape:
        n *= int(s)
    key = (tag, dtype, device)
    t = _WS.get(key)
    if t is None or t.numel() < n:
        t = torch.empty(n, dtype=dtype, device=device)
        _WS[key] = t
    return t[:n].view(*shape)


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <algorithm>
#include <cmath>

constexpr int kD = 64;
constexpr int kMaxL = 8;

__device__ __forceinline__ float fp8_to_f(uint8_t x) {
    return __half2float(__nv_cvt_fp8_to_halfraw(
        static_cast<__nv_fp8_storage_t>(x), __NV_E4M3));
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, m);
    }
    return x;
}

// One warp per query-head of a GQA group. Block = G warps = one (b, kv_head, split).
// Q layout [B, L, Hq, 64] BF16. K/V: flat rows [row, Hkv, 64] FP8. page_size=1.
template <int L, int G>
__global__ void paged_gqa_fwd_kernel(
        const __nv_bfloat16* __restrict__ q,
        const uint8_t* __restrict__ k_cache,
        const uint8_t* __restrict__ v_cache,
        const int* __restrict__ page_table,
        const int* __restrict__ cache_seqlens,
        float* __restrict__ part_o,
        float* __restrict__ part_m,
        float* __restrict__ part_l,
        int B, int Hq, int Hkv, int pt_stride,
        int window, int splits, float sm_scale, float k_scale) {
    const int split = blockIdx.z;
    const int kv_h = blockIdx.y;
    const int b = blockIdx.x;
    const int g = threadIdx.x >> 5;          // warp = GQA member
    const int lane = threadIdx.x & 31;
    if (g >= G) return;

    const int q_h = kv_h * G + g;
    const int kv_len = cache_seqlens[b];
    const int t_lo = window > 0 ? max(0, kv_len - window) : 0;
    const int t_hi = kv_len;
    const int span = t_hi - t_lo;
    const int chunk = (span + splits - 1) / splits;
    const int s0 = t_lo + split * chunk;
    const int s1 = min(s0 + chunk, t_hi);

    float q0[L], q1[L];
#pragma unroll
    for (int j = 0; j < L; ++j) {
        const __nv_bfloat16* qj = q + (((b * L + j) * Hq + q_h) * kD);
        q0[j] = __bfloat162float(qj[lane]);
        q1[j] = __bfloat162float(qj[lane + 32]);
    }

    float acc0[L], acc1[L], m[L], ell[L];
#pragma unroll
    for (int j = 0; j < L; ++j) {
        acc0[j] = 0.f;
        acc1[j] = 0.f;
        m[j] = -INFINITY;
        ell[j] = 0.f;
    }

    const int* pt = page_table + b * pt_stride;
    const int kv_row = Hkv * kD;

    for (int t = s0; t < s1; ++t) {
        const int page = pt[t];
        if (page < 0) continue;
        const uint8_t* kptr = k_cache + (static_cast<long long>(page) * kv_row + kv_h * kD);
        const uint8_t* vptr = v_cache + (static_cast<long long>(page) * kv_row + kv_h * kD);
        const float k0 = fp8_to_f(kptr[lane]);
        const float k1 = fp8_to_f(kptr[lane + 32]);
        const float v0 = fp8_to_f(vptr[lane]);
        const float v1 = fp8_to_f(vptr[lane + 32]);

#pragma unroll
        for (int j = 0; j < L; ++j) {
            const int pos = kv_len - L + j;
            if (t > pos) continue;
            if (window > 0 && t < pos - window + 1) continue;
            float qk = (q0[j] * k0 + q1[j] * k1);
            qk = warp_sum(qk) * sm_scale * k_scale;
            const float m_new = fmaxf(m[j], qk);
            const float e = __expf(qk - m_new);
            const float alpha = __expf(m[j] - m_new);
            acc0[j] = acc0[j] * alpha + e * v0;
            acc1[j] = acc1[j] * alpha + e * v1;
            ell[j] = ell[j] * alpha + e;
            m[j] = m_new;
        }
    }

    // part_o: [splits, B, L, Hq, D]
    const long long base = ((((static_cast<long long>(split) * B + b) * L) * Hq + q_h) * kD);
#pragma unroll
    for (int j = 0; j < L; ++j) {
        const long long off = base + static_cast<long long>(j) * Hq * kD;
        part_o[off + lane] = acc0[j];
        part_o[off + lane + 32] = acc1[j];
        if (lane == 0) {
            const long long ml = (((static_cast<long long>(split) * B + b) * L + j) * Hq + q_h);
            part_m[ml] = m[j];
            part_l[ml] = ell[j];
        }
    }
}

__global__ void paged_gqa_reduce_kernel(
        const float* __restrict__ part_o,
        const float* __restrict__ part_m,
        const float* __restrict__ part_l,
        __nv_bfloat16* __restrict__ o,
        int B, int L, int Hq, int splits, float v_scale) {
    // One warp per (b, j, q_h)
    const int idx = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nout = B * L * Hq;
    if (idx >= nout) return;
    const int lane = threadIdx.x & 31;
    const int q_h = idx % Hq;
    const int tmp = idx / Hq;
    const int j = tmp % L;
    const int b = tmp / L;

    float m = -INFINITY, ell = 0.f, acc0 = 0.f, acc1 = 0.f;
    for (int s = 0; s < splits; ++s) {
        const long long ml = (((static_cast<long long>(s) * B + b) * L + j) * Hq + q_h);
        const float ms = part_m[ml];
        if (!isfinite(ms)) continue;
        const long long off = ((((static_cast<long long>(s) * B + b) * L + j) * Hq + q_h) * kD);
        const float a0 = part_o[off + lane];
        const float a1 = part_o[off + lane + 32];
        const float ls = part_l[ml];
        const float m_new = fmaxf(m, ms);
        const float a = __expf(m - m_new);
        const float bsc = __expf(ms - m_new);
        acc0 = acc0 * a + a0 * bsc;
        acc1 = acc1 * a + a1 * bsc;
        ell = ell * a + ls * bsc;
        m = m_new;
    }
    const float inv = (ell > 0.f) ? (v_scale / ell) : 0.f;
    __nv_bfloat16* oj = o + ((b * L + j) * Hq + q_h) * kD;
    oj[lane] = __float2bfloat16(acc0 * inv);
    oj[lane + 32] = __float2bfloat16(acc1 * inv);
}

void paged_gqa(
        torch::Tensor q,          // [B, L, Hq, 64] bf16
        torch::Tensor k_cache,    // [rows, Hkv, 64] float8
        torch::Tensor v_cache,
        torch::Tensor page_table, // int32 [B, max_pages]
        torch::Tensor cache_seqlens,
        torch::Tensor part_o,
        torch::Tensor part_m,
        torch::Tensor part_l,
        torch::Tensor o,          // [B, L, Hq, 64] bf16
        int window, int splits, float sm_scale, float k_scale, float v_scale) {
    const int B = q.size(0);
    const int L = q.size(1);
    const int Hq = q.size(2);
    const int Hkv = k_cache.size(1);
    const int G = Hq / Hkv;
    TORCH_CHECK(q.size(3) == kD, "head_dim");
    TORCH_CHECK(L >= 1 && L <= kMaxL, "q_len");
    TORCH_CHECK(G >= 1 && G <= 16, "gqa");

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(B, Hkv, splits);
    dim3 block(G * 32);

    auto launch = [&](auto kernel) {
        kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
            reinterpret_cast<const uint8_t*>(v_cache.data_ptr()),
            page_table.data_ptr<int>(),
            cache_seqlens.data_ptr<int>(),
            part_o.data_ptr<float>(),
            part_m.data_ptr<float>(),
            part_l.data_ptr<float>(),
            B, Hq, Hkv, (int)page_table.stride(0),
            window, splits, sm_scale, k_scale);
    };

    if (L == 1 && G == 8) launch(paged_gqa_fwd_kernel<1, 8>);
    else if (L == 8 && G == 8) launch(paged_gqa_fwd_kernel<8, 8>);
    else if (L == 1 && G == 16) launch(paged_gqa_fwd_kernel<1, 16>);
    else if (L == 8 && G == 16) launch(paged_gqa_fwd_kernel<8, 16>);
    else if (L == 1 && G == 4) launch(paged_gqa_fwd_kernel<1, 4>);
    else if (L == 8 && G == 4) launch(paged_gqa_fwd_kernel<8, 4>);
    else TORCH_CHECK(false, "unsupported L/G");

    const int nout = B * L * Hq;
    const int warps = 8;
    const int rblocks = (nout + warps - 1) / warps;
    paged_gqa_reduce_kernel<<<rblocks, warps * 32, 0, stream>>>(
        part_o.data_ptr<float>(),
        part_m.data_ptr<float>(),
        part_l.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
        B, L, Hq, splits, v_scale);
}
"""


def _cuda():
    global _CUDA
    if _CUDA is False:
        return None
    if _CUDA is None:
        try:
            _CUDA = load_inline(
                name="muse_attn_paged_gqa",
                cpp_sources="""
#include <torch/extension.h>
void paged_gqa(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor page_table, torch::Tensor cache_seqlens,
    torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l,
    torch::Tensor o, int window, int splits, float sm_scale, float k_scale, float v_scale);
""",
                cuda_sources=_CUDA_SRC,
                functions=["paged_gqa"],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS_FUNCTION__",
                ],
                verbose=False,
            )
        except Exception as e:
            print("paged_gqa compile failed:", type(e).__name__, e)
            _CUDA = False
            return None
    return _CUDA


def _split_count(batch, hkv, window, q_len):
    work = max(1, batch * hkv)
    splits = max(1, min(64, (H200_SMS * 4) // work))
    if window:
        splits = min(splits, 16)
    if q_len > 1:
        splits = min(splits, 32)
    return max(1, splits)


def _fa3(q, k_cache, v_cache, data, cfg, tokens, hq, d, length, scaled):
    window = cfg["window"]
    descale = {}
    if scaled:
        b = cfg["batch"]
        hkv = cfg["num_kv_heads"]
        descale = dict(
            k_descale=data["k_scale"].reshape(1, 1).expand(b, hkv),
            v_descale=data["v_scale"].reshape(1, 1).expand(b, hkv),
        )
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
    return out


def _paged_cuda(q, k_cache, v_cache, data, cfg, batch, length, hq, hkv, d):
    mod = _cuda()
    if mod is None:
        return None
    splits = _split_count(batch, hkv, cfg["window"], length)
    q4 = q.view(batch, length, hq, d).contiguous()
    k_rows = k_cache.reshape(-1, hkv, d)
    v_rows = v_cache.reshape(-1, hkv, d)
    part_o = _buf("part_o", (splits, batch, length, hq, d), torch.float32, q.device)
    part_m = _buf("part_m", (splits, batch, length, hq), torch.float32, q.device)
    part_l = _buf("part_l", (splits, batch, length, hq), torch.float32, q.device)
    o = _buf("attn_o", (batch, length, hq, d), torch.bfloat16, q.device)
    mod.paged_gqa(
        q4,
        k_rows,
        v_rows,
        data["page_table"],
        data["cache_seqlens"],
        part_o,
        part_m,
        part_l,
        o,
        int(cfg["window"]),
        int(splits),
        float(cfg["sm_scale"] * cfg["qk_scale_factor"]),
        float(cfg["kv_scale"]),
        float(cfg["kv_scale"]),
    )
    return o


def kernel_fn(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device

    if cfg["rope_theta"]:
        q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, device))
    else:
        q, k = _qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]

    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    store_cache(
        k.to(FP8).view(torch.uint8),
        v.to(FP8).view(torch.uint8),
        k_cache.view(-1, hkv * d).view(torch.uint8),
        v_cache.view(-1, hkv * d).view(torch.uint8),
        data["cache_loc"],
        reserved_skip_index=-1,
    )

    use_cuda = (
        cfg["page_size"] == 1
        and d == 64
        and length in (1, 8)
        and hq % hkv == 0
        and (hq // hkv) in (4, 8, 16)
    )
    out = None
    if use_cuda:
        out = _paged_cuda(q, k_cache, v_cache, data, cfg, batch, length, hq, hkv, d)
    if out is None:
        out = _fa3(q, k_cache, v_cache, data, cfg, tokens, hq, d, length, scaled)

    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(
        out.reshape(tokens, hq * d).to(torch.bfloat16),
        gate,
        None,
        head_dim=d,
        eps=RMS_EPS,
        has_norm=True,
    )
    return o.view(batch, length, hq * d)
