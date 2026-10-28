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
  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),
                                   silu_and_mul, Triton post-expert norm weighted sum
                                   (MuseSpark1xFp4Experts._forward_marlin); then the
                                   all-reduce (identity here)
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

import types

import importlib.util
import os
import shutil

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
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
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
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
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


@triton.jit
def _e2m1_times(codes, scale):
    # E2M1 code -> FP16 bits scaled by 2^-14 (exact, incl. the 0.5 subnormal); 2^14 is folded into gs.
    bits = ((codes & 0x8) << 12) | ((codes & 0x7) << 9)
    mag = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return (mag * scale).to(tl.bfloat16)


@triton.jit
def _nvfp4_moe_gemm(a_ptr, b_ptr, s_ptr, gs_ptr, c_ptr, sorted_ptr, bexp_ptr, npad_ptr,
                    num_valid, num_pid_m, N, K, stride_am, stride_be, stride_bn, stride_se, stride_sn, stride_cm,
                    TOP_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                    GROUP_M: tl.constexpr):
    """C[slot] = A[slot // TOP_K] @ dequant(B[e]).T over the blocks moe_align_block_size sorted;
    blocks of non-local experts (-1) are skipped."""
    pid = tl.program_id(0)
    num_pid_n = N // BLOCK_N
    in_group = GROUP_M * num_pid_n
    first_m = (pid // in_group) * GROUP_M
    group_size = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % in_group) % group_size
    pid_n = (pid % in_group) // group_size
    if pid_m * BLOCK_M >= tl.load(npad_ptr):
        return
    e = tl.load(bexp_ptr + pid_m).to(tl.int64)
    if e < 0:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    slot = tl.load(sorted_ptr + offs_m)
    mask_m = slot < num_valid
    row = tl.where(mask_m, slot // TOP_K, 0).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + row[:, None] * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + e * stride_be + offs_n[:, None] * stride_bn + tl.arange(0, BLOCK_K // 2)[None, :]
    s_ptrs = s_ptr + e * stride_se + offs_n[:, None] * stride_sn + tl.arange(0, BLOCK_K // 16)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        packed = tl.load(b_ptrs).to(tl.int32)
        sf = tl.load(s_ptrs).to(tl.float32)
        codes = tl.reshape(tl.join(packed & 0xF, packed >> 4), (BLOCK_N, BLOCK_K))
        scale = tl.reshape(tl.broadcast_to(sf[:, :, None], (BLOCK_N, BLOCK_K // 16, 16)), (BLOCK_N, BLOCK_K))
        acc = tl.dot(a, tl.trans(_e2m1_times(codes, scale)), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K // 2
        s_ptrs += BLOCK_K // 16
    acc = acc * (tl.load(gs_ptr + e) * 16384.0)
    c_ptrs = c_ptr + slot.to(tl.int64)[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])


def _nvfp4_gemm(a, out, w, sf, gs, sorted_ids, block_experts, n_padded, top_k, cfg):
    bm, bn, bk, warps, stages = cfg
    n, k = w.shape[1], w.shape[2] * 2
    num_pid_m = triton.cdiv(sorted_ids.numel(), bm)
    _nvfp4_moe_gemm[(num_pid_m * (n // bn),)](
        a, w, sf, gs, out, sorted_ids, block_experts, n_padded, out.shape[0], num_pid_m, n, k,
        a.stride(0), w.stride(0), w.stride(1), sf.stride(0), sf.stride(1), out.stride(0),
        TOP_K=top_k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=warps, num_stages=stages)
    return out


@triton.jit
def _dequant_kernel(b_ptr, s_ptr, gs_ptr, out_ptr, rows_per_expert, K,
                    BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr):
    """NVFP4 [R, K/2] (+ scales) -> BF16 [R, K], R = experts * rows_per_expert."""
    r = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
    kb = tl.program_id(1) * BLOCK_K
    packed = tl.load(b_ptr + r[:, None] * (K // 2) + kb // 2 + tl.arange(0, BLOCK_K // 2)[None, :]).to(tl.int32)
    sf = tl.load(s_ptr + r[:, None] * (K // 16) + kb // 16 + tl.arange(0, BLOCK_K // 16)[None, :]).to(tl.float32)
    gs = tl.load(gs_ptr + r // rows_per_expert) * 16384.0
    codes = tl.reshape(tl.join(packed & 0xF, packed >> 4), (BLOCK_R, BLOCK_K))
    scale = tl.reshape(tl.broadcast_to((sf * gs[:, None])[:, :, None], (BLOCK_R, BLOCK_K // 16, 16)),
                       (BLOCK_R, BLOCK_K))
    w = _e2m1_times(codes, scale)
    tl.store(out_ptr + r[:, None] * K + kb + tl.arange(0, BLOCK_K)[None, :], w)


def _dequant(w, sf, gs):
    e, n, half = w.shape
    k = half * 2
    out = torch.empty((e, n, k), device=w.device, dtype=torch.bfloat16)
    _dequant_kernel[(e * n // 32, k // 256)](w, sf, gs, out, n, k, BLOCK_R=32, BLOCK_K=256, num_warps=4)
    return out


@triton.jit
def _gather_rows(src_ptr, idx_ptr, dst_ptr, total_ptr, C, DIV: tl.constexpr, BLOCK_C: tl.constexpr):
    """dst[i] = src[idx[i] // DIV] for i < total."""
    i = tl.program_id(0)
    if i >= tl.load(total_ptr):
        return
    src = tl.load(idx_ptr + i).to(tl.int64) // DIV
    for c in range(0, C, BLOCK_C):
        offs = c + tl.arange(0, BLOCK_C)
        tl.store(dst_ptr + i.to(tl.int64) * C + offs, tl.load(src_ptr + src * C + offs))


@triton.jit
def _scatter_rows(src_ptr, idx_ptr, dst_ptr, total_ptr, C, BLOCK_C: tl.constexpr):
    """dst[idx[i]] = src[i] for i < total."""
    i = tl.program_id(0)
    if i >= tl.load(total_ptr):
        return
    dst = tl.load(idx_ptr + i).to(tl.int64)
    for c in range(0, C, BLOCK_C):
        offs = c + tl.arange(0, BLOCK_C)
        tl.store(dst_ptr + dst * C + offs, tl.load(src_ptr + i.to(tl.int64) * C + offs))


@triton.jit
def _silu_mul_rows(h_ptr, out_ptr, total_ptr, D, BLOCK_C: tl.constexpr):
    i = tl.program_id(0)
    if i >= tl.load(total_ptr):
        return
    base = i.to(tl.int64)
    for c in range(0, D, BLOCK_C):
        offs = c + tl.arange(0, BLOCK_C)
        g = tl.load(h_ptr + base * 2 * D + offs).to(tl.float32)
        u = tl.load(h_ptr + base * 2 * D + D + offs).to(tl.float32)
        tl.store(out_ptr + base * D + offs, (g * tl.sigmoid(g) * u).to(tl.bfloat16))


def _expert_outputs_gmm(x, local_ids, fp4):
    """Per-slot expert outputs [T * 8, 4096]: dequantize to BF16, sort the local routes by expert,
    and run two torch._grouped_mm (CUTLASS SM90) GEMMs."""
    num_tokens = x.shape[0]
    slots = num_tokens * TOP_K
    key = local_ids.view(-1).long()
    key = torch.where(key < 0, LOCAL_EXPERTS, key)
    order = torch.argsort(key, stable=True).to(torch.int32)
    counts = torch.zeros(LOCAL_EXPERTS + 1, device=x.device, dtype=torch.int32)
    counts.scatter_add_(0, key, torch.ones_like(key, dtype=torch.int32))
    offs = counts[:LOCAL_EXPERTS].cumsum(0).to(torch.int32)
    total = offs[-1:]
    xs = torch.empty((slots, D_EXPERT), device=x.device, dtype=x.dtype)
    _gather_rows[(slots,)](x, order, xs, total, D_EXPERT, DIV=TOP_K, BLOCK_C=1024, num_warps=4)
    w13 = _dequant(fp4["w13"], fp4["w13_sf"], fp4["w13_gs"])
    h = torch._grouped_mm(xs, w13.transpose(1, 2), offs=offs)
    del w13
    inter = torch.empty((slots, D_EXPERT), device=x.device, dtype=x.dtype)
    _silu_mul_rows[(slots,)](h, inter, total, D_EXPERT, BLOCK_C=1024, num_warps=4)
    del h
    w2 = _dequant(fp4["w2"], fp4["w2_sf"], fp4["w2_gs"])
    ys = torch._grouped_mm(inter, w2.transpose(1, 2), offs=offs)
    y = torch.empty((slots, D_EXPERT), device=x.device, dtype=x.dtype)
    _scatter_rows[(slots,)](ys, order, y, total, D_EXPERT, BLOCK_C=1024, num_warps=4)
    return y


MARLIN_MAX_TOKENS = 1024
# Above this many tokens the experts run as dequantize-to-BF16 + torch._grouped_mm (CUTLASS SM90).
GMM_MIN_TOKENS = 4097
# (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages) of _nvfp4_moe_gemm between the two.
FUSED_CFG = (128, 128, 64, 4, 4)
# humming's tile config is picked for T * 8 * 32 / 256 routes times this.
HUMMING_ROUTES_FACTOR = 1.0


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


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _report_toolchain():
    found = {name: importlib.util.find_spec(name) is not None
             for name in ("humming", "triton_kernels", "flashinfer", "cutlass", "deep_gemm", "vllm")}
    print(f"[toolchain] modules={found} nvcc={shutil.which('nvcc')} "
          f"CUDA_HOME={os.environ.get('CUDA_HOME')} triton={triton.__version__} torch={torch.__version__}",
          flush=True)


def _humming_layer(weights):
    """The local experts as a humming W4A16 MoE layer, from the ModelOpt NVFP4 tensors."""
    from humming.layer import HummingInputSchema, HummingMethod
    from humming.schema import BaseWeightSchema

    layer = nn.Module()
    layer.num_local_experts = LOCAL_EXPERTS
    layer.params_dtype = torch.bfloat16
    layer.with_bias = False
    for name in ("w13", "w2"):
        layer.register_parameter(f"{name}_weight", _param(weights[name].clone()))
        layer.register_parameter(f"{name}_weight_scale", _param(weights[f"{name}_sf"].clone()))
        layer.register_parameter(f"{name}_weight_scale_2", _param(weights[f"{name}_gs"].float().clone()))
    schema = BaseWeightSchema.from_config({"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16})
    print(f"[humming] checkpoint schema {type(schema).__name__}: {schema}", flush=True)
    shapes = {"w13": (2 * D_EXPERT, D_EXPERT), "w2": (D_EXPERT, D_EXPERT)}
    for sub, (shape_n, shape_k) in shapes.items():
        tensors = {k.removeprefix(sub + "_"): v for k, v in layer.state_dict().items() if k.startswith(sub + "_")}
        stacks = [shape_n // 2] * 2 if sub == "w13" else [shape_n]
        new_schema, tensors = schema.convert_humming(tensors=tensors, shape_n_stacks=stacks, shape_k_stacks=[shape_k],
                                                     num_experts=LOCAL_EXPERTS, param_dtype=torch.bfloat16)
        print(f"[humming] {sub} schema {new_schema} tensors "
              f"{ {k: (tuple(v.shape), str(v.dtype)) for k, v in tensors.items()} }", flush=True)
        for name, _ in list(layer.named_parameters()):
            if name.startswith(sub + "_"):
                delattr(layer, name)
        for name, tensor in tensors.items():
            setattr(layer, f"{sub}_{name}", nn.Parameter(tensor, requires_grad=False))
        HummingMethod.prepare_layer_meta(layer=layer, shape_n=shape_n, shape_k=shape_k, pad_n_to_multiple=256,
                                         pad_k_to_multiple=128, input_schema=HummingInputSchema(),
                                         weight_schema=new_schema, has_bias=False, num_experts=LOCAL_EXPERTS,
                                         torch_dtype=torch.bfloat16, sublayer_name=sub)
        HummingMethod.transform_humming_layer(layer, sublayer_name=sub)
    layer.register_buffer("locks", torch.zeros(1024, dtype=torch.int32, device=weights["w13"].device))
    return layer


def _pick(cfg_list, m):
    return next((c for lo, hi, c in cfg_list if lo < m <= hi), cfg_list[-1][2])


def _humming_outputs(x, local_ids, layer, cfgs, routes_factor=HUMMING_ROUTES_FACTOR):
    """Per-slot expert outputs [T * 8, 4096] (rows of non-local slots unset). The tile config is
    chosen for the expected routes on this GPU's experts (about T), not SGLang's T * 8 bound."""
    import json

    from humming.layer import HummingMethod

    num_tokens = x.shape[0]
    expect_m = max(1, int(num_tokens * TOP_K * LOCAL_EXPERTS / N_EXPERTS * routes_factor))
    w13_cfg, w2_cfg = _pick(cfgs["w13"], expect_m), _pick(cfgs["w2"], expect_m)
    sorted_ids, expert_ids, n_padded = moe_align_block_size(local_ids, w13_cfg["block_shape"][0], LOCAL_EXPERTS,
                                                            ignore_invalid_expert=True)
    common = dict(sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=n_padded,
                  compute_config=cfgs["compute"], valid_shape_m=local_ids.numel())
    big = 1 << 30
    h = torch.empty((num_tokens * TOP_K, 2 * D_EXPERT), device=x.device, dtype=x.dtype)
    HummingMethod.forward_layer(layer=layer, inputs=x, input_scale=None, outputs=h, sublayer_name="w13",
                                top_k=TOP_K, tuning_config=json.dumps([[0, big, w13_cfg]]), **common)
    inter = torch.empty((num_tokens * TOP_K, D_EXPERT), device=x.device, dtype=x.dtype)
    silu_and_mul(h, inter, expert_ids=local_ids.view(-1), expert_step=1)
    y = torch.empty((num_tokens * TOP_K, D_EXPERT), device=x.device, dtype=x.dtype)
    w2_cfg = dict(w2_cfg, block_shape=[w13_cfg["block_shape"][0]] + list(w2_cfg["block_shape"][1:]),
                  warp_shape=[w13_cfg["block_shape"][0]] + list(w2_cfg["warp_shape"][1:]))
    HummingMethod.forward_layer(layer=layer, inputs=inter, input_scale=None, outputs=y, sublayer_name="w2",
                                top_k=1, tuning_config=json.dumps([[0, big, w2_cfg]]), **common)
    return y


def _humming_cfgs(layer):
    import json

    from humming.config import GemmType
    from humming.layer import HummingMethod

    out = {"compute": json.dumps({"use_f16_accum": False, "gemm_type": GemmType.INDEXED.value})}
    for sub in ("w13", "w2"):
        cfg = HummingMethod.get_default_tuning_configs(layer=layer, use_f16_accum=False, gemm_type=GemmType.INDEXED,
                                                       sublayer_name=sub)
        out[sub], out[sub + "_str"] = cfg, json.dumps(cfg)
    return out


def _marlin_outputs(x, local_ids, bank):
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    w = torch.ones(local_ids.shape, device=x.device, dtype=torch.float32)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    h = torch.empty((num_tokens * TOP_K, 2 * size_n), device=x.device, dtype=x.dtype)
    h = moe_wna16_marlin_gemm(x, h, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None,
                              bank["workspace"], sorted_ids, block_experts, n_padded, w, top_k=TOP_K,
                              size_m=num_tokens, size_n=2 * size_n, size_k=size_k, **common)
    inter = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    silu_and_mul(h, inter, expert_ids=local_ids.view(-1), expert_step=1)
    y = torch.empty((num_tokens * TOP_K, size_k), device=x.device, dtype=x.dtype)
    return moe_wna16_marlin_gemm(inter, y, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None,
                                 bank["workspace"], sorted_ids, block_experts, n_padded, w, top_k=1,
                                 size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common)


def _time(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / reps


def _humming_probe(state, weights):
    import traceback

    try:
        layer = _humming_layer(weights)
        cfgs = _humming_cfgs(layer)
        print(f"[humming] w13 tuning {cfgs['w13_str'][:1500]}", flush=True)
        device = weights["w13"].device
        gen = torch.Generator(device=device).manual_seed(0)
        for tokens in (753, 3012, 16384):
            x = torch.randn((tokens, D_EXPERT), device=device, dtype=torch.bfloat16, generator=gen)
            ids = torch.rand((tokens, N_EXPERTS), device=device, generator=gen).topk(TOP_K, dim=-1).indices
            local_ids = state["bank"]["expert_map"][ids].to(torch.int32).contiguous()
            valid = local_ids.view(-1) >= 0
            ref = _marlin_outputs(x, local_ids, state["bank"])[valid].float()
            t_ref = _time(lambda: _marlin_outputs(x, local_ids, state["bank"]))
            for factor in (1.0, 1.6, 2.5):
                out = _humming_outputs(x, local_ids, layer, cfgs, factor)[valid].float()
                err = float((out - ref).norm() / ref.norm())
                t = _time(lambda: _humming_outputs(x, local_ids, layer, cfgs, factor))
                bm = _pick(cfgs["w13"], int(tokens * factor))["block_shape"]
                print(f"[humming] T={tokens} factor={factor} block={bm} err_vs_marlin={err:.2e} "
                      f"humming={t:.1f}us marlin={t_ref:.1f}us x{t_ref / t:.2f}", flush=True)
        return layer
    except Exception:  # noqa: BLE001
        print("[humming] probe failed:\n" + traceback.format_exc()[-3000:], flush=True)
        return None


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
        "fp4": {name: weights[name].clone().contiguous()
                for name in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")},
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    state["fp4"]["w13_gs"] = state["fp4"]["w13_gs"].float()
    state["fp4"]["w2_gs"] = state["fp4"]["w2_gs"].float()
    state["bank"] = _marlin_bank(weights)
    del freqs
    torch.cuda.synchronize()
    _report_toolchain()
    state["humming"] = _humming_probe(state, weights)
    if state["humming"] is not None:
        state["humming_cfgs"] = _humming_cfgs(state["humming"])
    return state


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def _expert_outputs(x, local_ids, fp4, cfg):
    """Per-slot expert outputs [T * 8, 4096] with the NVFP4 Triton GEMMs (rows of non-local slots unset)."""
    num_tokens = x.shape[0]
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, cfg[0], N_EXPERTS)
    h = torch.empty((num_tokens * TOP_K, 2 * D_EXPERT), device=x.device, dtype=x.dtype)
    _nvfp4_gemm(x, h, fp4["w13"], fp4["w13_sf"], fp4["w13_gs"], sorted_ids, block_experts, n_padded, TOP_K, cfg)
    inter = torch.empty((num_tokens * TOP_K, D_EXPERT), device=x.device, dtype=x.dtype)
    silu_and_mul(h, inter, expert_ids=local_ids.view(-1), expert_step=1)
    y = torch.empty((num_tokens * TOP_K, D_EXPERT), device=x.device, dtype=x.dtype)
    return _nvfp4_gemm(inter, y, fp4["w2"], fp4["w2_sf"], fp4["w2_gs"], sorted_ids, block_experts, n_padded, 1, cfg)


def _experts_fp4(x, topk_weights, topk_ids, state):
    if state.get("humming") is not None:
        local_ids = state["bank"]["expert_map"][topk_ids].contiguous()
        y = _humming_outputs(x, local_ids, state["humming"], state["humming_cfgs"]).view(x.shape[0], TOP_K, D_EXPERT)
        return fused_post_expert_norm_weighted_sum(
            y, topk_weights.to(torch.float32).contiguous(), norm_weight=state["post_expert_w"], eps=POST_NORM_EPS,
            weight_first=True, slot_expert_ids=local_ids)
    if x.shape[0] <= MARLIN_MAX_TOKENS:
        return _experts(x, topk_weights, topk_ids, state)
    local_ids = state["bank"]["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    if x.shape[0] >= GMM_MIN_TOKENS:
        y = _expert_outputs_gmm(x, local_ids, state["fp4"])
    else:
        y = _expert_outputs(x, local_ids, state["fp4"], FUSED_CFG)
    y = y.view(x.shape[0], TOP_K, D_EXPERT)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


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
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _experts_fp4(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
