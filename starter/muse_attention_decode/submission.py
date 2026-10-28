# okc: problem=muse_attention_decode
"""Paged FP8 GQA decode/verify for H200 CUDA-graph replay.

SGLang's FA3 path is the baseline, but page_size=1 (the scored shape) turns
every KV token into a page-table lookup and leaves Hopper TMA on the table.
This kernel keeps the fused QK-norm / cache-store / gated RMSNorm from SGLang
and replaces decode (q_len=1) with Triton's split-K flash-decode, which is
written for page_size=1. Speculative verify (q_len=8) stays on FA3, which
reuses KV loads across the draft tokens.
"""

from typing import TypedDict

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
from sglang.kernels.ops.kvcache.kv_indices import (
    create_flashinfer_kv_indices_triton,
    kv_indices_num_token_blocks,
)
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64]
    positions: torch.Tensor    # int64 [B*L]
    cache_loc: torch.Tensor    # int64 [B*L]
    cache_seqlens: torch.Tensor  # int32 [B]
    cu_seqlens_q: torch.Tensor   # int32 [B+1]
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64]
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64]
    k_scale: torch.Tensor     # float32 []
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages]
    seq_lens: torch.Tensor    # int32 [B]
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
H200_SMS = 132
ROPE_POSITIONS = 1 << 21
_GROUPED_BLOCK_H = 16
try:
    _PDL = bool(is_arch_support_pdl())
except Exception:
    _PDL = False

_NORMS = {}
_ROPE = {}
_WS = {}
_POOL = {}


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
    """Grow-only workspace. Sized from tensor shapes / config, never from values."""
    n = 1
    for s in shape:
        n *= int(s)
    key = (tag, dtype, device)
    t = _WS.get(key)
    if t is None or t.numel() < n:
        t = torch.empty(n, dtype=dtype, device=device)
        _WS[key] = t
    return t[:n].view(*shape)


def _req_pool(batch, device):
    key = (batch, device)
    t = _POOL.get(key)
    if t is None:
        t = torch.arange(batch, dtype=torch.int32, device=device)
        _POOL[key] = t
    return t


def _split_count(batch, hq, hkv, window):
    """Stage-1 grid is (B, head_tiles, splits). Size splits from config only."""
    group = max(1, hq // max(1, hkv))
    block_h = min(_GROUPED_BLOCK_H, group)
    head_tiles = (hq + block_h - 1) // block_h
    token_grid = max(1, batch * head_tiles)
    splits = max(4, min(128, (H200_SMS * 4) // token_grid))
    if window:
        splits = min(splits, 32)
    p2 = 4
    while p2 < splits:
        p2 *= 2
    return min(p2, 128)


def _fill_csr(page_table, page, req_pool, kv_start, kv_lens, kv_indptr, kv_indices):
    """page_table -> ragged token-level physical rows. Graph-safe, no host reads."""
    batch = kv_lens.shape[0]
    kv_indptr.zero_()
    torch.cumsum(kv_lens, dim=0, out=kv_indptr[1:])
    max_tokens = page_table.shape[1] * page
    nblk = kv_indices_num_token_blocks(max_tokens, batch)
    parallel = nblk > 1
    grid = (batch, nblk) if parallel else (batch,)
    create_flashinfer_kv_indices_triton[grid](
        page_table,
        req_pool,
        kv_lens,
        kv_indptr,
        kv_start,
        kv_indices,
        page_table.stride(0),
        ENTRY_PAGE_SIZE=page,
        TOKEN_BLOCK_PARALLEL=parallel,
    )


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
        pack_gqa=True,
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def _decode(q, k_cache, v_cache, data, cfg, batch, hq, hkv, d, page):
    """Split-K flash-decode over the paged FP8 cache. q is [B, Hq, D]."""
    window = cfg["window"]
    cache_seqlens = data["cache_seqlens"]
    page_table = data["page_table"]
    device = q.device
    max_tokens = page_table.shape[1] * page
    splits = _split_count(batch, hq, hkv, window)

    kv_lens = _buf("kv_lens", (batch,), torch.int32, device)
    kv_start = _buf("kv_start", (batch,), torch.int32, device)
    if window:
        kv_lens.copy_(cache_seqlens.clamp(max=window))
    else:
        kv_lens.copy_(cache_seqlens)
    kv_start.copy_(cache_seqlens - kv_lens)

    kv_indptr = _buf("kv_indptr", (batch + 1,), torch.int32, device)
    kv_indices = _buf("kv_indices", (batch * max_tokens,), torch.int64, device)
    _fill_csr(page_table, page, _req_pool(batch, device), kv_start, kv_lens, kv_indptr, kv_indices)

    attn_logits = _buf("attn_logits", (batch, hq, splits, d), torch.float32, device)
    attn_lse = _buf("attn_lse", (batch, hq, splits), torch.float32, device)
    num_kv_splits = _buf("num_kv_splits", (batch,), torch.int32, device)
    num_kv_splits.fill_(splits)
    o = _buf("attn_o", (batch, hq, d), torch.bfloat16, device)

    k_buf = k_cache.reshape(-1, hkv, d)
    v_buf = v_cache.reshape(-1, hkv, d)
    decode_attention_fwd(
        q,
        k_buf,
        v_buf,
        o,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        splits,
        cfg["sm_scale"] * cfg["qk_scale_factor"],
        cfg["kv_scale"],
        cfg["kv_scale"],
        use_pdl=_PDL,
        page_size=1,
        enable_lean=False,
    )
    return o


def kernel_fn(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    page = cfg["page_size"]
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
    # Slot 0 is a live row here (SGLang reserves it for padding). FusedQKNorm's
    # kv_store skips loc==0, so the store stays a separate kernel.
    store_cache(
        k.to(FP8).view(torch.uint8),
        v.to(FP8).view(torch.uint8),
        k_cache.view(-1, hkv * d).view(torch.uint8),
        v_cache.view(-1, hkv * d).view(torch.uint8),
        data["cache_loc"],
        reserved_skip_index=-1,
    )

    if length == 1:
        out = _decode(
            q.view(batch, hq, d),
            k_cache,
            v_cache,
            data,
            cfg,
            batch,
            hq,
            hkv,
            d,
            page,
        )
    else:
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
