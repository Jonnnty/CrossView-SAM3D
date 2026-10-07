from __future__ import annotations

import numpy as np
import torch

from crossview.constants import ATTN_CHUNK, DROP, NGRID, PAIR, RGB_LEN


def _patch_index(view: int, nkeys: int, device, base: int) -> torch.Tensor:
    idx = torch.arange(1, 1 + NGRID * NGRID, device=device, dtype=torch.long) + view * PAIR + base
    return idx[idx < nkeys]


def response_grid(q, k, v, lse, bias, idx) -> np.ndarray:
    scale = q.shape[-1] ** -0.5
    qf = q.float() * scale
    acc = []
    for s in range(0, idx.shape[0], 256):
        ii = idx[s : s + 256]
        logits = torch.einsum("blhd,bkhd->blhk", qf, k.float().index_select(1, ii))
        logits = logits + bias.index_select(0, ii).view(1, 1, 1, -1)
        weight = torch.exp(logits - lse.unsqueeze(-1))
        ve = v.float().index_select(1, ii)
        contrib = torch.einsum("blhk,bkhd->blhkd", weight, ve)
        acc.append(contrib.norm(dim=-1).mean(dim=(0, 1, 2)))
        del logits, weight, ve, contrib
    return torch.cat(acc).reshape(NGRID, NGRID).detach().cpu().numpy().astype(np.float32)


def _attend(q, k, v, bias):
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = q.shape[-1] ** -0.5
    bsz, length, heads, dim = q.shape
    klen = k.shape[1]
    running = torch.full((bsz, length, heads), -1e20, device=q.device, dtype=torch.float32)
    mass = torch.zeros(bsz, length, heads, device=q.device, dtype=torch.float32)
    out = torch.zeros(bsz, length, heads, dim, device=q.device, dtype=torch.float32)
    for s in range(0, klen, ATTN_CHUNK):
        e = min(s + ATTN_CHUNK, klen)
        logits = torch.einsum("blhd,bkhd->blhk", qf * scale, kf[:, s:e])
        logits = logits + bias[s:e].view(1, 1, 1, -1)
        peak = logits.amax(dim=-1)
        nxt = torch.maximum(running, peak)
        alpha = torch.exp(running - nxt)
        w = torch.exp(logits - nxt.unsqueeze(-1))
        out = out * alpha.unsqueeze(-1) + torch.einsum("blhk,bkhd->blhd", w, vf[:, s:e])
        mass = mass * alpha + w.sum(dim=-1)
        running = nxt
        del logits, w, peak, nxt, alpha
    out = out / mass.clamp_min(1e-12).unsqueeze(-1)
    lse = running + torch.log(mass.clamp_min(1e-12))
    return out, lse


def _qkv(attn, x, context):
    bsz, length, channels = x.shape
    lkv = context.shape[1]
    q = attn.to_q(x).reshape(bsz, length, attn.num_heads, attn.head_dim)
    kv = attn.to_kv(context).reshape(bsz, lkv, 2, attn.num_heads, attn.head_dim)
    k, v = kv[:, :, 0], kv[:, :, 1]
    if attn.qk_rms_norm:
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    return q, k, v, channels


def layer_mean(rows: list[tuple[int, np.ndarray]]) -> np.ndarray:
    last = {}
    for layer, grid in rows:
        last[layer] = grid
    return np.stack([last[k] for k in sorted(last)], 0).mean(0)


def install_rgb_attention_equalization(backbone, states, holder: dict):
    saved = []
    for bi, block in enumerate(backbone.blocks):
        attn = block.cross_attn["shape"]
        orig = attn.forward

        def forward(x, context=None, orig=orig, attn=attn, bi=bi):
            if context is None or float(context.detach().float().abs().mean()) < 1e-6:
                return orig(x, context)
            q, k, v, channels = _qkv(attn, x, context)
            lkv = context.shape[1]
            bias = torch.zeros(lkv, device=x.device, dtype=torch.float32)
            if holder["equalize"]:
                for view, state in enumerate(states):
                    if not state.apply_b:
                        continue
                    idx = _patch_index(view, lkv, x.device, 0)
                    flat = torch.from_numpy(state.B.reshape(-1)[: idx.numel()]).to(x.device)
                    bias.index_copy_(0, idx[: flat.numel()], flat)
            if holder["drop"]:
                a = int(holder["segment"]) * PAIR
                b = min(a + PAIR, lkv)
                bias[a:b] = DROP
            out, lse = _attend(q, k, v, bias)
            for view, state in enumerate(states):
                if state.record:
                    state.rows.append((bi + 1, response_grid(q, k, v, lse, bias, _patch_index(view, lkv, x.device, 0))))
            return attn.to_out(out.reshape(x.shape[0], x.shape[1], channels).to(x.dtype))

        attn.forward = forward
        saved.append((attn, orig))
    return saved


def install_mask_stream_alignment(backbone, states):
    saved = []
    for bi, block in enumerate(backbone.blocks):
        attn = block.cross_attn["shape"]
        orig = attn.forward

        def forward(x, context=None, orig=orig, attn=attn, bi=bi):
            if context is None or float(context.detach().float().abs().mean()) < 1e-6:
                return orig(x, context)
            q, k, v, channels = _qkv(attn, x, context)
            lkv = context.shape[1]
            bias = torch.zeros(lkv, device=x.device, dtype=torch.float32)
            for state in states:
                if not state.apply_b:
                    continue
                idx = _patch_index(0, lkv, x.device, RGB_LEN)
                flat = torch.from_numpy(state.B.reshape(-1)[: idx.numel()]).to(x.device)
                bias.index_copy_(0, idx[: flat.numel()], flat)
            out, lse = _attend(q, k, v, bias)
            for state in states:
                if state.record:
                    idx = _patch_index(0, lkv, x.device, RGB_LEN)
                    state.rows.append((bi + 1, response_grid(q, k, v, lse, bias, idx)))
            return attn.to_out(out.reshape(x.shape[0], x.shape[1], channels).to(x.dtype))

        attn.forward = forward
        saved.append((attn, orig))
    return saved


def restore(saved) -> None:
    for attn, orig in saved:
        attn.forward = orig
