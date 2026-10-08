from __future__ import annotations

import json

import numpy as np
import torch

from sam3d_objects.model.backbone.generator.flow_matching.solver import linear_approximation_step

from crossview.attention import install_rgb_attention_equalization, layer_mean, restore
from crossview.constants import ALPHA, EQUALIZE_FROM, SEED, SIDE, STEPS, VIEW_LEN
from crossview.rgb_attention_equalization import rgb_attention_equalization
from crossview.share_alignment import on_support


def _detach(x):
    if isinstance(x, dict):
        return {k: v.detach() for k, v in x.items()}
    return x.detach()


def _noise(gen, latent_shape, device):
    torch.manual_seed(SEED)
    return _detach(gen._generate_noise(latent_shape, device))


def shape_speed_gap(aligned, reference) -> np.ndarray:
    delta = aligned["shape"] - reference["shape"]
    return delta.detach().float().reshape(-1, 8).norm(dim=-1).cpu().numpy().reshape(SIDE, SIDE, SIDE)


def weighted_velocity_fusion(velocities, weight: torch.Tensor):
    fused = {}
    count = len(velocities)
    for key in velocities[0]:
        terms = [v[key] for v in velocities]
        if key != "shape":
            fused[key] = sum(terms) / count
            continue
        acc = None
        for i, term in enumerate(terms):
            wi = weight[i].to(dtype=term.dtype, device=term.device)
            wi = wi.view(1, -1, 1) if term.ndim == 3 else wi.view(-1, 1)
            scaled = term * wi
            acc = scaled if acc is None else acc + scaled
        fused[key] = acc
    return fused


def gap_fusion_weight(speed: np.ndarray) -> np.ndarray:
    scaled = []
    for field in speed:
        values = field.astype(np.float64)
        center = float(np.median(values))
        width = max(float(np.percentile(values, 75) - np.percentile(values, 25)), 1e-8)
        logit = np.clip((values - center) / (0.5 * width), -20.0, 20.0)
        scaled.append(1.0 / (1.0 + np.exp(-logit)))
    score = -ALPHA * np.stack(scaled)
    score -= score.max(axis=0, keepdims=True)
    weight = np.exp(score)
    weight /= np.maximum(weight.sum(axis=0, keepdims=True), 1e-12)
    return weight.astype(np.float32)


def reference_mask_shares(gen, backbone, latent_shape, conditions, states) -> list[np.ndarray]:
    stride = int(conditions[0].shape[1])
    if stride != VIEW_LEN:
        raise RuntimeError(f"expected {VIEW_LEN} tokens per view, got {stride}")
    concat = torch.cat(list(conditions), dim=1)
    holder = {"equalize": True, "capture": True, "sink": [], "stride": stride}
    saved = install_rgb_attention_equalization(backbone, states, holder)
    shares = None
    x = _noise(gen, latent_shape, concat.device)
    t_seq, d_val = gen._prepare_t_and_d()
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
            for step, (t_a, t_b) in enumerate(zip(t_seq[:STEPS], t_seq[1 : STEPS + 1]), start=1):
                for state in states:
                    state.apply_b = step >= EQUALIZE_FROM
                    state.record = step >= EQUALIZE_FROM - 1
                    state.rows = []
                holder["sink"].clear()
                full = gen._generate_dynamics(x, float(t_a), d_val, concat)
                if step == STEPS:
                    by_view: list[list] = [[] for _ in states]
                    for view, layer, grid in holder["sink"]:
                        by_view[view].append((layer, grid))
                    shares = [
                        on_support(layer_mean(rows), state.support) for rows, state in zip(by_view, states)
                    ]
                if EQUALIZE_FROM - 1 <= step < STEPS:
                    for state in states:
                        if state.rows:
                            rgb_attention_equalization(state, layer_mean(state.rows))
                for state in states:
                    state.record = False
                    state.apply_b = False
                x = _detach(linear_approximation_step(x, float(t_b - t_a), full))
                del full
                torch.cuda.empty_cache()
                print(json.dumps({"reference_step": step}), flush=True)
    finally:
        restore(saved)
    if shares is None:
        raise RuntimeError("reference shares were not recorded")
    return shares
