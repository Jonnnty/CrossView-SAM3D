from __future__ import annotations

import numpy as np
import torch

from sam3d_objects.model.backbone.generator.flow_matching.solver import linear_approximation_step

from crossview.constants import EQUALIZE_FROM, SEED, SIDE, STEPS
from crossview.attention import install_rgb_attention_equalization, layer_mean, restore
from crossview.rgb_attention_equalization import rgb_attention_equalization


def _detach(x):
    if isinstance(x, dict):
        return {k: v.detach() for k, v in x.items()}
    return x.detach()


def _noise(gen, latent_shape, device):
    torch.manual_seed(SEED)
    return _detach(gen._generate_noise(latent_shape, device))


def leave_one_out_shape_speed(full, dropped) -> np.ndarray:
    delta = full["shape"] - dropped["shape"]
    return delta.detach().float().reshape(-1, 8).norm(dim=-1).cpu().numpy().reshape(SIDE, SIDE, SIDE)


def discrepancy_weighted_velocity_fusion(velocities, weight: torch.Tensor):
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


def collect_leave_one_out_shape_speeds(gen, backbone, latent_shape, condition, states, equalize: bool):
    holder = {"equalize": equalize, "drop": False, "segment": 0}
    saved = install_rgb_attention_equalization(backbone, states, holder)
    speeds = []
    x = _noise(gen, latent_shape, condition.device)
    t_seq, d_val = gen._prepare_t_and_d()
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
            for step, (t_a, t_b) in enumerate(zip(t_seq[:STEPS], t_seq[1 : STEPS + 1]), start=1):
                for state in states:
                    state.apply_b = equalize and step >= EQUALIZE_FROM
                    state.record = equalize and step >= EQUALIZE_FROM - 1
                    state.rows = []
                holder["drop"] = False
                full = gen._generate_dynamics(x, float(t_a), d_val, condition)
                if step == STEPS:
                    for state in states:
                        state.record = False
                    for segment in range(len(states)):
                        holder["segment"] = segment
                        holder["drop"] = True
                        dropped = gen._generate_dynamics(x, float(t_a), d_val, condition)
                        holder["drop"] = False
                        speeds.append(leave_one_out_shape_speed(full, dropped))
                        del dropped
                if equalize and EQUALIZE_FROM - 1 <= step < STEPS:
                    for state in states:
                        if state.rows:
                            rgb_attention_equalization(state, layer_mean(state.rows))
                for state in states:
                    state.record = False
                    state.apply_b = False
                x = _detach(linear_approximation_step(x, float(t_b - t_a), full))
                del full
                torch.cuda.empty_cache()
    finally:
        restore(saved)
    return speeds, x
