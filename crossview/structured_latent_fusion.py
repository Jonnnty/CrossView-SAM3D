from __future__ import annotations

import numpy as np
import torch

from sam3d_objects.model.backbone.generator.flow_matching.solver import linear_approximation_step
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from sam3d_objects.pipeline.inference_utils import downsample_sparse_structure

from crossview.constants import GRID, SEED, STEPS


def _frame(points: np.ndarray) -> np.ndarray:
    center = points.mean(0)
    _u, _s, vt = np.linalg.svd(points - center, full_matrices=False)
    if np.dot(vt[0], [0.0, 0.0, 1.0]) < 0:
        vt = vt.copy()
        vt[0] *= -1
    if np.linalg.det(vt) < 0:
        vt = vt.copy()
        vt[1] *= -1
    return vt


def upright_yaw(src: np.ndarray, dst: np.ndarray) -> float:
    rot = _frame(dst).T @ _frame(src)
    return float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))


def apply_yaw(raw: np.ndarray, yaw: int) -> np.ndarray:
    x, y, z = raw[:, 0], raw[:, 1], raw[:, 2]
    if yaw == 0:
        return np.stack([x, y, z], axis=1)
    if yaw == 90:
        return np.stack([-y, x, z], axis=1)
    if yaw == 180:
        return np.stack([-x, -y, z], axis=1)
    return np.stack([y, -x, z], axis=1)


def align_occupancy(raw: np.ndarray, weights: np.ndarray, reference: np.ndarray):
    yaw = upright_yaw(raw.astype(np.float64), reference.astype(np.float64))
    snap = int(np.round(yaw / 90.0) * 90) % 360
    spun = apply_yaw(raw, snap)
    shift = np.rint(reference.astype(np.float64).mean(0) - spun.mean(0)).astype(np.int32)
    aligned = np.clip(spun + shift, 0, GRID - 1).astype(np.int32)
    order = np.lexsort((aligned[:, 2], aligned[:, 1], aligned[:, 0]))
    aligned = aligned[order]
    weights = weights[:, order]
    if len(aligned) > 1:
        change = np.any(np.diff(aligned, axis=0), axis=1)
        starts = np.r_[0, np.flatnonzero(change) + 1]
        ends = np.r_[starts[1:], len(aligned)]
        coords = aligned[starts]
        merged = np.stack([weights[:, s:e].mean(axis=1) for s, e in zip(starts, ends)], axis=1)
    else:
        coords, merged = aligned, weights
    merged = merged / np.maximum(merged.sum(axis=0, keepdims=True), 1e-12)
    return coords.astype(np.int32), merged.astype(np.float32), snap, shift


def structured_latent_fusion(pipeline, rgba_views, coords: np.ndarray, weight: np.ndarray):
    gen = pipeline.models["slat_generator"]
    gen.inference_steps = STEPS
    gen.no_shortcut = True
    gen.reverse_fn.strength = pipeline.slat_cfg_strength
    conditions = []
    with pipeline.device:
        for rgba in rgba_views:
            slat_input = pipeline.preprocess_image(rgba, pipeline.slat_preprocessor)
            args, _kwargs = pipeline.get_condition_input(
                pipeline.condition_embedders["slat_condition_embedder"],
                slat_input,
                pipeline.slat_condition_input_mapping,
            )
            conditions.append(args[0])
    coord = np.concatenate([np.zeros((len(coords), 1), np.int32), coords], axis=1)
    coord_t = torch.as_tensor(coord, device=pipeline.device, dtype=torch.int32)
    coord_t, scale = downsample_sparse_structure(coord_t)
    if int(scale) != 1:
        raise RuntimeError(f"sparse coordinates were rescaled by {scale}")
    coord_np = coord_t.detach().cpu().numpy()
    w = torch.as_tensor(weight, device=pipeline.device)
    torch.manual_seed(SEED)
    x = gen._generate_noise((1, coord_t.shape[0], 8), pipeline.device)
    t_seq = gen._prepare_t().to(pipeline.device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=pipeline.dtype):
        for t_a, t_b in zip(t_seq[:-1], t_seq[1:]):
            velocities = [gen._generate_dynamics(x, t_a, cond, coord_np).detach() for cond in conditions]
            acc = None
            for i, vel in enumerate(velocities):
                wi = w[i].to(dtype=vel.dtype, device=vel.device)
                if vel.ndim == 3 and vel.shape[1] == wi.shape[0]:
                    wi = wi.view(1, -1, 1)
                elif vel.ndim == 2 and vel.shape[0] == wi.shape[0]:
                    wi = wi.view(-1, 1)
                else:
                    raise RuntimeError(f"weight {tuple(wi.shape)} does not match velocity {tuple(vel.shape)}")
                term = vel * wi
                acc = term if acc is None else acc + term
            x = linear_approximation_step(x, float(t_b - t_a), acc).detach()
            del velocities
            torch.cuda.empty_cache()
    feats = x[0].float()
    slat = sp.SparseTensor(coords=coord_t, feats=feats).to(pipeline.device)
    slat = slat * pipeline.slat_std.to(pipeline.device) + pipeline.slat_mean.to(pipeline.device)
    outputs = pipeline.decode_slat(slat, ["gaussian"])
    outputs = pipeline.postprocess_slat_output(
        outputs, with_mesh_postprocess=False, with_texture_baking=False, use_vertex_color=True
    )
    gs = outputs.get("gs") or (outputs.get("gaussian") or [None])[0]
    if gs is None:
        raise RuntimeError(f"no gaussian {list(outputs)}")
    return gs
