from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from sam3d_objects.model.backbone.generator.flow_matching.solver import linear_approximation_step

from crossview.attention import install_mask_stream_alignment, layer_mean, restore
from crossview.connected_component import largest_26_connected_component
from crossview.constants import RGB_LEN, SEED, STEPS
from crossview.discrepancy_fusion import (
    _detach,
    _noise,
    collect_leave_one_out_shape_speeds,
    discrepancy_weighted_velocity_fusion,
)
from crossview.equalization_discrepancy import discrepancy_fusion_weight, equalization_discrepancy
from crossview.mask_stream_alignment import AlignmentState, mask_stream_alignment
from crossview.render import render_five_views
from crossview.rgb_attention_equalization import EqualizationState
from crossview.structural_target import mask_support, structural_target
from crossview.structured_latent_fusion import align_occupancy, structured_latent_fusion


def load_pipeline(root: Path):
    cfg = root / "checkpoints" / "hf" / "pipeline.yaml"
    if not cfg.is_file():
        raise FileNotFoundError(f"missing {cfg}")
    config = OmegaConf.load(str(cfg))
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = str(cfg.parent)
    return instantiate(config)


def _views(folder: Path):
    masks = {p.stem: p for p in (folder / "masks").glob("*.png") if "eat" not in p.stem}
    found = []
    for path in folder.iterdir():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if path.stem not in masks:
            continue
        found.append((path.stem, path, masks[path.stem]))
    found.sort(key=lambda item: int(item[0]) if item[0].isdigit() else item[0])
    if len(found) < 2:
        raise RuntimeError(f"need at least two views in {folder}")
    return found


def _load_mask(path: Path, shape) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"))
    if mask.shape[:2] != shape:
        mask = np.array(Image.fromarray(mask).resize((shape[1], shape[0]), Image.NEAREST))
    return mask > 127


def _fit_mask(mask: np.ndarray, shape) -> np.ndarray:
    if mask.shape[:2] == shape:
        return mask
    resized = np.array(
        Image.fromarray((mask.astype(np.uint8) * 255)).resize((shape[1], shape[0]), Image.NEAREST)
    )
    return resized > 127


def _rgba(photo: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.concatenate([photo, (mask.astype(np.uint8) * 255)[..., None]], axis=-1)


def _embed(pipeline, photo: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    rgba = _rgba(photo, mask)
    with pipeline.device:
        pointmap = pipeline.compute_pointmap(rgba, pointmap=None)["pointmap"]
        ss_input = pipeline.preprocess_image(rgba, pipeline.ss_preprocessor, pointmap=pointmap)
        args, _kwargs = pipeline.get_condition_input(
            pipeline.condition_embedders["ss_condition_embedder"],
            ss_input,
            pipeline.ss_condition_input_mapping,
        )
    cond = args[0].float()
    if cond.ndim == 2:
        cond = cond.unsqueeze(0)
    return cond


def _park(pipeline, keep: set[str]) -> None:
    for name, model in pipeline.models.items():
        if model is None:
            continue
        if name in keep:
            model.cuda()
        else:
            model.cpu()
    torch.cuda.empty_cache()


def _occupancy(decoder, latent) -> np.ndarray:
    decoder.cuda()
    decoder.eval()
    shape = latent["shape"]
    ss = decoder(shape.permute(0, 2, 1).contiguous().view(shape.shape[0], 8, 16, 16, 16))
    return torch.argwhere(ss[0, 0] > 0).detach().cpu().numpy().astype(np.int32)


def _place_depth(pipeline, cuda: bool) -> None:
    depth = pipeline.depth_model
    if cuda:
        depth.model.cuda()
        depth.device = torch.device("cuda")
    else:
        depth.model.cpu()
        depth.device = torch.device("cpu")
    torch.cuda.empty_cache()


def _prepare(gen):
    gen.inference_steps = STEPS
    gen.no_shortcut = True
    gen.eval()
    backbone = gen.reverse_fn.backbone
    for block in backbone.blocks:
        block.use_checkpoint = False
    latent_shape = {
        key: (1,) + (module.pos_emb.shape[0], module.input_layer.in_features)
        for key, module in backbone.latent_mapping.items()
    }
    return backbone, latent_shape


def _primary_occupancy(gen, decoder, latent_shape, condition) -> np.ndarray:
    x = _noise(gen, latent_shape, condition.device)
    t_seq, d_val = gen._prepare_t_and_d()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
        for t_a, t_b in zip(t_seq[:STEPS], t_seq[1 : STEPS + 1]):
            vel = gen._generate_dynamics(x, float(t_a), d_val, condition)
            x = _detach(linear_approximation_step(x, float(t_b - t_a), vel))
            del vel
    return _occupancy(decoder, x)


def _fuse(gen, decoder, backbone, latent_shape, conditions, support, target, salient, weight):
    states = []
    for _cond in conditions:
        state = AlignmentState(support, target, salient)
        state.salient = salient
        states.append(state)
    saved = install_mask_stream_alignment(backbone, states)
    x = _noise(gen, latent_shape, conditions[0].device)
    t_seq, d_val = gen._prepare_t_and_d()
    w = torch.as_tensor(weight.reshape(weight.shape[0], -1), device=conditions[0].device)
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
            for step, (t_a, t_b) in enumerate(zip(t_seq[:STEPS], t_seq[1 : STEPS + 1]), start=1):
                velocities = []
                for cond, state in zip(conditions, states):
                    for other in states:
                        other.apply_b = False
                        other.record = False
                        other.rows = []
                    state.apply_b = True
                    state.record = True
                    velocities.append(gen._generate_dynamics(x, float(t_a), d_val, cond))
                    if state.rows:
                        mask_stream_alignment(state, layer_mean(state.rows))
                for state in states:
                    state.apply_b = False
                    state.record = False
                x = _detach(linear_approximation_step(x, float(t_b - t_a), discrepancy_weighted_velocity_fusion(velocities, w)))
                del velocities
                if step in (1, STEPS):
                    print(json.dumps({"fusion_step": step}), flush=True)
                torch.cuda.empty_cache()
        occ = _occupancy(decoder, x)
    finally:
        restore(saved)
    return occ


def run_object(pipeline, folder: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    views = _views(folder)
    photos, masks = [], []
    for _stem, image_path, mask_path in views:
        photo = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = _load_mask(mask_path, photo.shape[:2])
        photos.append(photo)
        masks.append(mask)
    _place_depth(pipeline, True)
    for emb in pipeline.condition_embedders.values():
        emb.cuda()
    _park(pipeline, {"ss_generator", "ss_decoder"})
    embedded = [_embed(pipeline, photo, mask) for photo, mask in zip(photos, masks)]
    for emb in pipeline.condition_embedders.values():
        emb.cpu()
    torch.cuda.empty_cache()
    own = [torch.cat([cond[:, :RGB_LEN], cond[:, RGB_LEN : 2 * RGB_LEN]], dim=1) for cond in embedded]
    mask1 = embedded[0][:, RGB_LEN : 2 * RGB_LEN]
    aligned = [torch.cat([cond[:, :RGB_LEN], mask1], dim=1) for cond in embedded]
    concat = torch.cat(own, dim=1)
    gen = pipeline.models["ss_generator"]
    decoder = pipeline.models["ss_decoder"]
    backbone, latent_shape = _prepare(gen)
    support, target = structural_target(pipeline, photos[0], masks[0])
    _place_depth(pipeline, False)
    salient = target.astype(bool) & support
    print(json.dumps({"object": folder.name, "support": int(support.sum()), "salient": int(salient.sum())}), flush=True)
    states = [EqualizationState(mask_support(photo, mask)) for photo, mask in zip(photos, masks)]
    print("unregularized concat", flush=True)
    raw_speeds, raw_latent = collect_leave_one_out_shape_speeds(gen, backbone, latent_shape, concat, states, False)
    raw_occ = _occupancy(decoder, raw_latent)
    states = [EqualizationState(mask_support(photo, mask)) for photo, mask in zip(photos, masks)]
    print("equalized concat", flush=True)
    eq_speeds, _eq_latent = collect_leave_one_out_shape_speeds(gen, backbone, latent_shape, concat, states, True)
    weight = discrepancy_fusion_weight(equalization_discrepancy(eq_speeds, raw_speeds), raw_occ)
    print(json.dumps({"weight_mean": [float(v) for v in weight.mean(axis=(1, 2, 3))]}), flush=True)
    print("primary view", flush=True)
    reference = _primary_occupancy(gen, decoder, latent_shape, aligned[0])
    print("discrepancy-weighted fusion", flush=True)
    fused = _fuse(gen, decoder, backbone, latent_shape, aligned, support, target, salient, weight)
    kept = largest_26_connected_component(fused)
    print(json.dumps({"occupancy": int(len(fused)), "kept": int(len(kept))}), flush=True)
    if len(kept) == 0:
        print(json.dumps({"stage2": "skipped", "reason": "empty occupancy"}), flush=True)
        return
    tokens = np.clip(kept // 4, 0, 15)
    point_w = weight[:, tokens[:, 0], tokens[:, 1], tokens[:, 2]]
    coords, coord_w, snap, shift = align_occupancy(kept, point_w, reference)
    print(json.dumps({"snap": snap, "shift": shift.tolist(), "aligned": int(len(coords))}), flush=True)
    for emb in pipeline.condition_embedders.values():
        emb.cuda()
    _park(pipeline, {"slat_generator", "slat_decoder_gs"})
    rgba_views = [_rgba(photo, _fit_mask(masks[0], photo.shape[:2])) for photo in photos]
    gs = structured_latent_fusion(pipeline, rgba_views, coords, coord_w)
    gs.save_ply(str(out / "splat.ply"))
    render_five_views(gs, out / "stage2.png")
    print(json.dumps({"stage2": str(out / "stage2.png"), "gaussians": int(gs.get_xyz.shape[0])}), flush=True)
