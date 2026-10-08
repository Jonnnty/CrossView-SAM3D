from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from sam3d_objects.model.backbone.generator.flow_matching.solver import linear_approximation_step

from crossview.attention import install_share_alignment, restore
from crossview.constants import SIDE, STEPS, VIEW_LEN
from crossview.gap_fusion import (
    _detach,
    _noise,
    gap_fusion_weight,
    reference_mask_shares,
    shape_speed_gap,
    weighted_velocity_fusion,
)
from crossview.render import render_five_views
from crossview.rgb_attention_equalization import EqualizationState
from crossview.share_alignment import ShareAlignment, aligned_dynamics, reference_dynamics
from crossview.structural_target import mask_support
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


def _fuse(gen, decoder, backbone, latent_shape, conditions, states):
    holder = {"align": states[0], "sink": [], "apply": False, "record": False}
    saved = install_share_alignment(backbone, holder)
    x = _noise(gen, latent_shape, conditions[0].device)
    t_seq, d_val = gen._prepare_t_and_d()
    weight = None
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
            for step, (t_a, t_b) in enumerate(zip(t_seq[:STEPS], t_seq[1 : STEPS + 1]), start=1):
                aligned = []
                for view, cond in enumerate(conditions):
                    aligned.append(aligned_dynamics(gen, x, t_a, d_val, cond, states[view], holder, step, view))
                gaps = []
                for view, cond in enumerate(conditions):
                    reference = reference_dynamics(gen, x, t_a, d_val, cond, holder)
                    gaps.append(shape_speed_gap(aligned[view], reference))
                    del reference
                weight = gap_fusion_weight(np.stack(gaps))
                w = torch.as_tensor(weight.reshape(len(conditions), -1), device=conditions[0].device)
                mixed = weighted_velocity_fusion(aligned, w)
                x = _detach(linear_approximation_step(x, float(t_b - t_a), mixed))
                print(
                    json.dumps(
                        {
                            "step": step,
                            "weight_mean": [float(v) for v in weight.mean(axis=(1, 2, 3))],
                            "weight_min": [float(v) for v in weight.reshape(len(conditions), -1).min(1)],
                        }
                    ),
                    flush=True,
                )
                del aligned, mixed
                torch.cuda.empty_cache()
        occ = _occupancy(decoder, x)
    finally:
        restore(saved)
    return occ, weight


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
    _place_depth(pipeline, False)
    torch.cuda.empty_cache()
    width = int(embedded[0].shape[1])
    if any(int(cond.shape[1]) != width for cond in embedded) or width != VIEW_LEN:
        raise RuntimeError(f"expected {VIEW_LEN} tokens per view, got {width}")
    print(json.dumps({"object": folder.name, "tokens_per_view": width, "views": len(embedded)}), flush=True)
    gen = pipeline.models["ss_generator"]
    decoder = pipeline.models["ss_decoder"]
    backbone, latent_shape = _prepare(gen)
    supports = [mask_support(photo, mask) for photo, mask in zip(photos, masks)]
    print("equalized concatenation", flush=True)
    shares = reference_mask_shares(
        gen,
        backbone,
        latent_shape,
        embedded,
        [EqualizationState(support) for support in supports],
    )
    print("primary view", flush=True)
    reference = _primary_occupancy(gen, decoder, latent_shape, embedded[0])
    print("gap-weighted fusion", flush=True)
    states = [ShareAlignment(support, share) for support, share in zip(supports, shares)]
    fused, weight = _fuse(gen, decoder, backbone, latent_shape, embedded, states)
    print(json.dumps({"occupancy": int(len(fused))}), flush=True)
    if len(fused) == 0 or weight is None:
        print(json.dumps({"stage2": "skipped", "reason": "empty occupancy"}), flush=True)
        return
    tokens = np.clip(fused // 4, 0, SIDE - 1)
    point_w = weight[:, tokens[:, 0], tokens[:, 1], tokens[:, 2]]
    coords, coord_w, snap, shift = align_occupancy(fused, point_w, reference)
    print(
        json.dumps(
            {
                "snap": snap,
                "shift": shift.tolist(),
                "aligned": int(len(coords)),
                "color_weight_mean": [float(v) for v in coord_w.mean(axis=1)],
            }
        ),
        flush=True,
    )
    for emb in pipeline.condition_embedders.values():
        emb.cuda()
    _park(pipeline, {"slat_generator", "slat_decoder_gs"})
    rgba_views = [_rgba(photo, _fit_mask(masks[0], photo.shape[:2])) for photo in photos]
    gs = structured_latent_fusion(pipeline, rgba_views, coords, coord_w)
    gs.save_ply(str(out / "splat.ply"))
    render_five_views(gs, out / "stage2.png")
    print(json.dumps({"stage2": str(out / "stage2.png"), "gaussians": int(gs.get_xyz.shape[0])}), flush=True)
