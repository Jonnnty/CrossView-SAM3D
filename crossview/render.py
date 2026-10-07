from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3d_objects.model.backbone.tdfy_dit.utils.render_utils import (
    render_frames,
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
)

from crossview.constants import OPACITY


def _tile(frame) -> Image.Image:
    if torch.is_tensor(frame):
        arr = frame.detach().float().cpu().numpy()
    else:
        arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype == np.uint8:
        rgb = arr[..., :3]
    elif arr.max() <= 1.0:
        rgb = (np.clip(arr, 0, 1) * 255).astype(np.uint8)[..., :3]
    else:
        rgb = np.clip(arr, 0, 255).astype(np.uint8)[..., :3]
    return Image.fromarray(rgb)


def render_five_views(gs, path: Path, resolution: int = 512) -> None:
    gs.from_opacity(torch.full_like(gs.get_opacity, OPACITY))
    offset = (-16 / 180 * np.pi, 20 / 180 * np.pi)
    yaw_off = offset[0]
    yaws = [0 + yaw_off, np.pi / 2 + yaw_off, np.pi + yaw_off, 3 * np.pi / 2 + yaw_off, yaw_off]
    pitches = [offset[1], offset[1], offset[1], offset[1], np.pi / 2 - 0.08]
    labels = ["front", "right", "back", "left", "top"]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitches, 10, 8)
    frames = render_frames(
        gs,
        extrinsics,
        intrinsics,
        {"resolution": resolution, "bg_color": (1, 1, 1), "backend": "gsplat"},
        verbose=False,
    )["color"]
    tiles = [_tile(frame) for frame in frames]
    w, h = tiles[0].size
    pad = 8
    cell_w, cell_h = w + pad, h + 28 + pad
    canvas = Image.new("RGB", (cell_w * len(tiles) - pad, cell_h), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    for i, (tile, lab) in enumerate(zip(tiles, labels)):
        x = i * cell_w
        canvas.paste(tile, (x, 0))
        draw.text((x + 6, h + 4), lab, fill=(20, 20, 20))
    canvas.save(path)
