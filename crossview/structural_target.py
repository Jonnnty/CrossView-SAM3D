from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import uniform_filter
from scipy.spatial import ConvexHull
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import crop_around_mask_with_padding
from sam3d_objects.data.dataset.tdfy.img_processing import pad_to_square_centered

from crossview.constants import NGRID, PATCH, SIZE, WINDOW

K = WINDOW // 2


def _chw(photo: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(photo).permute(2, 0, 1).float() / 255.0


def _crop518(photo: np.ndarray, mask: np.ndarray):
    img = _chw(photo)
    m = torch.from_numpy(mask.astype(np.float32))
    rgb, mm = crop_around_mask_with_padding(img, m, box_size_factor=1.2, padding_factor=0.0)
    rgb = pad_to_square_centered(rgb)
    mm = pad_to_square_centered(mm[None]).squeeze(0)
    rgb518 = resize(rgb, [SIZE, SIZE], interpolation=InterpolationMode.BICUBIC, antialias=True)
    m518 = resize(mm[None], [SIZE, SIZE], interpolation=InterpolationMode.NEAREST).squeeze(0)
    crop = (rgb518.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return crop, m518.numpy() > 0.5


def mask_support(photo: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _crop, m518 = _crop518(photo, mask)
    on = np.zeros((NGRID, NGRID), bool)
    for r in range(NGRID):
        for c in range(NGRID):
            cell = m518[r * PATCH : (r + 1) * PATCH, c * PATCH : (c + 1) * PATCH]
            on[r, c] = float(cell.mean()) >= 0.08
    return on


def _depth518(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if depth.shape != mask.shape:
        depth = np.array(
            Image.fromarray(depth.astype(np.float32), mode="F").resize(
                (mask.shape[1], mask.shape[0]), Image.BILINEAR
            )
        )
    d = torch.from_numpy(depth.astype(np.float32))[None]
    m = torch.from_numpy(mask.astype(np.float32))
    d_c, _ = crop_around_mask_with_padding(d.expand(3, -1, -1), m, box_size_factor=1.2, padding_factor=0.0)
    d_c = pad_to_square_centered(d_c[:1])
    d518 = resize(d_c, [SIZE, SIZE], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return d518[0].numpy().astype(np.float32)


def _grid_depth(depth518: np.ndarray, m518: np.ndarray) -> np.ndarray:
    g = np.full((NGRID, NGRID), np.nan, np.float32)
    valid = m518 & np.isfinite(depth518) & (depth518 > 1e-6)
    for r in range(NGRID):
        for c in range(NGRID):
            sl = (slice(r * PATCH, (r + 1) * PATCH), slice(c * PATCH, (c + 1) * PATCH))
            w = valid[sl]
            if float(w.mean()) >= 0.25:
                g[r, c] = float(depth518[sl][w].mean())
    return g


def _norm01(a: np.ndarray) -> np.ndarray:
    vv = a[a > 0]
    if vv.size == 0:
        return a
    lo, hi = np.percentile(vv, [2, 98])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def _rgb_structure(crop: np.ndarray, m518: np.ndarray) -> np.ndarray:
    img = crop.astype(np.float32)
    gy, gx = np.gradient(img.mean(axis=2))
    edge = np.hypot(gx, gy)
    mean_rgb = np.zeros((NGRID, NGRID, 3), np.float32)
    mean_edge = np.zeros((NGRID, NGRID), np.float32)
    for r in range(NGRID):
        for c in range(NGRID):
            sl = (slice(r * PATCH, (r + 1) * PATCH), slice(c * PATCH, (c + 1) * PATCH))
            w = m518[sl]
            if w.any():
                mean_rgb[r, c] = img[sl][w].mean(axis=0)
                mean_edge[r, c] = edge[sl][w].mean()
    color = np.zeros((NGRID, NGRID), np.float32)
    for r in range(NGRID):
        for c in range(NGRID):
            y0, y1 = max(0, r - K), min(NGRID, r + K + 1)
            x0, x1 = max(0, c - K), min(NGRID, c + K + 1)
            nb = mean_rgb[y0:y1, x0:x1].reshape(-1, 3)
            color[r, c] = float(np.linalg.norm(mean_rgb[r, c] - nb.mean(axis=0)))
    return 0.5 * _norm01(color) + 0.5 * _norm01(mean_edge)


def _depth_vertices(z: np.ndarray, support: np.ndarray) -> np.ndarray:
    salient = np.zeros_like(support)
    vv = z[support]
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return salient
    lo, hi = float(np.percentile(vv, 2)), float(np.percentile(vv, 98))
    z01 = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    h, w = support.shape
    span = float(WINDOW)
    for y, x in np.argwhere(support):
        y0, y1 = max(0, y - K), min(h, y + K + 1)
        x0, x1 = max(0, x - K), min(w, x + K + 1)
        py, px = np.where(support[y0:y1, x0:x1])
        if py.size <= 3:
            salient[y, x] = True
            continue
        zz = z01[y0:y1, x0:x1][py, px]
        pts = np.stack([px.astype(np.float64), py.astype(np.float64), zz.astype(np.float64) * span], 1)
        uniq, inv = np.unique(np.round(pts, 5), axis=0, return_inverse=True)
        here = np.array([x - x0, y - y0, float(z01[y, x]) * span], np.float64)
        uid = int(inv[np.argmin(np.linalg.norm(pts - here, axis=1))])
        if len(uniq) < 4:
            salient[y, x] = True
            continue
        try:
            hull = ConvexHull(uniq)
        except Exception:
            salient[y, x] = True
            continue
        if uid not in set(int(i) for i in hull.vertices):
            continue
        salient[y, x] = float(z01[y, x]) <= float(np.median(zz))
    return salient


def structural_target(pipeline, photo: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgba = np.concatenate([photo, (mask.astype(np.uint8) * 255)[..., None]], axis=-1)
    with pipeline.device:
        pointmap = pipeline.compute_pointmap(rgba, pointmap=None)["pointmap"]
    if pointmap.ndim == 3 and pointmap.shape[0] == 3:
        depth = np.abs(pointmap[2].float().cpu().numpy())
    else:
        depth = pointmap.float().cpu().numpy()
        if depth.ndim == 3:
            depth = depth[..., 2]
        depth = np.abs(np.where(np.isfinite(depth), depth, 0.0))
    crop, m518 = _crop518(photo, mask)
    grid = _grid_depth(_depth518(depth.astype(np.float32), mask), m518)
    support = mask_support(photo, mask) & np.isfinite(grid)
    smoothed = uniform_filter(np.nan_to_num(grid, nan=0.0), size=WINDOW, mode="nearest")
    smoothed = np.where(support, smoothed, np.nan).astype(np.float32)
    depth_on = _depth_vertices(np.where(support, smoothed, 0.0).astype(np.float32), support)
    structure = _rgb_structure(crop, m518)
    rgb_on = np.zeros_like(support)
    h, w = support.shape
    for y, x in np.argwhere(support):
        y0, y1 = max(0, y - K), min(h, y + K + 1)
        x0, x1 = max(0, x - K), min(w, x + K + 1)
        vals = structure[y0:y1, x0:x1][support[y0:y1, x0:x1]]
        if vals.size < 4 or float(vals.std()) < 0.04:
            continue
        rgb_on[y, x] = structure[y, x] >= float(np.percentile(vals, 60))
    salient = depth_on | rgb_on
    target = np.zeros(support.shape, np.float32)
    target[support] = salient[support].astype(np.float32)
    return support, target
