from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from crossview.constants import ALPHA, GRID, SIGMA


def _surface_tokens(ijk: np.ndarray) -> np.ndarray:
    occ = np.zeros((GRID, GRID, GRID), np.uint8)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1
    surf = np.zeros_like(occ, bool)
    for ax, shift in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        if shift == 1:
            src[ax] = slice(0, GRID - 1)
            dst[ax] = slice(1, GRID)
        else:
            src[ax] = slice(1, GRID)
            dst[ax] = slice(0, GRID - 1)
        nb = np.zeros_like(occ)
        nb[tuple(dst)] = occ[tuple(src)]
        surf |= (occ == 1) & (nb == 0)
    return np.clip(np.argwhere(surf) // 4, 0, 15)


def equalization_discrepancy(equalized: list[np.ndarray], unregularized: list[np.ndarray]) -> np.ndarray:
    out = []
    for eq, raw in zip(equalized, unregularized):
        delta = np.abs(
            gaussian_filter(eq.astype(np.float64), SIGMA, mode="nearest")
            - gaussian_filter(raw.astype(np.float64), SIGMA, mode="nearest")
        )
        out.append(delta)
    return np.stack(out)


def discrepancy_fusion_weight(discrepancy: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    tokens = _surface_tokens(occupancy)
    scaled = []
    for i in range(discrepancy.shape[0]):
        field = discrepancy[i]
        tau = max(float(np.percentile(field[tokens[:, 0], tokens[:, 1], tokens[:, 2]], 95)), 1e-6)
        scaled.append(np.clip(field / tau, 0.0, 1.0))
    score = -ALPHA * np.stack(scaled)
    score -= score.max(axis=0, keepdims=True)
    weight = np.exp(score)
    weight /= np.maximum(weight.sum(axis=0, keepdims=True), 1e-12)
    return weight.astype(np.float32)
