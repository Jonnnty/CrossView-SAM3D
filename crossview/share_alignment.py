from __future__ import annotations

import json

import numpy as np

from crossview.attention import layer_mean
from crossview.constants import (
    ALIGN_B_LIMIT,
    ALIGN_EPS,
    ALIGN_MAX_INNER,
    ALIGN_TOL_L1,
    ALIGN_TOL_MAX,
    NGRID,
)


def on_support(grid: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid, dtype=np.float32)
    mass = float(grid[support].sum())
    if mass > 1e-12:
        out[support] = grid[support] / mass
    return out


class ShareAlignment:
    def __init__(self, support: np.ndarray, target: np.ndarray):
        self.support = support
        self.target = target.astype(np.float32)
        self.layers: dict[int, np.ndarray] = {}

    def bias(self, layer: int) -> np.ndarray:
        grid = self.layers.get(layer)
        if grid is None:
            grid = np.zeros((NGRID, NGRID), np.float32)
            self.layers[layer] = grid
        return grid

    def update(self, rows: list[tuple[int, np.ndarray]]) -> None:
        last: dict[int, np.ndarray] = {}
        for layer, grid in rows:
            last[layer] = grid
        support = self.support
        target = np.clip(self.target[support], 0, None)
        for layer, grid in last.items():
            share = on_support(grid, support)
            err = np.log((target + ALIGN_EPS) / (np.clip(share[support], 0, None) + ALIGN_EPS)).astype(np.float32)
            err -= float(err.mean()) if err.size else 0.0
            bias = self.bias(layer)
            bias[support] = np.clip(bias[support] + err, -ALIGN_B_LIMIT, ALIGN_B_LIMIT)


def _residual(share: np.ndarray, state: ShareAlignment) -> tuple[float, float]:
    diff = np.abs(share[state.support] - state.target[state.support])
    if diff.size == 0:
        return 0.0, 0.0
    return float(diff.sum()), float(diff.max())


def aligned_dynamics(gen, x, t_a, d_val, cond, state: ShareAlignment, holder: dict, step: int, view: int):
    holder["align"] = state
    holder["apply"] = True
    holder["record"] = True
    full = None
    gap = peak = 0.0
    for inner in range(1, ALIGN_MAX_INNER + 1):
        holder["sink"].clear()
        full = gen._generate_dynamics(x, float(t_a), d_val, cond)
        share = on_support(layer_mean(holder["sink"]), state.support)
        gap, peak = _residual(share, state)
        if (gap <= ALIGN_TOL_L1 and peak <= ALIGN_TOL_MAX) or inner == ALIGN_MAX_INNER:
            print(
                json.dumps({"step": step, "view": view + 1, "inner": inner, "l1": gap, "max_abs": peak}),
                flush=True,
            )
            break
        state.update(holder["sink"])
        del full
    return full


def reference_dynamics(gen, x, t_a, d_val, cond, holder: dict):
    holder["apply"] = False
    holder["record"] = False
    holder["sink"].clear()
    return gen._generate_dynamics(x, float(t_a), d_val, cond)
