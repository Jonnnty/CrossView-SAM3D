from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from crossview.constants import B_CLIP, CTRL_EPS, DEADZONE, GAMMA, KD, KI, KP, MU, NGRID, TAU_MULT


class EqualizationState:
    def __init__(self, support: np.ndarray):
        self.support = support
        self.B = np.zeros((NGRID, NGRID), np.float32)
        self.I = np.zeros((NGRID, NGRID), np.float32)
        self.e_prev = None
        self.lo = None
        self.hi = None
        self.target_hat = None
        self.apply_b = False
        self.record = False
        self.rows: list[tuple[int, np.ndarray]] = []


def _scale(grid: np.ndarray, support: np.ndarray):
    v = np.asarray(grid, np.float32)[support]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [8, 92])
    return float(lo), float(hi)


def _to01(grid: np.ndarray, support: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.zeros(grid.shape, np.float32)
    t = np.clip((np.asarray(grid, np.float32) - lo) / max(hi - lo, 1e-12), 0, 1)
    out[support] = t[support]
    return out


def _softmax(grid: np.ndarray, support: np.ndarray, tau: float) -> np.ndarray:
    out = np.zeros(grid.shape, np.float32)
    if not support.any():
        return out
    z = np.asarray(grid, np.float32)[support] / max(float(tau), 1e-12)
    z = z - float(z.max())
    e = np.exp(z)
    out[support] = (e / max(float(e.sum()), 1e-12)).astype(np.float32)
    return out


def _tau(grid: np.ndarray, support: np.ndarray) -> float:
    v = np.asarray(grid, np.float32)[support]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    return float(max(v.std() * TAU_MULT, 1e-12))


def _blur(grid: np.ndarray, support: np.ndarray) -> np.ndarray:
    g = np.where(support, grid, 0.0).astype(np.float32)
    num = uniform_filter(g, size=3, mode="nearest")
    den = uniform_filter(support.astype(np.float32), size=3, mode="nearest")
    return np.where(support, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)


def rgb_attention_equalization(state: EqualizationState, response: np.ndarray) -> None:
    support = state.support
    if state.lo is None or state.hi is None:
        state.lo, state.hi = _scale(response, support)
    level = _to01(response, support, state.lo, state.hi)
    if state.target_hat is None:
        state.target_hat = np.zeros_like(level)
        state.target_hat[support] = 1.0 / max(int(support.sum()), 1)
    target = np.zeros_like(level)
    target[support] = float(level[support].mean()) if support.any() else 0.5
    share = _softmax(response, support, _tau(response, support))
    rho = np.zeros_like(target, np.float32)
    rho[support] = np.log((target[support] + CTRL_EPS) / (level[support] + CTRL_EPS))
    share_e = np.zeros_like(target, np.float32)
    share_e[support] = np.log((state.target_hat[support] + CTRL_EPS) / (share[support] + CTRL_EPS))
    residual = rho + share_e
    dead = (np.abs(target - level) < DEADZONE) | ~support
    residual[dead] = 0.0
    smoothed = _blur(residual, support)
    sat_hi = (state.B >= B_CLIP - 1e-6) & (residual > 0)
    sat_lo = (state.B <= -B_CLIP + 1e-6) & (residual < 0)
    freeze = sat_hi | sat_lo
    integral = state.I + residual
    integral[freeze] = state.I[freeze]
    integral[~support] = 0.0
    state.I = integral.astype(np.float32)
    delta = np.zeros_like(residual) if state.e_prev is None else residual - state.e_prev
    state.e_prev = residual.astype(np.float32)
    proposed = GAMMA * (KP * smoothed + KI * state.I - KD * delta)
    state.B = np.clip(MU * state.B + (1.0 - MU) * proposed, -B_CLIP, B_CLIP).astype(np.float32)
    state.B[~support] = 0.0
