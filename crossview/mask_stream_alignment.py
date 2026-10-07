from __future__ import annotations

import numpy as np

from crossview.constants import (
    B_CLIP,
    CTRL_EPS,
    DEADZONE,
    FOCAL_T,
    GAMMA,
    KD,
    KI,
    KP,
    LAMBDA_BG,
    LAMBDA_MAG,
    MAG_BOOST,
    MU,
    NGRID,
    W_FLOOR,
)
from crossview.rgb_attention_equalization import _blur, _scale, _softmax, _tau, _to01


class AlignmentState:
    def __init__(self, support: np.ndarray, target: np.ndarray, salient: np.ndarray):
        self.support = support
        self.target = target
        self.salient = salient
        self.B = np.zeros((NGRID, NGRID), np.float32)
        self.I = np.zeros((NGRID, NGRID), np.float32)
        self.e_prev = None
        self.lo = None
        self.hi = None
        self.target_hat = None
        self.apply_b = False
        self.record = False
        self.rows: list[tuple[int, np.ndarray]] = []


def mask_stream_alignment(state: AlignmentState, response: np.ndarray) -> None:
    support = state.support
    target = state.target
    if state.lo is None or state.hi is None:
        state.lo, state.hi = _scale(response, support)
    if state.target_hat is None:
        state.target_hat = _softmax(target, support, _tau(target, support))
    level = _to01(response, support, state.lo, state.hi)
    rho = np.zeros_like(target, np.float32)
    rho[support] = np.log((target[support] + CTRL_EPS) / (level[support] + CTRL_EPS))
    gain = np.zeros_like(target, np.float32)
    gain[support] = W_FLOOR + (1.0 - W_FLOOR) * np.power(np.clip(target[support], 0, 1), FOCAL_T)
    gain[state.salient] *= MAG_BOOST
    background = np.zeros_like(target, np.float32)
    background[support] = np.maximum(level[support] - target[support], 0.0) * (1.0 - target[support])
    extra = np.zeros_like(target, np.float32)
    extra[state.salient] = LAMBDA_MAG * (target[state.salient] - level[state.salient])
    residual = gain * rho - LAMBDA_BG * background + extra
    dead = ((np.abs(target - level) < DEADZONE) & ~state.salient) | ~support
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
