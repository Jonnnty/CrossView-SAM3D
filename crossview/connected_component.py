from __future__ import annotations

import numpy as np
from scipy.ndimage import generate_binary_structure, label


def largest_26_connected_component(ijk: np.ndarray) -> np.ndarray:
    ijk = np.unique(np.asarray(ijk, np.int64), axis=0)
    if len(ijk) == 0:
        return ijk.astype(np.int32)
    lo = ijk.min(axis=0)
    hi = ijk.max(axis=0)
    local = ijk - lo
    occ = np.zeros(tuple(int(v) for v in (hi - lo + 1)), np.uint8)
    occ[local[:, 0], local[:, 1], local[:, 2]] = 1
    labeled, n_cc = label(occ, structure=generate_binary_structure(3, 3))
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    main = int(np.argmax(sizes))
    kept = np.argwhere(labeled == main) + lo
    return kept.astype(np.int32)
