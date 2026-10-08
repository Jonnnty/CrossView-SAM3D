from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import crop_around_mask_with_padding
from sam3d_objects.data.dataset.tdfy.img_processing import pad_to_square_centered

from crossview.constants import NGRID, PATCH, SIZE


def _chw(photo: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(photo).permute(2, 0, 1).float() / 255.0


def _crop518(photo: np.ndarray, mask: np.ndarray):
    img = _chw(photo)
    m = torch.from_numpy(mask.astype(np.float32))
    _rgb, mm = crop_around_mask_with_padding(img, m, box_size_factor=1.2, padding_factor=0.0)
    mm = pad_to_square_centered(mm[None]).squeeze(0)
    m518 = resize(mm[None], [SIZE, SIZE], interpolation=InterpolationMode.NEAREST).squeeze(0)
    return m518.numpy() > 0.5


def mask_support(photo: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m518 = _crop518(photo, mask)
    on = np.zeros((NGRID, NGRID), bool)
    for r in range(NGRID):
        for c in range(NGRID):
            cell = m518[r * PATCH : (r + 1) * PATCH, c * PATCH : (c + 1) * PATCH]
            on[r, c] = float(cell.mean()) >= 0.08
    return on
