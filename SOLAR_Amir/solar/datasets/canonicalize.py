from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from skimage import transform

# Canonicalization mirrors Stage 1 (_align_label):
# 1) translate cell centroid to image center
# 2) rotate long axis to the main diagonal
# 3) flip/rotate so nucleus lies in the upper-left triangle


def _centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("No pixels found for centroid computation")
    return float(xs.mean()), float(ys.mean())


def _pca_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    coords = np.column_stack((xs, ys))
    if coords.shape[0] < 2:
        return 0.0
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    return np.degrees(np.arctan2(principal[1], principal[0])) - 45.0


def _warp_per_channel(arr: np.ndarray, tform, order: int) -> np.ndarray:
    # arr: (C,H,W)
    warped = []
    for c in range(arr.shape[0]):
        warped.append(
            transform.warp(
                arr[c],
                tform.inverse,
                order=order,
                mode="constant",
                preserve_range=True,
            )
        )
    return np.stack(warped, axis=0)


def _rotate_per_channel(arr: np.ndarray, angle: float, order: int) -> np.ndarray:
    rotated = []
    for c in range(arr.shape[0]):
        rotated.append(
            transform.rotate(arr[c], angle, resize=False, order=order, mode="constant", preserve_range=True)
        )
    return np.stack(rotated, axis=0)


def canonicalize_label_and_stack(label: np.ndarray, stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Apply Stage 1 canonicalization to both label (H,W) and stack (C,H,W).

    Returns (label_aligned, stack_aligned, metadata).
    """
    if label.ndim != 2:
        raise ValueError("label must be 2D")
    if stack.ndim != 3:
        raise ValueError("stack must be (C,H,W)")
    h, w = label.shape
    meta: Dict[str, float] = {}

    # 1) translate cell centroid to center
    try:
        cell_cx, cell_cy = _centroid(label > 0)
        target = (w / 2.0, h / 2.0)
        translation = (target[0] - cell_cx, target[1] - cell_cy)
        tform = transform.AffineTransform(translation=translation)
        label = transform.warp(label, tform.inverse, order=0, mode="constant", preserve_range=True)
        stack = _warp_per_channel(stack, tform, order=1)
        meta["translation_x"] = translation[0]
        meta["translation_y"] = translation[1]
    except ValueError:
        pass

    # 2) rotate long axis to diagonal
    angle = _pca_angle(label > 0)
    if abs(angle) > 1e-3:
        label = transform.rotate(label, angle, resize=False, order=0, mode="constant", preserve_range=True)
        stack = _rotate_per_channel(stack, angle, order=1)
        meta["rotation_deg"] = angle
    else:
        meta["rotation_deg"] = 0.0

    # 3) ensure nucleus is upper-left-ish
    try:
        nuc_cx, nuc_cy = _centroid(label == 2)
        if nuc_cx + nuc_cy - label.shape[1] < 0:
            label = transform.rotate(label, 180, resize=False, order=0, mode="constant", preserve_range=True)
            stack = _rotate_per_channel(stack, 180, order=1)
            meta["flip_180"] = 1.0
        else:
            meta["flip_180"] = 0.0
        nuc_cx, nuc_cy = _centroid(label == 2)
        if nuc_cx - nuc_cy > 0:
            label = np.flipud(label)
            stack = np.flipud(stack)
            label = transform.rotate(label, -90, resize=False, order=0, mode="constant", preserve_range=True)
            stack = _rotate_per_channel(stack, -90, order=1)
            meta["flip_diag"] = 1.0
        else:
            meta["flip_diag"] = 0.0
    except ValueError:
        meta["flip_180"] = 0.0
        meta["flip_diag"] = 0.0

    return np.rint(label).astype(np.uint8), stack.astype(np.float32), meta


def downsample_stack(stack: np.ndarray, target: int) -> np.ndarray:
    if stack.shape[-1] == target and stack.shape[-2] == target:
        return stack
    return transform.resize(
        stack,
        (stack.shape[0], target, target),
        order=1,
        mode="constant",
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def downsample_mask(mask: np.ndarray, target: int) -> np.ndarray:
    if mask.shape[-1] == target and mask.shape[-2] == target:
        return mask
    if mask.ndim == 2:
        return transform.resize(
            mask,
            (target, target),
            order=0,
            mode="constant",
            preserve_range=True,
            anti_aliasing=True,
        )
    if mask.ndim == 3:
        return transform.resize(
            mask,
            (mask.shape[0], target, target),
            order=0,
            mode="constant",
            preserve_range=True,
            anti_aliasing=True,
        )
    raise ValueError("mask must be 2D or 3D")
