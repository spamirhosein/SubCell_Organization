"""Apply Cellpose segmentation masks to TIFF images in batch.

This module walks an input directory tree, pairs each image with its
corresponding ``*_seg.npy`` Cellpose mask, applies the mask to remove
background, and writes the result to a mirrored output directory along
with a copy of the original mask file.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile as tiff

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR: Path = Path(r"D:\image_data\Hi-res_Data\Phase 3")
OUTPUT_DIR: Path = Path(r"D:\image_data\Hi-res_Data\No_BG")
IMAGE_EXT: str = ".tiff"
MASK_SUFFIX: str = "_seg.npy"
# Directory-name prefixes that mark outputs from previous runs. Any image
# living under such a directory is skipped during discovery so the script
# never re-processes its own outputs when OUTPUT_DIR is nested inside
# INPUT_DIR.
EXCLUDE_DIR_PREFIXES: tuple[str, ...] = ("No_BG",)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def find_image_files(
    root: Path,
    exclude: Path,
    exclude_dir_prefixes: tuple[str, ...] = EXCLUDE_DIR_PREFIXES,
) -> list[Path]:
    """Return all image files under ``root``, skipping prior-run outputs.

    A file is excluded when it lives under ``exclude`` or under any ancestor
    directory whose name starts with one of ``exclude_dir_prefixes``. This
    prevents the script from re-discovering its own outputs when ``exclude``
    is nested inside ``root``.
    """
    results: list[Path] = []
    for path in root.rglob(f"*{IMAGE_EXT}"):
        if exclude in path.parents or path == exclude:
            continue
        ancestor_names = path.relative_to(root).parts[:-1]
        if any(name.startswith(exclude_dir_prefixes) for name in ancestor_names):
            continue
        results.append(path)
    return results


# ---------------------------------------------------------------------------
# Mask handling
# ---------------------------------------------------------------------------
def load_mask(mask_path: Path) -> np.ndarray:
    """Load a Cellpose mask array from ``mask_path``.

    Cellpose stores masks either as a raw ``ndarray`` or as a zero-dimensional
    object array wrapping a dictionary with a ``"masks"`` entry. Both formats
    are normalised to a plain ``ndarray``.
    """
    data = np.load(mask_path, allow_pickle=True)
    if data.shape == ():
        return data.item()["masks"]
    return data


def apply_mask(image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Apply a 2D binary mask to a 2D or 3D image.

    Supports channel-first ``(C, H, W)`` and channel-last ``(H, W, C)`` layouts,
    as well as single-channel 2D images. Returns ``None`` when the image and
    mask shapes are incompatible.
    """
    binary = mask > 0

    if image.ndim == 2 and image.shape == mask.shape:
        return image * binary
    if image.ndim == 3:
        if image.shape[1:] == mask.shape:
            return image * binary[np.newaxis, :, :]
        if image.shape[:2] == mask.shape:
            return image * binary[:, :, np.newaxis]
    return None


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------
def process_image(image_path: Path, input_dir: Path, output_dir: Path) -> bool:
    """Mask a single image and mirror the result into ``output_dir``.

    Returns ``True`` on success and ``False`` if the image was skipped or
    failed to process.
    """
    mask_path = image_path.with_name(image_path.stem + MASK_SUFFIX)
    if not mask_path.exists():
        logger.warning(
            "Skipping %s: mask %s not found.", image_path.name, mask_path.name
        )
        return False

    logger.info("Processing %s", image_path.name)
    try:
        image = tiff.imread(image_path)
        mask = load_mask(mask_path)

        masked = apply_mask(image, mask)
        if masked is None:
            logger.error(
                "Shape mismatch for %s: image=%s, mask=%s.",
                image_path.name,
                image.shape,
                mask.shape,
            )
            return False

        relative_path = image_path.relative_to(input_dir)
        output_image_path = output_dir / relative_path
        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        tiff.imwrite(output_image_path, masked)
        shutil.copy2(
            mask_path, output_image_path.with_name(mask_path.name)
        )

        logger.info("Saved masked image to %s", output_image_path)
        return True
    except Exception:
        logger.exception("Failed to process %s", image_path.name)
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> int:
    """Process every image under ``input_dir`` and return the success count."""
    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        return 0

    logger.info("Scanning directory: %s", input_dir)
    images = find_image_files(input_dir, exclude=output_dir)
    logger.info("Found %d %s files to process.", len(images), IMAGE_EXT)

    processed = sum(
        process_image(image, input_dir, output_dir) for image in images
    )
    logger.info(
        "Batch complete. Successfully processed %d of %d files.",
        processed,
        len(images),
    )
    return processed


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())