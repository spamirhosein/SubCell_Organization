#!/usr/bin/env python3
"""
flow_nuclear_seg.py
===================
Nuclear segmentation in two stages, both runnable from the notebook:

  STAGE 1 (run_cellpose_folder): run the Cellpose NUCLEUS model on your two-channel
           images (nucleus + membrane) and save a Cellpose *_seg.npy per FOV. The
           membrane channel helps the model decide where one nucleus ends.

  STAGE 2 (run_pipeline / batch): take those *_seg.npy files and rebuild the nuclear
           masks from the flow, using the SAME logic as the membrane pipeline:
             - flow divergence -> nucleus CENTERS (markers via h-maxima)
             - membrane wall   -> boundary WHERE a wall exists
             - flow separatrix -> boundary WHERE membrane is absent
           The membrane veto-merge is OFF by default here (use_veto=False): between
           two nuclei inside one cell there is often no membrane, so the veto would
           wrongly merge real touching nuclei.

Set MODEL_PATH below to your saved nucleus model. Set CELLPOSE_CHANNELS and
NUCLEUS_DIAMETER to match what worked for you in the Cellpose GUI.

Input  (stage 1): per-FOV subfolders of raw 2-channel images.
Input  (stage 2): the *_seg.npy files written by stage 1.
Output (stage 2): uint16 label mask (.npy + .tif) + per-nucleus table (.csv).

Author: packaged from an interactive session.
"""

from __future__ import annotations
import argparse
import glob
import os
from dataclasses import dataclass, asdict

import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import morphology, measure, segmentation
from skimage.morphology import skeletonize, dilation, disk
from skimage.segmentation import find_boundaries, relabel_sequential


# --------------------------------------------------------------------------- #
# Cellpose model / stage-1 configuration  --  EDIT THESE
# --------------------------------------------------------------------------- #
# Path to your saved nucleus model. Leave as "" to use Cellpose's built-in
# 'nuclei' model instead.
MODEL_PATH = "/Users/amir/.cellpose/models/nucleitorch_0"

# Cellpose channel setting for model.eval, as [primary, secondary].
# Match what worked in your GUI run. For nucleus + membrane, primary is the
# nucleus channel and secondary is the membrane. If your tiff has nucleus first,
# this is [1, 2]; if membrane first, [2, 1]. Set to match your data.
CELLPOSE_CHANNELS = [1, 2]

# Nucleus diameter in pixels. None = let Cellpose estimate. If auto looks wrong,
# set the value you used in the GUI.
NUCLEUS_DIAMETER = 100

# Cellpose eval thresholds (stage 1 only; these shape Cellpose's own output/flow).
FLOW_THRESHOLD     = 0.7       # <-- Max allowed flow error per mask (GUI default 0.4)
CELLPROB_THRESHOLD = 0.0       # <-- Foreground cutoff for Cellpose (GUI default 0.0)

# The image file inside each FOV subfolder is named after the folder itself,
# e.g. folder "B_4n_C01_R01/" contains "B_4n_C01_R01.tiff". Change the extension
# here if your files are ".tif".
IMAGE_EXT = ".tiff"            # <-- EDIT if your images are .tif


# --------------------------------------------------------------------------- #
# Parameters (stage 2 reconstruction)
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    # --- the three main knobs ---
    sink_h: float = 0.12        # h-maxima depth on -divergence (seed sensitivity)
    flow_weight: float = 1      # weight of the flow-separatrix term in the elevation
    veto_wall: float = 0.45     # merge adjacent labels if shared-border mean-membrane < this

    # --- secondary / rarely changed ---
    cellprob_thr: float = 0.0   # foreground = cellprob > this
    div_sigma: float = 2.0      # smoothing of the divergence field
    mem_sigma: float = 1.0      # smoothing of the membrane elevation term
    membrane_bin: float = 0.5   # threshold to skeletonize the membrane
    min_size: int = 80          # remove objects smaller than this (px)
    veto_min_border: int = 5    # only apply veto when shared border >= this many px
    nuc_thr: float = 0.45       # HH3 threshold for the nuclear-content confidence metric
    hole_ratio: float = 0.95    # delete a mask if real area < this * hole-filled area (has a hole)
    use_veto: bool = False      # NUCLEAR DEFAULT: veto OFF (no membrane between nuclei in one cell)



# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_seg(path: str) -> dict:
    """Load a Cellpose *_seg.npy dict."""
    d = np.load(path, allow_pickle=True).item()
    if "img" not in d or "flows" not in d:
        raise ValueError(f"{path}: not a Cellpose _seg.npy (missing 'img'/'flows').")
    return d


# --------------------------------------------------------------------------- #
# STAGE 1: run the Cellpose nucleus model and save *_seg.npy files
# --------------------------------------------------------------------------- #
def load_model(model_path: str = MODEL_PATH):
    """Load the nucleus model. Custom path if given, else Cellpose's built-in 'nuclei'."""
    from cellpose import models
    if model_path:
        print(f"Loading custom nucleus model: {model_path}")
        return models.CellposeModel(pretrained_model=model_path)
    print("Loading built-in Cellpose 'nuclei' model")
    return models.Cellpose(model_type="nuclei")


def run_cellpose_folder(input_dir: str, seg_dir: str, model=None,
                        channels=None, diameter=None,
                        flow_threshold=None, cellprob_threshold=None,
                        image_ext: str = IMAGE_EXT):
    """
    STAGE 1. For every per-FOV subfolder in input_dir, run the nucleus model on the
    2-channel image (named after the folder, e.g. FOV/FOV.tiff) and save a Cellpose
    *_seg.npy into seg_dir (flat, named by FOV). These are the input to stage 2.
    """
    from cellpose import io
    if model is None:
        model = load_model()
    if channels is None:
        channels = CELLPOSE_CHANNELS
    if diameter is None:
        diameter = NUCLEUS_DIAMETER
    if flow_threshold is None:
        flow_threshold = FLOW_THRESHOLD
    if cellprob_threshold is None:
        cellprob_threshold = CELLPROB_THRESHOLD

    folders = sorted(f for f in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(f))
    print(f"Stage 1: {len(folders)} FOV folders found\n")

    for i, folder in enumerate(folders, 1):
        name = os.path.basename(folder)
        img_path = os.path.join(folder, f"{name}{image_ext}")   # image named after its folder
        if not os.path.exists(img_path):
            print(f"  [{i}/{len(folders)}] SKIP {name} — {name}{image_ext} not found")
            continue

        img = tiff.imread(img_path).astype(np.float32)   # expected (2, H, W)

        # CellposeModel.eval returns (masks, flows, styles); Cellpose.eval adds diams.
        out = model.eval(img, channels=channels, diameter=diameter,
                         flow_threshold=flow_threshold,
                         cellprob_threshold=cellprob_threshold)
        masks, flows = out[0], out[1]

        # Save in the standard _seg.npy layout so stage 2 can read flows[4] and img.
        seg_out = os.path.join(folder, f"{name}_nuclear.npy")
        io.masks_flows_to_seg([img], [masks], [flows], [diameter or 0.0], [seg_out], channels=channels)
        print(f"  [{i}/{len(folders)}] {name}: {int(np.asarray(masks).max())} nuclei -> {seg_out}")

    print("\nStage 1 done. Feed seg_dir into stage 2.")


def raw_flows(seg: dict):
    """Return (dY, dX, cellprob) from the raw network output flows[4] = (3, H, W)."""
    raw = np.asarray(seg["flows"][4]).astype(np.float32)
    if raw.ndim != 3 or raw.shape[0] != 3:
        raise ValueError(
            "Expected flows[4] with shape (3, H, W) = [dY, dX, cellprob]. "
            f"Got {raw.shape}. Your Cellpose version may store flows differently."
        )
    return raw[0], raw[1], raw[2]


# --------------------------------------------------------------------------- #
# Core steps
# --------------------------------------------------------------------------- #
def divergence(dY: np.ndarray, dX: np.ndarray, sigma: float) -> np.ndarray:
    """Divergence of the flow field; strongly negative at sinks (cell centers)."""
    div = np.gradient(dX, axis=1) + np.gradient(dY, axis=0)
    return ndi.gaussian_filter(div, sigma)


def detect_channels(img: np.ndarray, div: np.ndarray, fg: np.ndarray, sink_h: float):
    """Return (nucleus, membrane). Nucleus is the channel bright at flow sinks."""
    sinks = morphology.h_maxima((-div) * fg, sink_h) > 0
    if sinks.sum() == 0:                       # fallback: use most-negative div pixels
        sinks = div < np.percentile(div, 1)
    m0, m1 = img[0][sinks].mean(), img[1][sinks].mean()
    if m1 >= m0:
        return img[1], img[0], 1               # nucleus = plane 1
    return img[0], img[1], 0                    # nucleus = plane 0


def get_markers(div: np.ndarray, fg: np.ndarray, sink_h: float) -> np.ndarray:
    """Labelled flow-sink markers (ALL sinks; no nucleus gating)."""
    seeds = morphology.h_maxima((-div) * fg, sink_h)
    return measure.label(seeds)


def build_elevation(membrane: np.ndarray, div: np.ndarray, p: Params) -> np.ndarray:
    """Watershed elevation = membrane wall (+skeleton) + weighted flow separatrix."""
    skel = skeletonize(membrane > p.membrane_bin)
    mem_term = ndi.gaussian_filter(membrane, p.mem_sigma) + skel.astype(np.float32)
    divn = (div - div.min()) / (np.ptp(div) + 1e-9)     # high along between-cell ridges
    return mem_term + p.flow_weight * divn


def veto_merge(labels: np.ndarray, membrane: np.ndarray, p: Params) -> np.ndarray:
    """Merge adjacent labels whose shared border has no membrane wall behind it."""
    pair_vals: dict = {}
    for ax in (0, 1):
        a = labels
        b = np.roll(labels, -1, axis=ax)
        mm = (a != b) & (a > 0) & (b > 0)
        val = 0.5 * (membrane + np.roll(membrane, -1, axis=ax))[mm]
        ai, bi = a[mm], b[mm]
        for i, j, v in zip(ai, bi, val):
            k = (min(i, j), max(i, j))
            pair_vals.setdefault(k, []).append(v)

    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for (i, j), vals in pair_vals.items():
        if len(vals) >= p.veto_min_border and float(np.mean(vals)) < p.veto_wall:
            union(i, j)

    out = labels.copy()
    for lab in np.unique(labels):
        if lab:
            out[labels == lab] = find(lab)
    out, _, _ = relabel_sequential(out)
    return out


# ADDED: delete masks that have an interior hole
def remove_holey(labels: np.ndarray, p: Params) -> np.ndarray:
    """Delete any mask whose area is < hole_ratio * its hole-filled area (i.e. it has a hole)."""
    out = labels.copy()
    for r in measure.regionprops(labels):
        filled = ndi.binary_fill_holes(r.image)          # r.image = mask cropped to its bbox
        if r.image.sum() < p.hole_ratio * filled.sum():  # filling added area => had a hole
            out[labels == r.label] = 0
    out, _, _ = relabel_sequential(out)
    return out


def run_pipeline(seg: dict, p: Params = Params()):
    """
    Full frozen pipeline.
    Returns (labels, ctx) where ctx holds intermediate arrays for inspection/plots.
    """
    img = np.asarray(seg["img"]).astype(np.float32)
    dY, dX, cellprob = raw_flows(seg)
    fg = cellprob > p.cellprob_thr

    div = divergence(dY, dX, p.div_sigma)
    nucleus, membrane, nuc_plane = detect_channels(img, div, fg, p.sink_h)

    markers = get_markers(div, fg, p.sink_h)
    elev = build_elevation(membrane, div, p)
    seg_ws = segmentation.watershed(elev, markers, mask=fg)
    seg_ws = morphology.remove_small_objects(seg_ws, p.min_size)
    labels = veto_merge(seg_ws, membrane, p) if p.use_veto else seg_ws   # ADDED: skip veto for nuclei
    labels = remove_holey(labels, p)      # ADDED: final step, delete masks with holes

    ctx = dict(nucleus=nucleus, membrane=membrane, nuc_plane=nuc_plane,
               div=div, cellprob=cellprob, fg=fg, markers=markers,
               n_markers=int(markers.max()), n_cells=int(labels.max()))
    return labels, ctx


# --------------------------------------------------------------------------- #
# Per-cell confidence table
# --------------------------------------------------------------------------- #
def cell_table(labels: np.ndarray, ctx: dict, p: Params):
    """Per-cell QC metrics. Returns list[dict]; low-confidence cells flagged."""
    nucleus, membrane, div, cellprob, fg = (
        ctx["nucleus"], ctx["membrane"], ctx["div"], ctx["cellprob"], ctx["fg"])
    inner = find_boundaries(labels, mode="inner") & fg
    rows = []
    for r in measure.regionprops(labels):
        m = labels == r.label
        b = m & inner
        boundary_membrane = float(membrane[b].mean()) if b.sum() else 0.0
        nuclear_frac = float((nucleus[m] > p.nuc_thr).mean())
        mean_cellprob = float(cellprob[m].mean())
        sink_strength = float((-div[m]).max())      # strong negative div => strong attractor
        rows.append(dict(
            label=int(r.label),
            area_px=int(r.area),
            boundary_membrane=round(boundary_membrane, 4),
            nuclear_frac=round(nuclear_frac, 4),
            mean_cellprob=round(mean_cellprob, 4),
            sink_strength=round(sink_strength, 4),
        ))
    # simple low-confidence flag: weak wall AND little nucleus
    for row in rows:
        row["low_confidence"] = int(
            (row["boundary_membrane"] < p.veto_wall) and (row["nuclear_frac"] < 0.05)
        )
    return rows


def save_table(rows, path):
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(path, index=False)
    except Exception:
        import csv
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


# --------------------------------------------------------------------------- #
# Optional overlay
# --------------------------------------------------------------------------- #
def make_overlay(seg: dict, labels: np.ndarray, ctx: dict, path: str,
                 crop=None, dim=0.55):
    """Save a comparison PNG: RG input + Cellpose (yellow) vs pipeline (cyan dashed)."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("  [overlay skipped: Pillow not available]")
        return
    nucleus, membrane = ctx["nucleus"], ctx["membrane"]
    H, W = nucleus.shape
    if crop is None:
        y0, x0, s = 0, 0, min(H, W)
    else:
        y0, x0, s = crop
    sub = np.s_[y0:y0 + s, x0:x0 + s]

    rgb = np.zeros((s, s, 3), np.float32)
    rgb[..., 0] = np.clip(nucleus[sub], 0, 1)
    rgb[..., 1] = np.clip(membrane[sub], 0, 1)
    rgb *= dim
    out = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    fb = dilation(find_boundaries(labels[sub], mode="inner"), disk(1))
    Y, X = np.mgrid[0:s, 0:s]
    fb = fb & (((X + Y) % 30) < 22)             # dashed
    if "masks" in seg and seg["masks"] is not None:
        cb = dilation(find_boundaries(np.asarray(seg["masks"])[sub], mode="inner"), disk(1))
        out[cb] = [255, 255, 0]                 # Cellpose = solid yellow
    out[fb] = [0, 220, 255]                     # pipeline = dashed cyan

    scale = min(1000, s)
    im = Image.fromarray(out).resize((scale, scale), Image.NEAREST)
    d = ImageDraw.Draw(im)
    if "masks" in seg and seg["masks"] is not None:
        d.text((10, 10), f"yellow = Cellpose ({int(np.asarray(seg['masks']).max())})",
               fill=(255, 255, 0))
    d.text((10, 28), f"cyan dashed = flow+membrane ({int(labels.max())})", fill=(0, 220, 255))
    im.save(path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def process_one(seg_path: str, out_dir: str, p: Params, overlay: bool):
    name = os.path.splitext(os.path.basename(seg_path))[0]
    seg = load_seg(seg_path)
    labels, ctx = run_pipeline(seg, p)
    rows = cell_table(labels, ctx, p)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{name}.labels.npy"), labels.astype(np.uint16))
    tiff.imwrite(os.path.join(out_dir, f"{name}.labels.tif"), labels.astype(np.uint16))  # ADDED: TIFF for your viewer
    save_table(rows, os.path.join(out_dir, f"{name}.cells.csv"))
    if overlay:
        make_overlay(seg, labels, ctx, os.path.join(out_dir, f"{name}.overlay.png"))

    n_low = sum(r["low_confidence"] for r in rows)
    base = int(np.asarray(seg["masks"]).max()) if "masks" in seg and seg["masks"] is not None else -1
    print(f"[{name}] sinks={ctx['n_markers']}  cells={ctx['n_cells']} "
          f"(Cellpose={base})  low_conf={n_low}  nucleus=plane{ctx['nuc_plane']}")
    return labels, rows


def main():
    ap = argparse.ArgumentParser(description="Flow + membrane hybrid segmentation.")
    ap.add_argument("input", help="a *_seg.npy file OR a directory of them")
    ap.add_argument("-o", "--out", default="fm_seg_out", help="output directory")
    ap.add_argument("--overlay", action="store_true", help="also write comparison PNGs")
    ap.add_argument("--sink-h", type=float, default=Params.sink_h)
    ap.add_argument("--flow-weight", type=float, default=Params.flow_weight)
    ap.add_argument("--veto-wall", type=float, default=Params.veto_wall)
    args = ap.parse_args()

    p = Params(sink_h=args.sink_h, flow_weight=args.flow_weight, veto_wall=args.veto_wall)
    print("params:", asdict(p))

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*_seg.npy")) or
                       glob.glob(os.path.join(args.input, "*.npy")))
        if not files:
            raise SystemExit(f"No .npy files found in {args.input}")
        for f in files:
            try:
                process_one(f, args.out, p, args.overlay)
            except Exception as e:
                print(f"[{os.path.basename(f)}] FAILED: {e}")
    else:
        process_one(args.input, args.out, p, args.overlay)


if __name__ == "__main__":
    main()