# Flow + Membrane Segmentation

A refinement step for Cellpose segmentation of 2-channel (nucleus + membrane) tissue
images such as MIBI. It takes Cellpose's **flow field** and rebuilds the instance
masks, letting each signal do only what it is reliable at.

Input is a Cellpose `*_seg.npy` — it already contains the image *and* the flow field,
so the source TIFFs are not needed. Output is an instance-label mask.

> **Use `flow_membrane_seg_v2.py`.** v1 (`flow_membrane_seg.py`) is kept for
> reproducibility. See [What changed in v2](#what-changed-in-v2).

---

## Results

![v1 vs v2](fig1_v1_vs_v2.png)

*Same region of `A_1a_C01_R01_p2`. Left: the Cellpose masks it starts from.
Middle: v1, which merged most cells away. Right: v2, which keeps them and follows
the membrane.*

---

## How it works

```
Cellpose flow (_seg.npy)
        │
        ├─ divergence ────────► cell-centre seeds (h-maxima)      [sink_h]
        ├─ membrane wall ─┐
        ├─ flow separatrix ┼──► watershed elevation ──► watershed [flow_weight]
        │                 │
        ├─ membrane veto ─────► merge false splits                [veto_wall]
        ├─ hole removal ──────► delete masks with interior holes  [hole_ratio]
        └─ shape filter ──────► delete non-cell shapes  (v2)      [min_circularity,
                                                                   min_thickness,
                                                                   min_area]
```

- **Flow divergence → cell centres.** Robust even where the membrane is dim.
- **Membrane wall → boundaries where a wall exists.** Snaps edges onto real signal.
- **Flow separatrix → boundaries where the membrane is absent.** Avoids arbitrary guesses.
- **Membrane veto → merges neighbours with no wall between them.** *Off by default in v2.*
- **Hole removal → deletes perforated masks.**
- **Shape filter (v2) → deletes masks that are not cell-shaped.**

## Usage

```bash
micromamba activate mohit      # needs skimage, scipy, tifffile — NOT in cellpose2

python flow_membrane_seg_v2.py <dir of *_seg.npy> -o <out_dir> --overlay
python flow_membrane_seg_v2.py one_tile_seg.npy   -o <out_dir>
```

CPU only, no GPU needed. From Python:

```python
import flow_membrane_seg_v2 as fm

seg = fm.load_seg("tile_seg.npy")
labels, ctx = fm.run_pipeline(seg, fm.Params())
```

### Output

Per input, into `-o`:

| file | content |
|---|---|
| `<name>.whole_cell.tif` | `uint16` label image |
| `<name>.whole_cell.npy` | same, as an array |
| `<name>.cells.csv` | per-cell QC: area, boundary membrane, nuclear fraction, cellprob, sink strength, `low_confidence` |
| `<name>.overlay.png` | Cellpose vs pipeline comparison (with `--overlay`) |

## What changed in v2

Two tuned defaults and one new step.

| parameter | v1 | v2 | why |
|---|---|---|---|
| `veto_wall` | 0.45 | **0** | 0.45 merged away ~half of all cells |
| `flow_weight` | 1 | **0.8** | slightly more weight on real membrane signal |
| `min_circularity` | — | **0.22** | delete branched / tentacle masks |
| `min_thickness` | — | **0.45** | delete thin slivers |
| `min_area` | — | **800** | delete fragments |

### Why `veto_wall = 0`

![veto_wall](fig4_veto_wall.png)

The veto merges two neighbours when the membrane along their shared border is
weaker than `veto_wall`. On this data the membrane brightness on real boundaries
sits around 0.6, so a threshold of 0.45 cuts into genuine walls: cell count
collapses from ~283 to ~118. Setting it to 0 disables merging.

Note that with `veto_wall = 0`, **`veto_min_border` has no effect** — the merge
condition can never fire.

### The shape filter

![shape filter](fig2_shape_filter.png)

*Left: masks before the filter. Right: the ones it removes, in red — the sprawling
shapes that snake between real cells.*

Three tests, OR'd — a mask is deleted if **any** fires:

- **`circularity`** = `4πA/P²`. Falls with total perimeter, so it catches branched
  shapes. Does most of the work.
- **`thickness`** = `2·max(distance transform) / equivalent diameter`. How fat a mask
  is relative to its size. Built on the distance transform, so unlike circularity it
  is *not* thrown off by a ragged outline. Catches slivers circularity misses.
- **`area`** — fragments.

The two shape tests fail on different shapes, which is why both are used: a branched
blob can still contain one fat lobe (good thickness, bad circularity), and a smooth
sliver can have a tidy perimeter (good circularity, bad thickness).

Set any threshold to `0` to disable that test.

### Reproducing v1

```python
fm.Params(veto_wall=0.45, flow_weight=1,
          min_circularity=0, min_thickness=0, min_area=0)
```
This is pixel-identical to v1 output.

## Parameters (`Params`)

| parameter | default | effect |
|---|---|---|
| `veto_wall` | 0.0 | merge neighbours whose shared wall is weaker than this. **Higher = more merging = fewer cells.** 0 = off |
| `veto_min_border` | 5 | px of shared border needed before the veto applies. No effect when `veto_wall = 0` |
| `sink_h` | 0.12 | seed sensitivity. **Lower = more seeds = more, smaller cells** |
| `flow_weight` | 0.8 | flow vs membrane in the boundary. Higher = trust flow where membrane is weak |
| `min_circularity` | 0.22 | below this = not a cell |
| `min_thickness` | 0.45 | below this = sliver |
| `min_area` | 800 | below this = fragment |
| `hole_ratio` | 0.99 | delete a mask if it has an interior hole (0.99 = a 1% hole is enough) |
| `min_size` | 80 | early `remove_small_objects`, before the veto |
| `cellprob_thr` | 0.0 | foreground extent. Moves every boundary at once |
| `div_sigma` | 2.0 | smoothing of the divergence field before seeding |
| `mem_sigma` | 1.0 | smoothing of the membrane term |
| `membrane_bin` | 0.5 | threshold for skeletonising the membrane |
| `nuc_thr` | 0.45 | nuclear-content threshold, QC table only |

CLI flags exist for `--sink-h`, `--flow-weight`, `--veto-wall`, `--min-circularity`,
`--min-thickness`, `--min-area`. The rest are Python-only.

## Notebooks

| notebook | purpose |
|---|---|
| `flow_membrane_finetune.ipynb` | **tune parameters.** Pick a FOV, compare settings side by side, preview the shape filter |
| `evaluate_refinement.ipynb` | **evaluate a finished run** against the Cellpose masks (matching, size change, per-cell QC) |
| `flow_membrane_seg_walkthrough.ipynb` | step-by-step walkthrough of the algorithm, stage by stage |

## Requirements

`numpy`, `scipy`, `scikit-image`, `tifffile` — all present in the `mohit` env.
`cellpose` is needed only to *produce* the input `_seg.npy` (see
`../cellpose/README.md`), not to run this.

## Notes

- Thresholds were tuned on whole-cell FOVs from `positivity_map_v3`. Check them on a
  few FOVs before trusting a new dataset — membrane brightness and cell density shift
  the right values.
- The shape thresholds are tuned against the *current* mask raggedness. If you raise
  `mem_sigma` to smooth outlines, circularity rises across the board and
  `min_circularity` needs re-checking.
- Designed for whole-cell segmentation. The nuclear masks in
  `High/nuclear/TN2/` have not been run through this.
