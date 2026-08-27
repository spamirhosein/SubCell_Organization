# Cellpose Segmentation

Runs a custom-trained Cellpose **2.x** model over a folder of 2-channel
(nucleus + membrane) TIFFs on the DKFZ cluster. Produces label masks and the
`_seg.npy` flow files used by [`flow_membrane_seg.py`](../flow_membrane_seg.py).

Two scripts, both in this directory:

| script | when to use it | writes |
|---|---|---|
| [`Run_cellpose.py`](Run_cellpose.py) | the **production run** — one parameter set, every image in the folder | `_mask.tif` **and** `_seg.npy` |
| [`Sweep_cellpose.py`](Sweep_cellpose.py) | **choosing parameters** — many parameter sets, a few sample images | `_mask.tif` only, plus `sweep_summary.tsv` |

Sweep first to pick the settings, then run. The sweep deliberately skips
`_seg.npy` (~165 MB each) since you throw most combinations away.

---

## Environment

Use the `cellpose2` env. **Cellpose 4 will not work** — the trained models are
Cellpose 2/3 checkpoints and CP4 rejects them outright.

```bash
micromamba activate cellpose2      # cellpose 2.3.2, torch 2.5.1+cu121, numpy 1.26
```

Do **not** `module load CUDA` — the pip torch wheel bundles its own CUDA runtime.

## Input

A **flat** folder of `*.tif` / `*.tiff`, each `(2, H, W)` or `(H, W, 2)` with
channel order `(nucleus, membrane)`. Subfolders are ignored — one job per folder.

## Run it

Replace the placeholder paths with your own:

```bash
MODEL=/path/to/models/my_cellpose_model      # Cellpose 2/3 checkpoint
IMGDIR=/path/to/images/whole_cell            # flat folder of 2-channel TIFFs
LOGDIR=/path/to/logs                         # must already exist
SCRIPTS=/path/to/scripts/segmentation/cellpose   # this directory

bsub -J whole_cell_seg -gpu num=1:j_exclusive=yes:gmem=10.7G -q gpu -n 4 \
 -oo $LOGDIR/cellpose.%J.out \
 -eo $LOGDIR/cellpose.%J.err \
"source ~/.bashrc; micromamba activate cellpose2 && \
 python $SCRIPTS/Run_cellpose.py -m $MODEL -i $IMGDIR"
```

`source ~/.bashrc` is required — it is what makes `micromamba` available in the
batch shell. The log directory must already exist or the job fails immediately.

## Output

Into `<img-dir>/<model-name>/`, so different models never overwrite each other
(override with `-o`):

| file | content |
|---|---|
| `<stem>_mask.tif` | `uint16` labels, `0` = background, `1..N` = cells |
| `<stem>_seg.npy` | dict with `masks`, `outlines`, `img`, **`flows`**, `est_diam` |

Load the npy with `np.load(path, allow_pickle=True).item()`.

⚠️ **`_seg.npy` is ~165 MB per 2048×2048 image** (~24 GB for 150 images) — it
stores the image and flow field. Budget disk accordingly.

## Parameters

All optional; append to the `python` line.

| flag | default | effect |
|---|---|---|
| `-d, --diameter` | the model's own `diam_labels` | strongest knob; sets `rescale = 30/diameter` |
| `--cellprob-threshold` | `0.5` | cell **extent**. Lower → bigger cells. Range −6…6 |
| `--flow-threshold` | `0.6` | shape filter. Higher → accepts more (incl. malformed) masks |
| `--min-size` | `15` | drops masks under N px. A ~110 px cell is ~7000 px in area, so 15 filters nothing — try `1000`–`2000` |
| `--cpu` | off | force CPU (very slow at 2048²) |

```bash
python $SCRIPTS/Run_cellpose.py -m $MODEL -i $IMGDIR \
  -d 110 --flow-threshold 0.6 --cellprob-threshold 0.5 --min-size 1000
```

## Tuning

Grid-search on a few images first — same flags, plural, space-separated:

```bash
python $SCRIPTS/Sweep_cellpose.py -m $MODEL -i $IMGDIR -o /path/to/sweep_out -n 4 \
  -d 100 110 125 --flow-thresholds 0.4 0.6 \
  --cellprob-thresholds -1.0 0.0 0.5 --min-sizes 1000
```

Writes one subfolder of masks per combination plus `sweep_summary.tsv`
(cell counts, areas, frame coverage). Compare masks visually — counts alone
can't tell undersegmentation from correctness. Tune `diameter` first, then
`cellprob`, then `flow`, and set `min_size` last from the area distribution.

Faster alternative: copy a couple of `_seg.npy` files to your laptop and open
them in a local Cellpose 2.x GUI — because the flows are stored, the GUI
recomputes masks at new `flow`/`cellprob` values instantly, with live sliders.
(The GUI extras are not installed in `cellpose2`.)

## Notes

- One job per input folder; each quality/compartment folder needs its own `-i`.
- Cellpose falls back to CPU with only a warning if CUDA is unavailable — check
  the `.err` file after the job starts rather than assuming the GPU was used.
- Models must be Cellpose 2/3 checkpoints, not Cellpose 4.
