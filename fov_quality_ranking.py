from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from skimage.filters import threshold_otsu
from skimage.transform import resize
import matplotlib.pyplot as plt

ROOT = Path(r"D:\image_data\Ha Anh\HLA-I_Channel3")
OUT_DIR = Path(r"D:\image_data\Ha Anh\HLA-I_Channel3\quality_ranking+abundance")
DEGENERATE_CUTOFF = 0.95  # flag if one class captures >95% of pixels
THUMB_SIZE = 128
MONTAGE_COLS = 12

# ── Abundance measurement ───────────────────────────────────────────────────────
USE_ABUNDANCE = True     # True  → penalise FOVs with little membrane coverage
                         # False → score by contrast (confusion) only
ABUNDANCE_WEIGHT = 0.3   # blend weight for abundance penalty [0.0 – 1.0]
                         # 0.0 = contrast only | 1.0 = abundance only


def load_and_normalize(tiff_path: Path) -> np.ndarray:
    img = tifffile.imread(tiff_path)
    if img.ndim > 2:
        img = np.squeeze(img)
    if img.ndim > 2:
        img = img[0]
    original_dtype = img.dtype
    img = img.astype(np.float32)
    if original_dtype == np.uint8:
        img /= 255.0
    elif original_dtype == np.uint16:
        img /= 65535.0
    elif not np.issubdtype(original_dtype, np.floating):
        vmax = img.max()
        if vmax > 1.0:
            img /= vmax
    return np.clip(img, 0.0, 1.0)


def compute_score(img: np.ndarray) -> dict:
    flat = img.ravel()
    try:
        thresh = threshold_otsu(flat)
    except Exception:
        return dict(p_background=np.nan, p_membrane=np.nan, otsu_threshold=np.nan,
                    clarity_score=np.nan, membrane_fraction=np.nan, degenerate=True)

    bg_mask = flat < thresh
    mem_mask = ~bg_mask
    bg_frac = bg_mask.sum() / len(flat)
    degenerate = bg_frac > DEGENERATE_CUTOFF or bg_frac < (1.0 - DEGENERATE_CUTOFF)

    p_bg = float(flat[bg_mask].mean()) if bg_mask.any() else 0.0
    p_mem = float(flat[mem_mask].mean()) if mem_mask.any() else 1.0
    score = abs(p_mem - p_bg)   # higher = better contrast
    mem_frac = float(mem_mask.sum()) / len(flat)

    return dict(p_background=p_bg, p_membrane=p_mem, otsu_threshold=float(thresh),
                clarity_score=score, membrane_fraction=mem_frac, degenerate=degenerate)


def make_thumbnail(img: np.ndarray, size: int = THUMB_SIZE) -> np.ndarray:
    return resize(img, (size, size), anti_aliasing=True, preserve_range=True).astype(np.float32)


def score_color(score: float) -> str:
    if score < 0.25:
        return "red"
    if score < 0.50:
        return "orange"
    return "green"


# ── Process all FOVs ───────────────────────────────────────────────────────────
fov_dirs = sorted(d for d in ROOT.iterdir() if d.is_dir())
print(f"Found {len(fov_dirs)} FOV folders.\n")

records = []
thumbnails = []  # list of (fov_name, thumb_array, confusion_score)

for fov_dir in fov_dirs:
    tiff_path = fov_dir / f"{fov_dir.name}.tiff"
    if not tiff_path.exists():
        candidates = list(fov_dir.glob("*.tiff")) + list(fov_dir.glob("*.tif"))
        if not candidates:
            print(f"  SKIP  {fov_dir.name}  — no TIFF found")
            continue
        tiff_path = candidates[0]

    try:
        img = load_and_normalize(tiff_path)
        result = compute_score(img)
        result["fov"] = fov_dir.name
        if USE_ABUNDANCE and not np.isnan(result["clarity_score"]):
            result["final_score"] = (
                (1.0 - ABUNDANCE_WEIGHT) * result["clarity_score"]
                + ABUNDANCE_WEIGHT * result["membrane_fraction"]
            )
        else:
            result["final_score"] = result["clarity_score"]
        records.append(result)
        thumbnails.append((fov_dir.name, make_thumbnail(img), result["final_score"]))
        flag = " [DEGENERATE]" if result["degenerate"] else ""
        abund_str = (f"  mem_frac={result['membrane_fraction']:.4f}" if USE_ABUNDANCE else "")
        print(f"  {fov_dir.name:<25}  final={result['final_score']:.4f}  "
              f"clarity={result['clarity_score']:.4f}{abund_str}{flag}")
    except Exception as exc:
        print(f"  ERROR  {fov_dir.name}  — {exc}")

# ── Build ranked DataFrame ─────────────────────────────────────────────────────
df = pd.DataFrame(records)
df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
df.insert(0, "rank", df.index + 1)  # rank 1 = best (highest final score)
df = df[["rank", "fov", "p_background", "p_membrane", "otsu_threshold",
         "clarity_score", "membrane_fraction", "final_score", "degenerate"]]
csv_path = OUT_DIR / "fov_quality_ranking.csv"
df.to_csv(csv_path, index=False, float_format="%.6f")
print(f"\nSaved ranking CSV  →  {csv_path}")

# ── Plot A: Bar chart ──────────────────────────────────────────────────────────
df_bar = df.sort_values("final_score", ascending=True)  # worst (low) left, best (high) right
colors = [score_color(s) for s in df_bar["final_score"]]

if USE_ABUNDANCE:
    fig, (ax, ax_abund) = plt.subplots(
        2, 1, figsize=(max(18, len(df_bar) * 0.22), 9),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
else:
    fig, ax = plt.subplots(figsize=(max(18, len(df_bar) * 0.22), 6))

ax.bar(range(len(df_bar)), df_bar["final_score"], color=colors, width=0.8, edgecolor="none")
ax.axhline(0.25, color="red", linestyle="--", linewidth=1.2, label="Low Confidence  < 0.25")
ax.axhline(0.50, color="orange", linestyle="--", linewidth=1.2, label="Moderate Confidence  < 0.50")
ax.set_xticks(range(len(df_bar)))
ax.set_xticklabels(df_bar["fov"], rotation=90, fontsize=6)
if not USE_ABUNDANCE:
    ax.set_xlabel("FOV  (sorted: worst quality → best quality)", fontsize=10)
score_label = (
    f"Final Score  =  {1-ABUNDANCE_WEIGHT:.1f}·clarity + {ABUNDANCE_WEIGHT:.1f}·mem_fraction"
    if USE_ABUNDANCE
    else "Clarity Score  =  |P(membrane) − P(background)|"
)
ax.set_ylabel(score_label, fontsize=9)
ax.set_title("FOV Quality Ranking — Membrane Channel Positivity Map Clarity", fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.legend(fontsize=9)

if USE_ABUNDANCE:
    ax_abund.bar(range(len(df_bar)), df_bar["membrane_fraction"],
                 color="steelblue", width=0.8, edgecolor="none", alpha=0.8)
    ax_abund.set_ylabel("Membrane\nFraction", fontsize=8)
    ax_abund.set_xlabel("FOV  (sorted: worst quality → best quality)", fontsize=10)
    ax_abund.set_ylim(0, df_bar["membrane_fraction"].max() * 1.15)
    ax_abund.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

plt.tight_layout()
bar_path = OUT_DIR / "fov_quality_scores_barplot.png"
fig.savefig(bar_path, dpi=150)
plt.close(fig)
print(f"Saved bar chart     →  {bar_path}")

# ── Plot B: Montage grid ───────────────────────────────────────────────────────
thumbnails_sorted = sorted(thumbnails, key=lambda x: x[2], reverse=False)  # worst (low score) first
n = len(thumbnails_sorted)
ncols = MONTAGE_COLS
nrows = (n + ncols - 1) // ncols

fig2, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.9))
axes_flat = np.array(axes).flatten()

for i, (fov_name, thumb, score) in enumerate(thumbnails_sorted):
    ax = axes_flat[i]
    ax.imshow(thumb, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    border = score_color(score)
    for spine in ax.spines.values():
        spine.set_edgecolor(border)
        spine.set_linewidth(3)
    short_name = fov_name.replace("_C01_R01", "")
    ax.set_title(f"{short_name}\n{score:.3f}", fontsize=5, pad=2,
                 color=border if score < 0.5 else "black")
    ax.set_xticks([])
    ax.set_yticks([])

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

score_mode_str = "final score (clarity + abundance)" if USE_ABUNDANCE else "clarity score (contrast only)"
fig2.suptitle(
    f"FOV Montage — sorted worst → best ({score_mode_str})\n"
    "Red border < 0.25 | Orange border 0.25–0.50 | Green border ≥ 0.50",
    fontsize=10,
)
plt.tight_layout()
montage_path = OUT_DIR / "fov_quality_montage.png"
fig2.savefig(montage_path, dpi=100)
plt.close(fig2)
print(f"Saved montage       →  {montage_path}")
