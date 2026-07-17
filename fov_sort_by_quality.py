import shutil
from pathlib import Path

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT     = Path(r"D:\image_data\New_Data\Ha Anh\HH3_Channel3")    # FOV source folder
TABLE_PATH = Path(r"D:\image_data\New_Data\Ha Anh\HLA-I_Channel3\quality_ranking+abundance\fov_quality_ranking.csv")    # CSV with FOV quality scores (must have 'fov' and 'final_score' columns)
OUT_DIR  = Path(r"D:\image_data\New_Data\Ha Anh\HH3_Channel3\sorted_by_quality")   # Output folder for sorted FOVs

SCORE_COL = "final_score"   # column from CSV used for sorting

# Thresholds — must match the ranking script's score_color logic
HIGH_THRESH = 0.50   # final_score >= HIGH_THRESH  → High
LOW_THRESH  = 0.25   # final_score <  LOW_THRESH   → Low
                     # everything in between        → Moderate


# ── Setup output folders ───────────────────────────────────────────────────────
for folder in ("High", "Moderate", "Low"):
    (OUT_DIR / folder).mkdir(parents=True, exist_ok=True)


# ── Load ranking CSV ───────────────────────────────────────────────────────────
df = pd.read_csv(TABLE_PATH)

missing_col = SCORE_COL not in df.columns
if missing_col:
    raise ValueError(
        f"Column '{SCORE_COL}' not found in CSV. "
        f"Available columns: {list(df.columns)}"
    )

counts = {"High": 0, "Moderate": 0, "Low": 0, "skipped": 0}

print(f"Sorting {len(df)} FOVs into {OUT_DIR}\n")

for _, row in df.iterrows():
    fov_name = row["fov"]
    score    = row[SCORE_COL]

    # Locate the source FOV folder
    fov_dir = INPUT / fov_name
    if not fov_dir.exists():
        print(f"  SKIP  {fov_name}  — folder not found in {INPUT}")
        counts["skipped"] += 1
        continue

    # Determine quality category
    if score >= HIGH_THRESH:
        category = "High"
    elif score >= LOW_THRESH:
        category = "Moderate"
    else:
        category = "Low"

    # Copy entire FOV folder
    dest = OUT_DIR / category / fov_name
    shutil.copytree(fov_dir, dest)
    counts[category] += 1
    print(f"  [{category:<8}]  {fov_name:<25}  {SCORE_COL}={score:.4f}")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"""
  High     ({SCORE_COL} >= {HIGH_THRESH}):  {counts['High']} FOVs  →  {OUT_DIR / 'High'}
  Moderate ({LOW_THRESH} <= score < {HIGH_THRESH}):  {counts['Moderate']} FOVs  →  {OUT_DIR / 'Moderate'}
  Low      ({SCORE_COL} <  {LOW_THRESH}):  {counts['Low']} FOVs  →  {OUT_DIR / 'Low'}
  Skipped:  {counts['skipped']} FOVs
""")
