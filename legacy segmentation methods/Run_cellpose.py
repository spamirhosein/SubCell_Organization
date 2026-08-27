from pathlib import Path
import numpy as np
import tifffile as tiff
from cellpose import models, io

model_path = r"C:\Users\Amirhossein\.cellpose\models\TN2_hiRes_Amir_posMap2"
img_dir = Path(r"C:\Users\Amirhossein\Desktop\omnipose_test\cellposetests")

tiff_files = sorted(img_dir.glob("*.tiff")) + sorted(img_dir.glob("*.tif"))
if not tiff_files:
    raise FileNotFoundError(f"No TIFF files found in {img_dir}")

print("Loading model...")
model = models.CellposeModel(pretrained_model=model_path)

for img_path in tiff_files:
    print(f"\nProcessing: {img_path.name}")
    img = tiff.imread(str(img_path)).astype(np.float32)

    if img.ndim != 3:
        print(f"  Skip: expected 3D, got {img.shape}")
        continue
    if img.shape[0] == 2:
        img_for_model = img[[1, 0], :, :]; channel_axis = 0
    elif img.shape[-1] == 2:
        img_for_model = img[:, :, [1, 0]]; channel_axis = -1
    else:
        print(f"  Skip: expected 2 channels, got {img.shape}")
        continue

    masks, flows, styles = model.eval(
        img_for_model,
        channel_axis=channel_axis,
        diameter=110,
        flow_threshold=0.6,
        cellprob_threshold=0.5,
        resample=True,
        interp=True,
        min_size=15,
    )

    # --- Cellpose 2.x: pass LISTS ---
    io.masks_flows_to_seg(
        [img_for_model],
        [masks],
        [flows],
        [110.0],
        [str(img_path)],
        channels=[2, 1],          # membrane, nucleus (1-indexed for GUI)
    )

    out_prefix = img_path.with_suffix("")
    seg_path = str(out_prefix) + "_seg.npy"
    if not Path(seg_path).exists():
        print(f"  WARNING: seg file not written for {img_path.name}")
        continue

    tiff.imwrite(str(out_prefix) + "_mask.tif", masks.astype(np.uint16))
    print(f"  Saved: {img_path.stem}_seg.npy and {out_prefix.name}_mask.tif")

print("\nAll segmentations complete.")