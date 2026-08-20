from pathlib import Path
import numpy as np, tifffile as tiff, cellpose
from cellpose import models, io

print("cellpose version:", cellpose.version)

img_path = Path(r"C:\Users\Amirhossein\Desktop\omnipose_test\cellposetests\A_9e_C01_R01.tiff")
img = tiff.imread(str(img_path)).astype(np.float32)

if img.shape[0] == 2:
    img_for_model = img[[1, 0], :, :]; channel_axis = 0
else:
    img_for_model = img[:, :, [1, 0]]; channel_axis = -1

model = models.CellposeModel(pretrained_model=r"C:\Users\Amirhossein\.cellpose\models\TN2_hiRes_Amir_posMap2")
masks, flows, styles = model.eval(
    img_for_model, channel_axis=channel_axis, diameter=110,
    flow_threshold=0.6, cellprob_threshold=0.5,
    resample=True, interp=True, min_size=15,
)

print("masks:", masks.shape, masks.dtype, "unique:", len(np.unique(masks)))
print("flows lens:", [getattr(f, "shape", type(f)) for f in flows])

# v2 call — try LIST form (most compatible across 2.x minor versions)
io.masks_flows_to_seg(
    [img_for_model],        # images (list)
    [masks],                # masks  (list)
    [flows],                # flows  (list of lists)
    [110.0],                # diams  (list)
    [str(img_path)],        # file_names (list)
    channels=[2, 1],
)

expected = str(img_path.with_suffix("")) + "_seg.npy"
print("exists:", Path(expected).exists(), "->", expected)