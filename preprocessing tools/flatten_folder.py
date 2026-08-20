import os
import glob
import shutil

INPUT_ROOT = r"D:\image_data\New_Data\Ha Anh\stack(HaAnh)\sorted_by_quality\High"
OUTPUT_DIR = os.path.join(INPUT_ROOT, "all_files")

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = []
for folder in sorted(glob.glob(os.path.join(INPUT_ROOT, "*"))):
    if not os.path.isdir(folder) or folder == OUTPUT_DIR:
        continue
    folder_name = os.path.basename(folder)
    for f in glob.glob(os.path.join(folder, f"{folder_name}.*")):
        files.append(f)

print(f"Found {len(files)} files.")

for src in files:
    dst = os.path.join(OUTPUT_DIR, os.path.basename(src))
    shutil.copy2(src, dst)
    print(f"  {os.path.relpath(src, INPUT_ROOT)}  ->  {os.path.basename(dst)}")

print("\nDone.")