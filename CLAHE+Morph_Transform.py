import os
import numpy as np
import cv2
import tifffile as tiff

# Input and Output directories
MEM_DIR = r"/omics/odcf/analysis/OE0622_projects/mibi_shared/Amir/SubCOrg_Opt/New_Data/Intensity_Membrane_96"
NUC_DIR = r"/omics/odcf/analysis/OE0622_projects/mibi_shared/Amir/SubCOrg_Opt/New_Data/Intensity_Nucleus_96"
OUTPUT_DIR = r"/omics/odcf/analysis/OE0622_projects/mibi_shared/Amir/SubCOrg_Opt/New_Data/Mem_Nuc_Clahe_96"

# Per-channel morphological operations - choose which operations apply to each channel
# Available operations: 'opening', 'closing', 'gradient', 'tophat'
# Set to list of operations or None for no operation
NUCLEUS_MORPH_OPERATIONS = None  # E.g., ['opening'], ['closing'], ['opening', 'closing'], or None
MEMBRANE_MORPH_OPERATIONS = ['opening']  # E.g., ['opening'], ['gradient'], ['opening', 'gradient'], or None

# Morphological kernel sizes (width, height) - adjust these to change operation intensity
MORPH_OPENING_KERNEL = (3, 3)    # Smaller = gentler denoising, Larger = more aggressive
MORPH_CLOSING_KERNEL = (5, 5)    # Larger = fill bigger gaps
MORPH_GRADIENT_KERNEL = (3, 3)   # Kernel size for edge extraction
MORPH_TOPHAT_KERNEL = (15, 15)   # Larger = removes larger background features

# Channel stacking order
CHANNEL_ORDER = ['nucleus', 'membrane']  # Order: ['nucleus', 'membrane'] or ['membrane', 'nucleus']
# Note: This determines the order in the output file (Channel 0, Channel 1)

# CLAHE (Contrast Limited Adaptive Histogram Equalization) settings
APPLY_CLAHE = True                # Set to True to apply CLAHE, False to skip it
CLAHE_CLIP_LIMIT = 3.0            # Higher = more contrast
CLAHE_TILE_SIZE = (16, 16)       # Larger = smoother enhancement
CLAHE_NORMALIZE_PERCENTILE = 99  # Percentile for intensity normalization (99.5 = minimal clipping, None = no normalization)
CLAHE_PRE_EQUALIZE = False         # Pre-process with histogram equalization before CLAHE

# Gamma Correction settings (applied after CLAHE to normalize brightness)
APPLY_GAMMA = True                # Set to True to apply gamma correction after CLAHE
GAMMA_VALUE = 1.5                 # Gamma value > 1 darkens mid-tones while preserving bright membranes (1.5-2.0 recommended)

# Median filter settings 
APPLY_MEDIAN_FILTER = True       # Set to True to apply median filter, False to skip it
MEDIAN_FILTER_SIZE = 3           # Pixel kernel size (3 = 3×3 kernel). Must be odd number

# File filtering - process specific files or all files
PROCESS_SPECIFIC_FILES = False   # Set to True to process only specific files, False to process all
SPECIFIC_FILES_TO_PROCESS = [    # Add file base names here (without extension)
    "B_5l_C01_R01"
]

def get_image_files(directory):
    """Collect all image files (.tif, .tiff) from directory.
    Returns dict: {base_name: full_path}
    """
    files_dict = {}
    if not os.path.isdir(directory):
        return files_dict
    
    for filename in os.listdir(directory):
        if filename.lower().endswith((".tif", ".tiff")):
            base_name = os.path.splitext(filename)[0]
            full_path = os.path.join(directory, filename)
            files_dict[base_name] = full_path
    return files_dict

def morph_opening(img, kernel_size=(3, 3)):
    """Remove small speckles (salt noise) without blurring edges.
    Best for noise removal on all channel types.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def morph_closing(img, kernel_size=(5, 5)):
    """Fill small dark gaps inside bright regions.
    Best for nuclear channels (HH3, Lamin A/C) to ensure solid blob.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def morph_gradient(img, kernel_size=(3, 3)):
    """Extract edge outlines (dilation - erosion).
    Best for membrane channels to sharpen cell boundaries.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def morph_tophat(img, kernel_size=(15, 15)):
    """Contrast booster: original - opening.
    Boosts dim membrane signal. Can replace or complement z-score normalization.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

def apply_morphological_operations(img, operations_list):
    """Apply a list of morphological operations to an image.
    
    Args:
        img: Input image
        operations_list: List of operation names (e.g., ['opening', 'closing'])
                        or None for no operations
    
    Returns:
        Processed image after applying all operations in sequence
    """
    if operations_list is None:
        return img
    
    result = img.copy()
    for op in operations_list:
        if op == 'opening':
            result = morph_opening(result, kernel_size=MORPH_OPENING_KERNEL)
        elif op == 'closing':
            result = morph_closing(result, kernel_size=MORPH_CLOSING_KERNEL)
        elif op == 'gradient':
            result = morph_gradient(result, kernel_size=MORPH_GRADIENT_KERNEL)
        elif op == 'tophat':
            result = morph_tophat(result, kernel_size=MORPH_TOPHAT_KERNEL)
        else:
            print(f"WARNING: Unknown operation '{op}', skipping")
    return result

def median_filter(img, kernel_size=MEDIAN_FILTER_SIZE):
    """Apply median filter to denoise image.
    
    Args:
        img: Input image
        kernel_size: Median filter kernel size (must be odd: 3, 5, 7, etc.)
    
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {kernel_size}")
    return cv2.medianBlur(img, kernel_size)

def apply_gamma(img_u8, gamma=GAMMA_VALUE):
    """Apply gamma correction to uint8 image using lookup table.
    Gamma > 1 darkens mid-tones while preserving bright structures.
    
    Args:
        img_u8: uint8 input image (0-255)
        gamma: Gamma value (> 1 darkens, < 1 brightens)
    
    Returns:
        Gamma-corrected uint8 image
    """
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_u8, lut)

def clahe_uint16(img, tilesize=CLAHE_TILE_SIZE, cliplimit=CLAHE_CLIP_LIMIT, normalize_percentile=99.5):
    """Apply CLAHE to image in 16-bit, converting to 8-bit for output.
    
    Args:
        img: Input image (can be any dtype)
        tilesize: Tile size for CLAHE (larger = smoother). Default: CLAHE_TILE_SIZE
        cliplimit: Clip limit for CLAHE (higher = more contrast). Default: CLAHE_CLIP_LIMIT
        normalize_percentile: Percentile for intensity normalization (99.5 = minimal clipping, None = no normalization)
    """
    img_f = img.astype(np.float32)
    p = np.percentile(img_f, normalize_percentile)
    if p > 0:
        img_f = np.clip(img_f / p, 0, 1)
    u16 = np.round(img_f * 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tilesize)
    return clahe.apply(u16)

# Create the output directory if it doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all membrane and nucleus channel files
mem_files = get_image_files(MEM_DIR)
nuc_files = get_image_files(NUC_DIR)

if not mem_files:
    raise RuntimeError(f"No image files found in {MEM_DIR}")
if not nuc_files:
    raise RuntimeError(f"No image files found in {NUC_DIR}")

# Find matching base names
matched_pairs = sorted(set(mem_files.keys()) & set(nuc_files.keys()))

if not matched_pairs:
    raise RuntimeError(f"No matching files found between {MEM_DIR} and {NUC_DIR}")

# Filter to specific files if enabled
if PROCESS_SPECIFIC_FILES:
    if not SPECIFIC_FILES_TO_PROCESS:
        raise RuntimeError("PROCESS_SPECIFIC_FILES is True but SPECIFIC_FILES_TO_PROCESS is empty")
    matched_pairs = [f for f in matched_pairs if f in SPECIFIC_FILES_TO_PROCESS]
    if not matched_pairs:
        raise RuntimeError(f"No matching files found for specified files: {SPECIFIC_FILES_TO_PROCESS}")
    print(f"Processing {len(matched_pairs)} SPECIFIC files:")
    for f in matched_pairs:
        print(f"  - {f}")
else:
    print(f"Processing ALL {len(matched_pairs)} available files:")
    for f in matched_pairs:
        print(f"  - {f}")

print(f"\nConfiguration:")
print(f"  Nucleus operations: {NUCLEUS_MORPH_OPERATIONS}")
print(f"  Membrane operations: {MEMBRANE_MORPH_OPERATIONS}")
print(f"  Channel order: {CHANNEL_ORDER}")
print(f"  Median filter enabled: {APPLY_MEDIAN_FILTER}")
if APPLY_MEDIAN_FILTER:
    print(f"    - Kernel size: {MEDIAN_FILTER_SIZE}×{MEDIAN_FILTER_SIZE}")
print(f"  CLAHE enabled: {APPLY_CLAHE}")
if APPLY_CLAHE:
    print(f"    - Tile size: {CLAHE_TILE_SIZE}")
    print(f"    - Clip limit: {CLAHE_CLIP_LIMIT}")
    print(f"    - Normalize percentile: {CLAHE_NORMALIZE_PERCENTILE}")
    print(f"    - Pre-equalize: {CLAHE_PRE_EQUALIZE}")
print()

done, skipped = 0, 0

for base_name in matched_pairs:
    mem_path = mem_files[base_name]
    nuc_path = nuc_files[base_name]
    
    try:
        # Read raw images
        nuc_raw = tiff.imread(nuc_path)
        mem_raw = tiff.imread(mem_path)
        
        # Apply morphological operations on raw images (before CLAHE)
        # Each channel has its own list of operations
        nuc_raw = apply_morphological_operations(nuc_raw, NUCLEUS_MORPH_OPERATIONS)
        mem_raw = apply_morphological_operations(mem_raw, MEMBRANE_MORPH_OPERATIONS)
        
        # Apply median filter (if enabled) - after morphology, before CLAHE
        if APPLY_MEDIAN_FILTER:
            nuc_raw = median_filter(nuc_raw, kernel_size=MEDIAN_FILTER_SIZE)
            mem_raw = median_filter(mem_raw, kernel_size=MEDIAN_FILTER_SIZE)
        
        # Apply CLAHE normalization (if enabled)
        if APPLY_CLAHE:
            nuc = clahe_uint16(nuc_raw)
            mem = clahe_uint16(mem_raw)
            # Convert from uint16 to uint8 for gamma correction
            nuc = (nuc.astype(np.float32) / 65535 * 255).astype(np.uint8)
            mem = (mem.astype(np.float32) / 65535 * 255).astype(np.uint8)
            # Apply gamma correction (if enabled)
            if APPLY_GAMMA:
                nuc = apply_gamma(nuc, gamma=GAMMA_VALUE)
                mem = apply_gamma(mem, gamma=GAMMA_VALUE)
        else:
            # Skip CLAHE, just convert to uint8 with min-max scaling
            nuc = nuc_raw.astype(np.uint8) if nuc_raw.dtype == np.uint8 else np.round(((nuc_raw.astype(np.float32) - np.min(nuc_raw)) / (np.max(nuc_raw) - np.min(nuc_raw) + 1e-8)) * 255).astype(np.uint8)
            mem = mem_raw.astype(np.uint8) if mem_raw.dtype == np.uint8 else np.round(((mem_raw.astype(np.float32) - np.min(mem_raw)) / (np.max(mem_raw) - np.min(mem_raw) + 1e-8)) * 255).astype(np.uint8)

        # Stack channels in the specified order
        if CHANNEL_ORDER == ['nucleus', 'membrane']:
            out = np.stack([nuc, mem], axis=0).astype(np.uint8)  # (2, H, W) - Channel 0 = Nucleus, Channel 1 = Membrane
        elif CHANNEL_ORDER == ['membrane', 'nucleus']:
            out = np.stack([mem, nuc], axis=0).astype(np.uint8)  # (2, H, W) - Channel 0 = Membrane, Channel 1 = Nucleus
        else:
            raise ValueError(f"Invalid CHANNEL_ORDER: {CHANNEL_ORDER}. Must be ['nucleus', 'membrane'] or ['membrane', 'nucleus']")
        
        # Save to the output directory
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}.tiff")
        
        # Save with ImageJ format so Cellpose reads the channels correctly
        tiff.imwrite(out_path, out, imagej=True)

        print(f"DONE: {base_name}")
        done += 1
        
    except Exception as e:
        print(f"SKIP ({str(e)}): {base_name}")
        skipped += 1
        continue

print(f"\nFinished. DONE={done}, SKIPPED={skipped}")
