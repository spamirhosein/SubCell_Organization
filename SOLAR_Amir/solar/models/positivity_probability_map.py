"""Positivity probability map generation for MIBI organelle markers."""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from solar.models.minotri_threshold import minotri_threshold

def positivity_probability_map(
    I,
    *,
    pixel_size_nm=90,
    organelle_diameter_nm=500,
    asinh_cofactor=1.0,
    despeckle_median_size=0,
    gaussian_sigmas_px=(1.5, 2.3, 3.0),
    z_windows_px=(31, 101),
    z_sigma_floor=0.0,
    normalize_z_to_bg=False,
    nbins=256,
    low_percentile=1.0,
    sigmoid_slope=0.7,
    tile_size=256,
    tile_overlap=64,
    min_component_area_px=30,
    return_debug=True
):
    """
    Generate pixel-wise positivity probability maps for organelle markers in MIBI images.

    Parameters:
    - I (ndarray): Input 2D marker image.
    - pixel_size_nm (float): Pixel size in nanometers.
    - organelle_diameter_nm (float): Typical organelle diameter in nanometers.
    - asinh_cofactor (float): Cofactor for asinh transformation.
    - despeckle_median_size (int): Median filter size for early despeckle (0 disables).
    - gaussian_sigmas_px (tuple): Gaussian smoothing sigmas in pixels.
    - z_windows_px (tuple): Window sizes for spatial z-scoring.
    - z_sigma_floor (float): Minimum local std for z-score normalization (0 disables).
    - normalize_z_to_bg (bool): Normalize Z to background and threshold scale (default False).
    - nbins (int): Number of bins for histogram computation.
    - low_percentile (float): Percentile for masking low-intensity pixels.
    - sigmoid_slope (float): Slope for sigmoid mapping.
    - tile_size (int | None): Size of tiles for gating. Use None to disable tiling.
    - tile_overlap (int): Overlap between tiles (ignored if tile_size is None).
    - min_component_area_px (int): Minimum area for connected components.
    - return_debug (bool): Whether to return debug information.

    Returns:
    - P (ndarray): Probability map.
    - debug (dict): Debugging information (if return_debug=True).
    """
    debug = {}

    # Step 1: Asinh transform
    I_asinh = np.arcsinh(I / asinh_cofactor).astype(np.float32)

    # Step 1b: Early despeckle (median filter)
    if despeckle_median_size is not None and despeckle_median_size > 0:
        I_asinh = median_filter(I_asinh, size=despeckle_median_size).astype(np.float32)

    # Step 2: Structure-supported intensity map S
    S = np.max([gaussian_filter(I_asinh, sigma) for sigma in gaussian_sigmas_px], axis=0)
    S = np.clip(S, 0, None)  # Ensure non-negative values

    # Step 3: Masking for histogram stability
    v = S[S > 0]
    if v.size < 10000:  # Safeguard for zero-signal images
        return np.zeros_like(S, dtype=np.float32), debug

    t_low = np.percentile(v, low_percentile)
    M = S >= t_low
    debug['t_low'] = t_low
    debug['mask_fraction'] = np.mean(M)

    # Step 4: Spatial Z-score
    Z = np.full_like(S, np.inf, dtype=np.float32)
    z_floor_fracs = []
    for w in z_windows_px:
        mu_w = uniform_filter(S, size=w, mode='reflect')
        sigma_w = np.sqrt(uniform_filter(S**2, size=w, mode='reflect') - mu_w**2 + 1e-8)
        if z_sigma_floor is not None and z_sigma_floor > 0:
            z_floor_fracs.append(float(np.mean(sigma_w < z_sigma_floor)))
            sigma_w = np.maximum(sigma_w, z_sigma_floor)
        Z_w = (S - mu_w) / (sigma_w + 1e-8)
        Z = np.minimum(Z, Z_w)
    if z_floor_fracs:
        debug['z_sigma_floor_fraction'] = z_floor_fracs

    # Step 5: Minotri threshold
    values = Z[M]
    t_minotri, otsu_score, t_triangle, t_otsu = minotri_threshold(values, nbins=nbins)
    debug.update({
        't_minotri': t_minotri,
        't_triangle': t_triangle,
        't_otsu': t_otsu,
        'otsu_score': otsu_score
    })

    # Step 6: Probability mapping (optional background normalization)
    if normalize_z_to_bg:
        z_bg = np.median(Z[M])
        den = max(t_minotri - z_bg, 1e-5)
        Z_norm = (Z - z_bg) / den
        t_use = 1.0
        debug['z_bg'] = float(z_bg)
        debug['z_den'] = float(den)
        debug['t_use'] = float(t_use)
        debug['normalize_z_to_bg'] = True
    else:
        Z_norm = Z
        t_use = t_minotri
        debug['normalize_z_to_bg'] = False

    P = 1 / (1 + np.exp(-(Z_norm - t_use) / sigmoid_slope))

    # Step 7: Tile-based gating (optional)
    if tile_size is not None and tile_size > 0:
        tiles = []
        tile_size = min(tile_size, S.shape[0], S.shape[1])
        step = max(tile_size - tile_overlap, 1)
        for i in range(0, S.shape[0] - tile_size + 1, step):
            for j in range(0, S.shape[1] - tile_size + 1, step):
                tile = Z[i:i + tile_size, j:j + tile_size]
                evidence = np.percentile(tile, 99.9) - np.median(tile)
                if evidence < 0.1:  # Example threshold
                    P[i:i + tile_size, j:j + tile_size] *= 0
                tiles.append(evidence)
        debug['tile_evidence'] = tiles

    # Step 8: Minimum size enforcement
    from skimage.measure import label, regionprops
    B = P > 0.9
    labeled = label(B)
    for region in regionprops(labeled):
        if region.area < min_component_area_px:
            P[labeled == region.label] = 0

    return (P.astype(np.float32), debug) if return_debug else P