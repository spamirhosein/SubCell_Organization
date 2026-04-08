"""Minotri thresholding wrapper for Nellie Otsu/Triangle thresholds."""

from nellie.utils.gpu_functions import otsu_threshold, triangle_threshold, _get_xp

def minotri_threshold(matrix, nbins=256, xp=None):
    """
    Compute the Minotri threshold, which is the minimum of the Otsu and Triangle thresholds.

    Parameters:
    - matrix (ndarray): Input 2D array (image) for thresholding.
    - nbins (int): Number of bins for histogram computation.
    - xp (module): Backend module (numpy or cupy). If None, inferred from `matrix`.

    Returns:
    - t_minotri (float): The Minotri threshold value.
    - otsu_score (float): The score associated with the Otsu threshold.
    - t_triangle (float): The Triangle threshold value.
    - t_otsu (float): The Otsu threshold value.
    """
    xp = _get_xp(matrix, xp)

    # Compute Otsu threshold and score
    t_otsu, otsu_score = otsu_threshold(matrix, nbins=nbins, xp=xp)

    # Compute Triangle threshold
    t_triangle = triangle_threshold(matrix, nbins=nbins, xp=xp)

    # Compute Minotri threshold
    t_minotri = min(t_otsu, t_triangle)

    return t_minotri, otsu_score, t_triangle, t_otsu