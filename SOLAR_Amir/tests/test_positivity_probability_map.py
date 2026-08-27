import numpy as np
from solar.models.positivity_probability_map import positivity_probability_map

def test_no_signal():
    """Test case: No signal (pure background + noise)."""
    I = np.random.normal(0, 0.01, (2046, 2046)).astype(np.float32)
    P, debug = positivity_probability_map(I, return_debug=True)

    assert np.allclose(P, 0, atol=1e-3), "Probability map should be near-zero for no signal."

def test_clustered_blobs():
    """Test case: Clustered blobs of known size."""
    I = np.zeros((2046, 2046), dtype=np.float32)
    for x, y in [(512, 512), (1024, 1024), (1536, 1536)]:
        I[x-5:x+5, y-5:y+5] = 1.0

    P, debug = positivity_probability_map(I, return_debug=True)

    assert np.all(P[512-5:512+5, 512-5:512+5] > 0.9), "High probability expected in blob regions."
    assert np.all(P[0:256, 0:256] < 0.1), "Low probability expected outside blobs."

def test_salt_and_pepper():
    """Test case: Salt-and-pepper noise suppression."""
    I = np.random.normal(0, 0.01, (2046, 2046)).astype(np.float32)
    I[100, 100] = 10.0  # Hot pixel

    P, debug = positivity_probability_map(I, return_debug=True)

    assert P[100, 100] < 0.1, "Hot pixel should be suppressed."

def test_broad_haze():
    """Test case: Broad haze suppression."""
    I = np.linspace(0, 1, 2046).reshape(2046, 1).astype(np.float32)
    I = np.tile(I, (1, 2046))

    P, debug = positivity_probability_map(I, return_debug=True)

    assert np.all(P < 0.5), "Broad haze should result in low probabilities."


def test_despeckle_median_filter_reduces_spikes():
    """Test case: Median despeckle reduces spike artifacts while preserving blobs."""
    rng = np.random.default_rng(0)
    I = rng.normal(0, 0.01, (256, 256)).astype(np.float32)

    # Add a true blob
    I[120:130, 120:130] = 1.0

    # Add random single-pixel and 2x2 spikes
    spike_coords = []
    for _ in range(50):
        x, y = rng.integers(0, 256, size=2)
        if 120 <= x < 130 and 120 <= y < 130:
            continue
        I[x, y] = 5.0
        spike_coords.append((x, y))

    for _ in range(20):
        x, y = rng.integers(0, 255, size=2)
        if 120 <= x < 130 and 120 <= y < 130:
            continue
        I[x:x + 2, y:y + 2] = 3.0
        spike_coords.extend([(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)])

    P_no, _ = positivity_probability_map(I, return_debug=True, despeckle_median_size=0)
    P_med, _ = positivity_probability_map(I, return_debug=True, despeckle_median_size=3)

    spike_vals_no = np.array([P_no[x, y] for x, y in spike_coords], dtype=np.float32)
    spike_vals_med = np.array([P_med[x, y] for x, y in spike_coords], dtype=np.float32)

    assert spike_vals_med.mean() < spike_vals_no.mean(), "Despeckle should reduce spike probabilities."
    assert np.percentile(spike_vals_med, 90) < 0.1, "Most spikes should be suppressed by despeckle."

    blob_mean = P_med[120:130, 120:130].mean()
    assert blob_mean > 0.8, "True blob should remain high probability after despeckle."


def test_z_sigma_floor_reduces_spike_zscore():
    """Test case: z-score floor reduces spike amplification in flat regions."""
    rng = np.random.default_rng(1)
    I = rng.normal(0, 0.005, (256, 256)).astype(np.float32)

    # Add a true blob
    I[120:130, 120:130] = 1.0

    # Add isolated spikes
    spike_coords = []
    for _ in range(40):
        x, y = rng.integers(0, 256, size=2)
        if 120 <= x < 130 and 120 <= y < 130:
            continue
        I[x, y] = 4.0
        spike_coords.append((x, y))

    P_no, _ = positivity_probability_map(I, return_debug=True, z_sigma_floor=0.0)
    P_floor, _ = positivity_probability_map(I, return_debug=True, z_sigma_floor=0.05)

    spike_vals_no = np.array([P_no[x, y] for x, y in spike_coords], dtype=np.float32)
    spike_vals_floor = np.array([P_floor[x, y] for x, y in spike_coords], dtype=np.float32)

    assert spike_vals_floor.mean() < spike_vals_no.mean(), "z_sigma_floor should reduce spike probabilities."
    assert np.percentile(spike_vals_floor, 90) < 0.1, "Most spikes should be suppressed by z_sigma_floor."

    blob_mean = P_floor[120:130, 120:130].mean()
    assert blob_mean > 0.8, "True blob should remain high probability with z_sigma_floor."


def test_normalize_z_to_bg_aligns_background():
    """Test case: background-normalized mapping aligns background across images."""
    rng = np.random.default_rng(2)
    I1 = rng.normal(0.0, 0.01, (256, 256)).astype(np.float32)
    I2 = rng.normal(0.05, 0.01, (256, 256)).astype(np.float32)

    # Add the same blob to both images
    I1[120:130, 120:130] = 1.0
    I2[120:130, 120:130] = 1.0

    P1_norm, _ = positivity_probability_map(I1, return_debug=True, normalize_z_to_bg=True)
    P2_norm, _ = positivity_probability_map(I2, return_debug=True, normalize_z_to_bg=True)

    bg_mask = np.ones((256, 256), dtype=bool)
    bg_mask[120:130, 120:130] = False
    bg_diff = abs(np.median(P1_norm[bg_mask]) - np.median(P2_norm[bg_mask]))
    assert bg_diff < 0.05, "Background probabilities should align with normalize_z_to_bg."

    P_default = positivity_probability_map(I1, return_debug=False)
    P_legacy = positivity_probability_map(I1, return_debug=False, normalize_z_to_bg=False)
    assert np.allclose(P_default, P_legacy), "Legacy behavior should be unchanged by default."