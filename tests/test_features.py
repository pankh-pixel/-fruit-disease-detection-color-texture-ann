"""
tests/test_features.py
======================
Unit tests for the feature extraction pipeline.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import (
    extract_color_histogram,
    extract_color_statistics,
    extract_features,
    get_feature_dim,
)


@pytest.fixture
def dummy_rgb_image():
    """128×128 random float32 RGB image."""
    rng = np.random.RandomState(42)
    return rng.rand(128, 128, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Colour histogram tests
# ---------------------------------------------------------------------------

class TestColorHistogram:
    def test_output_length(self, dummy_rgb_image):
        bins = 32
        feat = extract_color_histogram(dummy_rgb_image, bins=bins)
        # 3 RGB + 3 HSV channels, each with `bins` bins
        assert feat.shape == (bins * 6,)

    def test_normalized_sums_to_one(self, dummy_rgb_image):
        bins = 16
        feat = extract_color_histogram(dummy_rgb_image, bins=bins, normalize=True)
        # Each of the 6 channel histograms should sum to ~1
        for i in range(6):
            channel_hist = feat[i * bins: (i + 1) * bins]
            assert abs(channel_hist.sum() - 1.0) < 1e-5

    def test_non_negative(self, dummy_rgb_image):
        feat = extract_color_histogram(dummy_rgb_image)
        assert (feat >= 0).all()

    def test_wrong_shape_raises(self):
        bad_image = np.random.rand(128, 128).astype(np.float32)   # grayscale
        with pytest.raises(ValueError):
            extract_color_histogram(bad_image)


# ---------------------------------------------------------------------------
# Colour statistics tests
# ---------------------------------------------------------------------------

class TestColorStatistics:
    def test_output_length(self, dummy_rgb_image):
        feat = extract_color_statistics(dummy_rgb_image)
        assert feat.shape == (9,)   # 3 channels × 3 stats

    def test_dtype(self, dummy_rgb_image):
        feat = extract_color_statistics(dummy_rgb_image)
        assert feat.dtype == np.float32

    def test_uniform_image_zero_std(self):
        """Uniform image must have zero standard deviation in every channel."""
        uniform = np.full((64, 64, 3), 0.5, dtype=np.float32)
        feat = extract_color_statistics(uniform)
        # Indices 1, 4, 7 are the std values for each channel
        for std_idx in [1, 4, 7]:
            assert feat[std_idx] < 1e-5, f"Expected near-zero std at index {std_idx}"


# ---------------------------------------------------------------------------
# Combined feature vector tests
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_consistent_dim(self, dummy_rgb_image):
        """Feature vector length must match get_feature_dim()."""
        expected_dim = get_feature_dim(image_shape=(128, 128, 3))
        feat = extract_features(dummy_rgb_image)
        assert feat.shape[0] == expected_dim

    def test_reproducibility(self, dummy_rgb_image):
        """Identical inputs must produce identical feature vectors."""
        f1 = extract_features(dummy_rgb_image)
        f2 = extract_features(dummy_rgb_image)
        np.testing.assert_array_equal(f1, f2)

    def test_different_images_differ(self):
        rng = np.random.RandomState(0)
        img_a = rng.rand(128, 128, 3).astype(np.float32)
        img_b = rng.rand(128, 128, 3).astype(np.float32)
        f_a = extract_features(img_a)
        f_b = extract_features(img_b)
        assert not np.allclose(f_a, f_b), "Different images produced identical features"
