"""
features.py  (v2 — upgraded for higher accuracy)
=================================================
Richer feature set:
  - Color histograms in RGB + HSV + LAB (more discriminative)
  - First-order color statistics (mean, std, skewness per channel)
  - GLCM texture at multiple scales
  - Local Binary Patterns (LBP) — very effective for disease textures
  - Hu Moments — shape-based features
"""
from __future__ import annotations
import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

try:
    from skimage.feature import local_binary_pattern
    from skimage.feature import grayscale_to_float as _  # noqa
    try:
        from skimage.feature._texture import greycomatrix, greycoprops
    except ImportError:
        from skimage.feature import grayscale_matrix as greycomatrix   # type: ignore
        from skimage.feature import grayscale_properties as greycoprops  # type: ignore
except Exception:
    greycomatrix = greycoprops = None  # type: ignore
    local_binary_pattern = None  # type: ignore

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)

def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_to_uint8(image), cv2.COLOR_RGB2GRAY)


# ── color features ────────────────────────────────────────────────────────────

def extract_color_histogram(image: np.ndarray, bins: int = 32,
                             normalize: bool = True) -> np.ndarray:
    img_u8  = _to_uint8(image)
    img_hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    features: List[np.ndarray] = []
    for src in [img_u8, img_hsv, img_lab]:
        for ch in range(3):
            hist, _ = np.histogram(src[:,:,ch], bins=bins, range=(0,256))
            if normalize:
                s = hist.sum()
                hist = hist / s if s > 0 else hist.astype(np.float32)
            features.append(hist.astype(np.float32))
    return np.concatenate(features)  # bins * 9


def extract_color_statistics(image: np.ndarray) -> np.ndarray:
    img_u8  = _to_uint8(image)
    img_hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    stats: List[float] = []
    for src in [image, img_hsv.astype(np.float32)/255.0,
                img_lab.astype(np.float32)/255.0]:
        for ch in range(3):
            c = src[:,:,ch].ravel()
            mean = float(np.mean(c))
            std  = float(np.std(c))
            skew = float(3*(mean - float(np.median(c))) / (std + 1e-9))
            kurt = float(np.mean((c - mean)**4) / (std**4 + 1e-9))
            stats.extend([mean, std, skew, kurt])
    return np.array(stats, dtype=np.float32)  # 3 spaces * 3 ch * 4 = 36


# ── GLCM texture ──────────────────────────────────────────────────────────────

def extract_glcm_features(image: np.ndarray,
                           distances: Optional[List[int]] = None,
                           angles: Optional[List[float]] = None,
                           properties: Optional[List[str]] = None) -> np.ndarray:
    if distances is None:
        distances = [1, 2, 3]
    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    if properties is None:
        properties = ["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]

    gray = _to_gray(image)
    n = len(distances) * len(angles) * len(properties)

    if greycomatrix is None:
        return np.zeros(n, dtype=np.float32)
    try:
        glcm = greycomatrix(gray, distances=distances, angles=angles,
                            symmetric=True, normed=True)
        vals: List[float] = []
        for prop in properties:
            vals.extend(greycoprops(glcm, prop).ravel().tolist())
        return np.array(vals, dtype=np.float32)
    except Exception as e:
        logger.warning("GLCM failed: %s", e)
        return np.zeros(n, dtype=np.float32)


# ── LBP texture (NEW) ─────────────────────────────────────────────────────────

def extract_lbp_features(image: np.ndarray, n_points: int = 24,
                          radius: int = 3, bins: int = 32) -> np.ndarray:
    """Local Binary Patterns — excellent for detecting surface anomalies."""
    gray = _to_gray(image)
    if local_binary_pattern is None:
        return np.zeros(bins, dtype=np.float32)
    try:
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=bins,
                               range=(0, n_points + 2), density=True)
        return hist.astype(np.float32)
    except Exception as e:
        logger.warning("LBP failed: %s", e)
        return np.zeros(bins, dtype=np.float32)


# ── Hu Moments (NEW) ──────────────────────────────────────────────────────────

def extract_hu_moments(image: np.ndarray) -> np.ndarray:
    """Shape descriptors — invariant to rotation, scale, translation."""
    gray = _to_gray(image)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    # Log-transform to bring into reasonable range
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu.astype(np.float32)  # 7 features


# ── Edge features (NEW) ───────────────────────────────────────────────────────

def extract_edge_features(image: np.ndarray, bins: int = 16) -> np.ndarray:
    """Canny edge density histogram — diseased areas have more edges."""
    gray = _to_gray(image)
    edges = cv2.Canny(gray, 50, 150)
    # Edge density in 4x4 grid blocks
    h, w = gray.shape
    bh, bw = h // 4, w // 4
    densities = []
    for i in range(4):
        for j in range(4):
            block = edges[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            densities.append(float(block.mean()) / 255.0)
    return np.array(densities, dtype=np.float32)  # 16 features


# ── Combined feature vector ───────────────────────────────────────────────────

def extract_features(image: np.ndarray, color_bins: int = 32,
                     glcm_distances: Optional[List[int]] = None,
                     glcm_angles: Optional[List[float]] = None,
                     glcm_properties: Optional[List[str]] = None,
                     normalize_hist: bool = True) -> np.ndarray:
    hist  = extract_color_histogram(image, bins=color_bins, normalize=normalize_hist)
    stats = extract_color_statistics(image)
    glcm  = extract_glcm_features(image, glcm_distances, glcm_angles, glcm_properties)
    lbp   = extract_lbp_features(image)
    hu    = extract_hu_moments(image)
    edge  = extract_edge_features(image)
    return np.concatenate([hist, stats, glcm, lbp, hu, edge])


def build_feature_matrix(images: np.ndarray, color_bins: int = 32,
                         glcm_distances: Optional[List[int]] = None,
                         glcm_angles: Optional[List[float]] = None,
                         glcm_properties: Optional[List[str]] = None) -> np.ndarray:
    feats = [extract_features(img, color_bins, glcm_distances,
                               glcm_angles, glcm_properties)
             for img in tqdm(images, desc="Extracting features", unit="img")]
    m = np.vstack(feats).astype(np.float32)
    logger.info("Feature matrix: %s", m.shape)
    return m


def get_feature_dim(image_shape: Tuple[int,int,int] = (128,128,3),
                    color_bins: int = 32, **kw) -> int:
    dummy = np.random.rand(*image_shape).astype(np.float32)
    return extract_features(dummy, color_bins=color_bins).shape[0]
