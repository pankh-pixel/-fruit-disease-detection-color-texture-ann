"""
data_ingestion.py
=================
Handles all data loading, validation, cleaning, and splitting operations.

Expected dataset folder structure
----------------------------------
data/raw/
    ├── Apple_Healthy/
    │   ├── img001.jpg
    │   └── ...
    ├── Apple_Scab/
    │   └── ...
    └── Mango_Anthracnose/
        └── ...
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the config.yaml file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded from %s", config_path)
    return config


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

def scan_dataset(raw_dir: str) -> pd.DataFrame:
    """Walk the dataset directory and build a flat image manifest.

    Args:
        raw_dir: Root directory where class sub-folders reside.

    Returns:
        DataFrame with columns ``['filepath', 'label']``.

    Raises:
        ValueError: If no images are found.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {raw_dir}")

    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    records: List[Dict[str, str]] = []

    for class_dir in sorted(raw_path.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in supported_ext:
                records.append({"filepath": str(img_file), "label": label})

    if not records:
        raise ValueError(
            f"No images found in '{raw_dir}'. "
            "Ensure sub-folders exist with supported image formats."
        )

    df = pd.DataFrame(records)
    logger.info(
        "Dataset scan complete — %d images across %d classes",
        len(df),
        df["label"].nunique(),
    )
    return df


def validate_images(df: pd.DataFrame, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Validate images are readable and non-corrupt.

    Args:
        df: Manifest DataFrame produced by :func:`scan_dataset`.
        sample_size: If set, only validate a random sample of this size.

    Returns:
        Cleaned DataFrame with corrupt/unreadable images removed.
    """
    indices_to_check = df.index.tolist()
    if sample_size and sample_size < len(df):
        indices_to_check = random.sample(indices_to_check, sample_size)

    bad_indices: List[int] = []
    for idx in tqdm(indices_to_check, desc="Validating images", unit="img"):
        filepath = df.at[idx, "filepath"]
        img = cv2.imread(filepath)
        if img is None or img.size == 0:
            logger.warning("Corrupt/unreadable image skipped: %s", filepath)
            bad_indices.append(idx)

    if bad_indices:
        df = df.drop(index=bad_indices).reset_index(drop=True)
        logger.info("Removed %d corrupt images.", len(bad_indices))

    return df


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(
    filepath: str,
    target_size: Tuple[int, int] = (128, 128),
    color_mode: str = "rgb",
) -> np.ndarray:
    """Load a single image from disk, resize, and normalise to [0, 1].

    Args:
        filepath: Absolute or relative path to the image file.
        target_size: ``(height, width)`` to resize to.
        color_mode: ``'rgb'`` or ``'grayscale'``.

    Returns:
        Numpy array of shape ``(H, W, C)`` for RGB or ``(H, W)`` for grayscale,
        with dtype ``float32`` values in ``[0, 1]``.

    Raises:
        IOError: If the image cannot be read.
    """
    img = cv2.imread(filepath)
    if img is None:
        raise IOError(f"Cannot read image: {filepath}")

    if color_mode == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_mode == "grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported color_mode: {color_mode!r}")

    img = cv2.resize(img, (target_size[1], target_size[0]))  # cv2 uses (W, H)
    return img.astype(np.float32) / 255.0


def load_dataset_images(
    df: pd.DataFrame,
    target_size: Tuple[int, int] = (128, 128),
    color_mode: str = "rgb",
) -> Tuple[np.ndarray, List[str]]:
    """Batch-load all images referenced in the manifest.

    Args:
        df: Manifest DataFrame with ``filepath`` column.
        target_size: ``(height, width)`` target resize dimensions.
        color_mode: ``'rgb'`` or ``'grayscale'``.

    Returns:
        Tuple of ``(images_array, labels_list)`` where *images_array* has
        shape ``(N, H, W, C)``.
    """
    images: List[np.ndarray] = []
    labels: List[str] = []
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images", unit="img"):
        try:
            img = load_image(row["filepath"], target_size, color_mode)
            images.append(img)
            labels.append(row["label"])
        except IOError as exc:
            logger.warning("Skipping image due to error: %s", exc)
            failed += 1

    if failed:
        logger.warning("%d images could not be loaded.", failed)

    return np.array(images, dtype=np.float32), labels


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    test_split: float = 0.20,
    val_split: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split.

    Args:
        df: Full manifest DataFrame.
        test_split: Fraction of data reserved for the test set.
        val_split: Fraction of *remaining* data reserved for validation.
        seed: Random state for reproducibility.

    Returns:
        Tuple of ``(train_df, val_df, test_df)``.
    """
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_split,
        stratify=df["label"],
        random_state=seed,
    )
    adjusted_val = val_split / (1.0 - test_split)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val,
        stratify=train_val_df["label"],
        random_state=seed,
    )

    logger.info(
        "Split sizes — Train: %d | Val: %d | Test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_labels(
    train_labels: List[str],
    val_labels: List[str],
    test_labels: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Fit a :class:`~sklearn.preprocessing.LabelEncoder` on training labels
    and transform all splits.

    Args:
        train_labels: String labels for the training split.
        val_labels: String labels for the validation split.
        test_labels: String labels for the test split.

    Returns:
        Tuple of ``(y_train, y_val, y_test, encoder)`` where integer arrays are
        returned (one-hot encoding is deferred to the model layer).
    """
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_labels)
    y_val   = encoder.transform(val_labels)
    y_test  = encoder.transform(test_labels)

    logger.info("Classes: %s", list(encoder.classes_))
    return y_train, y_val, y_test, encoder


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def run_ingestion(config: dict) -> dict:
    """Full ingestion pipeline: scan → validate → split → load.

    Args:
        config: Parsed configuration dictionary.

    Returns:
        Dictionary containing ``X_train``, ``X_val``, ``X_test``,
        ``y_train``, ``y_val``, ``y_test``, and ``encoder``.
    """
    data_cfg = config["data"]
    img_size = tuple(data_cfg["image_size"])

    df = scan_dataset(data_cfg["raw_dir"])
    df = validate_images(df)

    train_df, val_df, test_df = split_dataset(
        df,
        test_split=data_cfg["test_split"],
        val_split=data_cfg["val_split"],
        seed=config["project"]["seed"],
    )

    X_train, y_train_raw = load_dataset_images(train_df, img_size, data_cfg["color_mode"])
    X_val,   y_val_raw   = load_dataset_images(val_df,   img_size, data_cfg["color_mode"])
    X_test,  y_test_raw  = load_dataset_images(test_df,  img_size, data_cfg["color_mode"])

    y_train, y_val, y_test, encoder = encode_labels(
        y_train_raw, y_val_raw, y_test_raw
    )

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "encoder": encoder,
    }
