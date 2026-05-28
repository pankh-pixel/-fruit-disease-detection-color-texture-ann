"""
Image augmentation - on-the-fly generator to avoid memory issues with large datasets.
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Iterator, Tuple


def augment_image(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    aug = (image * 255).astype(np.uint8)

    if np.random.rand() > 0.5:
        aug = cv2.flip(aug, 1)
    if np.random.rand() > 0.3:
        aug = cv2.flip(aug, 0)

    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    alpha = np.random.uniform(0.7, 1.3)
    beta  = int(np.random.randint(-30, 30))
    aug   = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    scale = np.random.uniform(0.80, 1.0)
    cx, cy = w // 2, h // 2
    new_w, new_h = int(w * scale), int(h * scale)
    x1 = max(cx - new_w // 2, 0); y1 = max(cy - new_h // 2, 0)
    x2 = min(x1 + new_w, w);      y2 = min(y1 + new_h, h)
    aug = aug[y1:y2, x1:x2]
    aug = cv2.resize(aug, (w, h))

    if np.random.rand() > 0.7:
        aug = cv2.GaussianBlur(aug, (3, 3), 0)

    return aug.astype(np.float32) / 255.0


def augmented_generator(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    augment_factor: int = 8,
    shuffle: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yields (batch_images, batch_labels) forever, generating augmented
    images on the fly so the full augmented set is never held in RAM.

    Each epoch sees  len(images) * augment_factor  samples.
    """
    n = len(images)
    # Build an index pool: original index + whether to augment
    # Pool size = n * augment_factor  (first n are originals, rest are augmented)
    pool_size = n * augment_factor

    while True:                          # infinite loop — Keras calls next() each step
        indices = np.arange(pool_size)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, pool_size, batch_size):
            batch_idx = indices[start : start + batch_size]

            batch_images = []
            batch_labels = []

            for idx in batch_idx:
                orig_idx   = idx % n          # map back to original image
                orig_image = images[orig_idx]
                orig_label = labels[orig_idx]

                if idx < n:
                    # First pass through the pool → return original unchanged
                    batch_images.append(orig_image)
                else:
                    # All other passes → return an augmented copy
                    batch_images.append(augment_image(orig_image))

                batch_labels.append(orig_label)

            yield (
                np.array(batch_images, dtype=np.float32),
                np.array(batch_labels),
            )


def steps_per_epoch(n_samples: int, augment_factor: int, batch_size: int) -> int:
    """Helper so train.py can compute steps_per_epoch in one place."""
    return int(np.ceil((n_samples * augment_factor) / batch_size))