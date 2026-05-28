"""
inference.py
============
Single-image and batch inference utilities.

Typical usage
-------------
>>> from src.inference import Predictor
>>> predictor = Predictor(model_path="models/best.keras", encoder_path="models/encoder.pkl")
>>> result = predictor.predict("path/to/mango.jpg")
>>> print(result)
{'label': 'Mango_Anthracnose', 'confidence': 0.9412, 'top_3': [...]}
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.data_ingestion import load_image, load_config
from src.features import extract_features, build_feature_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class Predictor:
    """High-level inference wrapper for the fruit disease ANN.

    Args:
        model_path: Path to the saved ``.keras`` / ``SavedModel``.
        encoder_path: Path to the pickled :class:`~sklearn.preprocessing.LabelEncoder`.
        config_path: Path to ``config.yaml``.
    """

    def __init__(
        self,
        model_path: str = "models/checkpoints/best_model.keras",
        encoder_path: str = "models/label_encoder.pkl",
        config_path: str = "config.yaml",
    ) -> None:
        self.config = load_config(config_path)
        self.model  = self._load_model(model_path)
        self.encoder = self._load_encoder(encoder_path)
        self._img_size = tuple(self.config["data"]["image_size"])
        self._color_mode = self.config["data"]["color_mode"]
        self._feat_cfg = self.config["features"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(path: str):
        """Load Keras model from disk."""
        from tensorflow import keras as _keras
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")
        model = _keras.models.load_model(path)
        logger.info("Model loaded from %s", path)
        return model

    @staticmethod
    def _load_encoder(path: str):
        """Load a pickled LabelEncoder."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Encoder not found: {path}")
        with open(path, "rb") as f:
            encoder = pickle.load(f)
        logger.info("Encoder loaded from %s — classes: %s", path, list(encoder.classes_))
        return encoder

    def _image_to_features(self, image: np.ndarray) -> np.ndarray:
        """Convert a loaded image array to a model-ready feature vector."""
        f_cfg = self._feat_cfg
        fv = extract_features(
            image,
            color_bins=f_cfg["color_histogram"]["bins"],
            glcm_distances=f_cfg["glcm"]["distances"],
            glcm_angles=f_cfg["glcm"]["angles"],
            glcm_properties=f_cfg["glcm"]["properties"],
            normalize_hist=f_cfg["color_histogram"]["normalize"],
        )
        return fv.reshape(1, -1)   # (1, F)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image_path: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Run inference on a single image file.

        Args:
            image_path: Path to the input image.
            top_k: Number of top predictions to include in the response.

        Returns:
            Dictionary with keys:
              - ``label``: Predicted class name.
              - ``confidence``: Model confidence (0–1).
              - ``top_k``: List of ``{'label': str, 'confidence': float}`` dicts.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image   = load_image(str(path), self._img_size, self._color_mode)
        x       = self._image_to_features(image)
        proba   = self.model.predict(x, verbose=0)[0]   # shape (C,)

        top_indices  = np.argsort(proba)[::-1][:top_k]
        top_labels   = self.encoder.inverse_transform(top_indices)
        top_confs    = proba[top_indices]

        return {
            "label":      top_labels[0],
            "confidence": float(top_confs[0]),
            "top_k": [
                {"label": lbl, "confidence": float(conf)}
                for lbl, conf in zip(top_labels, top_confs)
            ],
        }

    def predict_batch(
        self,
        image_paths: List[str],
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """Run inference on a list of image files.

        Args:
            image_paths: List of image paths.
            top_k: Number of top predictions per image.

        Returns:
            List of prediction dictionaries (same schema as :meth:`predict`).
        """
        results: List[Dict[str, Any]] = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path, top_k=top_k)
            except Exception as exc:
                logger.warning("Inference failed for %s: %s", img_path, exc)
                result = {"label": "ERROR", "confidence": 0.0, "top_k": [], "error": str(exc)}
            results.append(result)
        return results

    def predict_array(
        self,
        image: np.ndarray,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Run inference on a pre-loaded numpy image array.

        Args:
            image: Float32 RGB array ``(H, W, 3)`` with values in ``[0, 1]``.
            top_k: Number of top predictions.

        Returns:
            Prediction dictionary (same schema as :meth:`predict`).
        """
        x     = self._image_to_features(image)
        proba = self.model.predict(x, verbose=0)[0]

        top_indices = np.argsort(proba)[::-1][:top_k]
        top_labels  = self.encoder.inverse_transform(top_indices)
        top_confs   = proba[top_indices]

        return {
            "label":      top_labels[0],
            "confidence": float(top_confs[0]),
            "top_k": [
                {"label": lbl, "confidence": float(conf)}
                for lbl, conf in zip(top_labels, top_confs)
            ],
        }


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

def run_inference_cli(
    image_path: str,
    model_path: str = "models/checkpoints/best_model.keras",
    encoder_path: str = "models/label_encoder.pkl",
    config_path: str = "config.yaml",
    top_k: int = 3,
) -> None:
    """Command-line entry point for single-image inference.

    Args:
        image_path: Path to the image to classify.
        model_path: Path to trained model.
        encoder_path: Path to pickled label encoder.
        config_path: Path to config.yaml.
        top_k: Number of top predictions to display.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    predictor = Predictor(model_path, encoder_path, config_path)
    result = predictor.predict(image_path, top_k=top_k)

    table = Table(title=f"[bold green]Prediction: {result['label']}[/bold green]",
                  show_header=True, header_style="bold cyan")
    table.add_column("Rank",       style="dim",   justify="center")
    table.add_column("Label",      style="white")
    table.add_column("Confidence", style="yellow", justify="right")

    for rank, item in enumerate(result["top_k"], start=1):
        bar   = "█" * int(item["confidence"] * 20)
        label = "[bold green]" + item["label"] + "[/bold green]" if rank == 1 else item["label"]
        table.add_row(str(rank), label, f"{item['confidence']:.4f}  {bar}")

    console.print(table)
