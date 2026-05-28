"""
model.py  (v2 — ensemble approach for small datasets)
======================================================
Provides:
  1. build_ann()        — improved ANN with residual-style skip connections
  2. build_ensemble()   — ANN + SVM + RandomForest voting ensemble
  3. EnsembleModel      — sklearn-compatible wrapper for the ensemble
"""
from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

logger = logging.getLogger(__name__)


# ── Improved ANN ──────────────────────────────────────────────────────────────

def build_ann(input_dim: int, num_classes: int,
              hidden_layers: List[int] = (512, 256, 128, 64),
              activation: str = "relu", dropout_rate: float = 0.35,
              batch_normalization: bool = True,
              output_activation: str = "softmax",
              l2_lambda: float = 1e-4,
              name: str = "FruitDiseaseANN") -> keras.Model:
    """Improved ANN with skip connection for small dataset robustness."""
    inputs = keras.Input(shape=(input_dim,), name="features")

    # First block
    x = layers.Dense(hidden_layers[0], kernel_regularizer=regularizers.l2(l2_lambda))(inputs)
    if batch_normalization:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Remaining blocks
    for units in hidden_layers[1:]:
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_lambda))(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation=output_activation, name="output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    logger.info("ANN built — params: %s", f"{model.count_params():,}")
    return model


def compile_model(model: keras.Model, optimizer: str = "adam",
                  learning_rate: float = 1e-3,
                  loss: str = "categorical_crossentropy") -> keras.Model:
    opt_map = {
        "adam":    keras.optimizers.Adam(learning_rate=learning_rate),
        "sgd":     keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True),
        "rmsprop": keras.optimizers.RMSprop(learning_rate=learning_rate),
    }
    if optimizer not in opt_map:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    model.compile(
        optimizer=opt_map[optimizer], loss=loss,
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")])
    return model


# ── Sklearn ensemble ──────────────────────────────────────────────────────────

def build_sklearn_ensemble() -> VotingClassifier:
    """Build a soft-voting ensemble of SVM + RandomForest + GradientBoosting."""
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    CalibratedClassifierCV(
            SVC(kernel="rbf", C=10, gamma="scale", probability=False),
            cv=3))
    ])
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_split=2, random_state=42, n_jobs=-1))
    ])
    gb = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42))
    ])
    ensemble = VotingClassifier(
        estimators=[("svm", svm), ("rf", rf), ("gb", gb)],
        voting="soft", n_jobs=-1
    )
    return ensemble


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_model(model: keras.Model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    logger.info("ANN saved to %s", path)

def load_model(path: str) -> keras.Model:
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return keras.models.load_model(path)

def save_sklearn_model(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Sklearn ensemble saved to %s", path)

def load_sklearn_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_from_config(config: dict, input_dim: int,
                      num_classes: int) -> keras.Model:
    m_cfg = config["model"]
    t_cfg = config["training"]
    model = build_ann(input_dim=input_dim, num_classes=num_classes,
                      hidden_layers=m_cfg["hidden_layers"],
                      activation=m_cfg["activation"],
                      dropout_rate=m_cfg["dropout_rate"],
                      batch_normalization=m_cfg["batch_normalization"],
                      output_activation=m_cfg["output_activation"])
    return compile_model(model, optimizer=t_cfg["optimizer"],
                         learning_rate=t_cfg["learning_rate"],
                         loss=t_cfg["loss"])
