"""
trainer.py
==========
End-to-end training orchestration with support for:
  - MLflow experiment tracking
  - Weights & Biases (W&B) logging
  - Keras callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
  - Classification report & confusion matrix generation
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, label_binarize
from tensorflow import keras
from tensorflow.keras import utils as keras_utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback builders
# ---------------------------------------------------------------------------

def build_callbacks(config: dict) -> list:
    """Instantiate Keras training callbacks from configuration.

    Args:
        config: Full parsed configuration dictionary.

    Returns:
        List of :class:`keras.callbacks.Callback` instances.
    """
    t_cfg = config["training"]
    ckpt_path = config["paths"]["checkpoint_path"]
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor=t_cfg["early_stopping"]["monitor"],
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=t_cfg["early_stopping"]["monitor"],
            patience=t_cfg["early_stopping"]["patience"],
            restore_best_weights=t_cfg["early_stopping"]["restore_best_weights"],
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=t_cfg["reduce_lr"]["monitor"],
            factor=t_cfg["reduce_lr"]["factor"],
            patience=t_cfg["reduce_lr"]["patience"],
            min_lr=t_cfg["reduce_lr"]["min_lr"],
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(config["paths"]["log_dir"], "training_log.csv"),
            append=False,
        ),
        keras.callbacks.TensorBoard(
            log_dir=config["paths"]["log_dir"],
            histogram_freq=1,
        ),
    ]
    return callbacks


# ---------------------------------------------------------------------------
# Experiment loggers
# ---------------------------------------------------------------------------

def _init_mlflow(config: dict) -> Optional[Any]:
    """Initialise an MLflow run.

    Args:
        config: Full configuration dictionary.

    Returns:
        Active MLflow run object, or ``None`` if MLflow is unavailable.
    """
    try:
        import mlflow
        log_cfg = config["logging"]
        mlflow.set_experiment(log_cfg["experiment_name"])
        run_name = log_cfg["run_name"] or f"run_{int(time.time())}"
        run = mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            **config["training"],
            **config["model"],
        })
        logger.info("MLflow run started: %s", run.info.run_id)
        return run
    except ImportError:
        logger.warning("MLflow not installed. Skipping MLflow tracking.")
        return None


def _init_wandb(config: dict) -> Optional[Any]:
    """Initialise a Weights & Biases run.

    Args:
        config: Full configuration dictionary.

    Returns:
        W&B run object, or ``None`` if wandb is unavailable.
    """
    try:
        import wandb
        log_cfg = config["logging"]
        run = wandb.init(
            project=config["project"]["name"],
            name=log_cfg["run_name"] or None,
            config={**config["training"], **config["model"]},
        )
        logger.info("W&B run started: %s", run.name)
        return run
    except ImportError:
        logger.warning("wandb not installed. Skipping W&B tracking.")
        return None


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    class_names: Optional[list] = None,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Execute the training loop.

    Args:
        model: Compiled :class:`keras.Model`.
        X_train: Feature matrix for training, shape ``(N_train, F)``.
        y_train: Integer labels for training, shape ``(N_train,)``.
        X_val: Feature matrix for validation.
        y_val: Integer labels for validation.
        config: Full configuration dictionary.
        class_names: Optional list of class names for logging.

    Returns:
        Tuple of ``(trained_model, history)``.
    """
    t_cfg = config["training"]
    num_classes = model.output_shape[-1]

    # One-hot encode labels
    Y_train = keras_utils.to_categorical(y_train, num_classes)
    Y_val   = keras_utils.to_categorical(y_val,   num_classes)

    # Callbacks
    callbacks = build_callbacks(config)

    # Optional experiment tracking
    log_backend = config["logging"]["backend"].lower()
    mlflow_run = wandb_run = None

    if log_backend == "mlflow":
        mlflow_run = _init_mlflow(config)
        if mlflow_run:
            try:
                from mlflow.keras import MlflowCallback  # type: ignore
                callbacks.append(MlflowCallback())
            except ImportError:
                pass

    elif log_backend == "wandb":
        wandb_run = _init_wandb(config)
        if wandb_run:
            try:
                import wandb
                callbacks.append(wandb.keras.WandbCallback(save_model=False))
            except ImportError:
                pass

    logger.info("Training started — epochs=%d | batch=%d", t_cfg["epochs"], t_cfg["batch_size"])
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=t_cfg["epochs"],
        batch_size=t_cfg["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    # Finalise trackers
    if mlflow_run:
        try:
            import mlflow
            if config["logging"]["log_model"]:
                mlflow.keras.log_model(model, artifact_path="model")
            mlflow.end_run()
        except Exception as exc:
            logger.warning("MLflow finalisation error: %s", exc)

    if wandb_run:
        try:
            wandb_run.finish()
        except Exception as exc:
            logger.warning("W&B finalisation error: %s", exc)

    return model, history


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    encoder: LabelEncoder,
    output_dir: str = "logs",
) -> Dict[str, Any]:
    """Evaluate model on the held-out test set and produce reports.

    Args:
        model: Trained :class:`keras.Model`.
        X_test: Feature matrix for testing.
        y_test: Integer labels for testing.
        encoder: Fitted :class:`~sklearn.preprocessing.LabelEncoder`.
        output_dir: Directory where reports and figures will be saved.

    Returns:
        Dictionary containing ``precision``, ``recall``, ``f1``, and
        ``accuracy`` (macro averages).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_classes = len(encoder.classes_)
    Y_test = keras_utils.to_categorical(y_test, num_classes)

    # Raw Keras evaluation
    results = model.evaluate(X_test, Y_test, verbose=0)
    metrics_names = model.metrics_names
    keras_metrics = dict(zip(metrics_names, results))
    logger.info("Keras test metrics: %s", keras_metrics)

    # Sklearn classification report
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    report_str = classification_report(
        y_test, y_pred, target_names=encoder.classes_, digits=4
    )
    logger.info("\n%s", report_str)

    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_str)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm, encoder.classes_, output_dir)

    # Training curves are plotted separately (from History)
    report_dict = classification_report(
        y_test, y_pred, target_names=encoder.classes_, output_dict=True
    )
    macro = report_dict.get("macro avg", {})

    summary = {
        "accuracy":  keras_metrics.get("accuracy", 0.0),
        "precision": macro.get("precision", 0.0),
        "recall":    macro.get("recall", 0.0),
        "f1":        macro.get("f1-score", 0.0),
    }
    with open(os.path.join(output_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Evaluation complete: %s", summary)
    return summary


def plot_training_curves(
    history: keras.callbacks.History,
    output_dir: str = "logs",
) -> None:
    """Save training / validation loss and accuracy curves.

    Args:
        history: Keras History object returned by :meth:`~keras.Model.fit`.
        output_dir: Directory to save the PNG figure.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(history.history["loss"],     label="Train Loss",  linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss",    linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Categorical Cross-Entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history.history["accuracy"],     label="Train Acc", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Acc",   linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training curves saved to %s", path)


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: np.ndarray,
    output_dir: str,
) -> None:
    """Internal helper: save a heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved to %s", path)
