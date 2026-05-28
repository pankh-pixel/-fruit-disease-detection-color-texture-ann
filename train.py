"""
train.py  (v4 — on-the-fly augmentation, no RAM crashes)
=============================================================
"""
from __future__ import annotations
import argparse, logging, os, pickle, sys, warnings
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.data_ingestion import load_config, run_ingestion
from src.features       import build_feature_matrix
from src.augmentation   import augment_image, augmented_generator, steps_per_epoch
from src.model          import (build_ann, compile_model, build_sklearn_ensemble,
                                 save_model, save_sklearn_model)
from src.trainer        import build_callbacks, plot_training_curves
from tensorflow.keras   import utils as ku


BATCH_SIZE     = 32
AUGMENT_FACTOR = 8


def setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "train.log"), "w"),
        ],
    )

def banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")

def report(name: str, y_true, y_pred, class_names) -> float:
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  [{name}]  Accuracy: {acc*100:.2f}%")
    print(classification_report(y_true, y_pred,
                                target_names=class_names, digits=4))
    return acc


def build_augmented_features(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    augment_factor: int,
    fkw: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix for sklearn ensemble without loading all
    augmented images into RAM at once.

    Processes one augmentation pass at a time, extracts features
    immediately, then discards the pixel data.
    """
    print(f"  [1/{augment_factor+1}] Original images...")
    X_feat = [build_feature_matrix(X_raw, **fkw)]
    y_feat = [y_raw]

    for i in range(augment_factor):
        print(f"  [{i+2}/{augment_factor+1}] Augmentation pass {i+1}...")
        # Augment one pass worth of images, extract features, then discard pixels
        X_pass = np.array(
            [augment_image(img) for img in X_raw], dtype=np.float32
        )
        X_feat.append(build_feature_matrix(X_pass, **fkw))
        y_feat.append(y_raw)
        del X_pass          # free pixel RAM immediately

    return np.concatenate(X_feat, axis=0), np.concatenate(y_feat, axis=0)


def main(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    setup_logging(config["paths"]["log_dir"])
    logger = logging.getLogger(__name__)

    np.random.seed(42)
    tf.random.set_seed(42)

    # ── 1. Ingest ─────────────────────────────────────────────────────────────
    banner("STAGE 1 — Loading Your Dataset")
    data        = run_ingestion(config)
    encoder     = data["encoder"]
    class_names = list(encoder.classes_)
    num_classes = len(class_names)

    print(f"\n  Dataset info:")
    print(f"    Classes     : {class_names}")
    print(f"    Train images: {len(data['X_train'])}")
    print(f"    Val images  : {len(data['X_val'])}")
    print(f"    Test images : {len(data['X_test'])}")

    # ── 2. Feature extraction (sklearn) ──────────────────────────────────────
    # For sklearn ensemble: augment + extract features pass-by-pass (no RAM crash)
    banner("STAGE 2 — Extracting Features (with on-the-fly augmentation)")
    f_cfg = config["features"]
    fkw   = dict(
        color_bins     = f_cfg["color_histogram"]["bins"],
        glcm_distances = f_cfg["glcm"]["distances"],
        glcm_angles    = f_cfg["glcm"]["angles"],
        glcm_properties= f_cfg["glcm"]["properties"],
    )

    print(f"  Building augmented feature matrix ({AUGMENT_FACTOR}x)...")
    print(f"  Memory-safe: processes one pass at a time, pixels freed after each pass.")
    X_tr_sk, y_tr_sk = build_augmented_features(
        data["X_train"], data["y_train"], AUGMENT_FACTOR, fkw
    )

    print("\n  Validation features...")
    X_val_sk = build_feature_matrix(data["X_val"],  **fkw)
    print("  Test features...")
    X_te_sk  = build_feature_matrix(data["X_test"], **fkw)

    y_val = data["y_val"]
    y_te  = data["y_test"]

    print(f"\n  Feature vector length : {X_tr_sk.shape[1]} dimensions")
    print(f"  Augmented train size  : {len(X_tr_sk)} samples")

    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_tr_sk)
    X_val_sc  = scaler.transform(X_val_sk)
    X_te_sc   = scaler.transform(X_te_sk)

    # ── 3. Sklearn Ensemble ───────────────────────────────────────────────────
    banner("STAGE 3 — Training Sklearn Ensemble (SVM + RF + GBM)")
    print("  This takes 2–4 minutes...")
    ensemble = build_sklearn_ensemble()
    ensemble.fit(X_tr_sc, y_tr_sk)
    print("  Ensemble training complete ✓")

    ens_val_pred  = ensemble.predict(X_val_sc)
    ens_val_acc   = accuracy_score(y_val, ens_val_pred)
    ens_test_pred = ensemble.predict(X_te_sc)
    ens_proba     = ensemble.predict_proba(X_te_sc)
    ens_acc       = report("Sklearn Ensemble", y_te, ens_test_pred, class_names)

    # ── 4. ANN (on-the-fly pixel augmentation via generator) ─────────────────
    banner("STAGE 4 — Training ANN (on-the-fly augmentation generator)")
    print("  Pixels are augmented per-batch — no large array ever allocated.")

    input_dim = X_tr_sk.shape[1]
    ann = build_ann(
        input_dim          = input_dim,
        num_classes        = num_classes,
        hidden_layers      = [256, 128, 64, 32],
        dropout_rate       = 0.25,
        batch_normalization= True,
    )
    ann = compile_model(ann, optimizer="adam", learning_rate=0.0005)
    ann.summary()

    Y_val_cat = ku.to_categorical(y_val, num_classes)

    # Generator yields (feature_batch, label_batch) — but for ANN we need
    # features, not raw pixels. So we wrap the pixel generator with feature
    # extraction on each batch.
    def feature_gen():
        """Wraps augmented_generator: pixels → features on the fly."""
        gen = augmented_generator(
            data["X_train"], data["y_train"],
            batch_size=BATCH_SIZE,
            augment_factor=AUGMENT_FACTOR,
        )
        for X_batch_px, y_batch in gen:
            X_batch_feat = build_feature_matrix(X_batch_px, **fkw)
            X_batch_sc   = scaler.transform(X_batch_feat)
            Y_batch_cat  = ku.to_categorical(y_batch, num_classes)
            yield X_batch_sc, Y_batch_cat

    train_steps = steps_per_epoch(len(data["X_train"]), AUGMENT_FACTOR, BATCH_SIZE)

    # Wrap as tf.data.Dataset for Keras compatibility
    train_ds = tf.data.Dataset.from_generator(
        feature_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
        )
    ).prefetch(tf.data.AUTOTUNE)

    history = ann.fit(
        train_ds,
        steps_per_epoch = train_steps,
        epochs          = 200,
        validation_data = (X_val_sc, Y_val_cat),
        callbacks       = build_callbacks(config),
        verbose         = 1,
    )
    plot_training_curves(history, output_dir=config["paths"]["log_dir"])

    ann_proba     = ann.predict(X_te_sc, verbose=0)
    ann_test_pred = np.argmax(ann_proba, axis=1)
    ann_acc       = report("ANN", y_te, ann_test_pred, class_names)

    # ── 5. Weighted Ensemble Vote ─────────────────────────────────────────────
    banner("STAGE 5 — Combined Vote (ANN + Sklearn Ensemble)")

    ann_val_pred = np.argmax(ann.predict(X_val_sc, verbose=0), axis=1)
    ann_val_acc  = accuracy_score(y_val, ann_val_pred)

    total = ann_val_acc + ens_val_acc
    w_ann = ann_val_acc / total
    w_ens = ens_val_acc / total

    print(f"\n  ANN val accuracy      : {ann_val_acc*100:.1f}%  (weight: {w_ann:.2f})")
    print(f"  Ensemble val accuracy : {ens_val_acc*100:.1f}%  (weight: {w_ens:.2f})")

    combined  = (w_ann * ann_proba) + (w_ens * ens_proba)
    comb_pred = np.argmax(combined, axis=1)
    comb_acc  = report("Combined Vote (ANN + Ensemble)", y_te, comb_pred, class_names)

    # ── 6. Save everything ────────────────────────────────────────────────────
    banner("STAGE 6 — Saving Models")
    model_dir = config["paths"]["model_dir"]
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    save_model(ann, os.path.join(model_dir, "ann_model.keras"))
    save_sklearn_model(ensemble, os.path.join(model_dir, "sklearn_ensemble.pkl"))

    with open(os.path.join(model_dir, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)
    with open(os.path.join(model_dir, "ensemble_weights.pkl"), "wb") as f:
        pickle.dump({"w_ann": float(w_ann), "w_ens": float(w_ens)}, f)

    print(f"\n  All models saved to {model_dir}/")

    # ── Final summary ─────────────────────────────────────────────────────────
    best = max(ens_acc, ann_acc, comb_acc)
    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(f"  Sklearn Ensemble   : {ens_acc*100:.2f}%")
    print(f"  ANN                : {ann_acc*100:.2f}%")
    print(f"  Combined (BEST)    : {comb_acc*100:.2f}%")
    print("="*60)

    if best >= 0.80:
        print(f"\n  ✅ TARGET REACHED!  Best = {best*100:.2f}%  🎉")
    elif best >= 0.70:
        print(f"\n  📈 Getting close! {best*100:.2f}%")
        print("     • Collect 50 more images per class")
        print("     • Try increasing AUGMENT_FACTOR to 12")
    else:
        print(f"\n  ⚠️  {best*100:.2f}% — check data/raw/ for mislabeled images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)