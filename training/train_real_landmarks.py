"""
Train a landmark DNN on REAL MediaPipe landmarks extracted from local ASL images.

This is the RECOMMENDED training script for production use.
It reads actual images from the dataset, runs MediaPipe Hands to extract
real landmarks, and trains a DNN using the SAME normalization used at inference
time (wrist-relative + L2 normalize), guaranteeing no train/inference mismatch.

Usage (run from inside the signai/training/ directory):
    cd "C:\\path\\to\\signai (1)\\signai\\training"
    python train_real_landmarks.py

Output:
    ../models/sign_language_model.h5    <- production model loaded by app.py
    ../models/model_meta.json           <- model metadata
    ../models/training_report.txt       <- per-letter accuracy report
"""

import os
import sys
import time
import json
import shutil
import collections

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Paths (all relative to this file's directory = signai/training/) ─────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))  # = signai/
MODEL_DIR    = os.path.join(_PROJECT_DIR, "models")

# Dataset: Kaggle ASL Alphabet has a double-nested structure after extraction
_DATASET_BASE = os.path.join(_PROJECT_DIR, "dataset", "asl_alphabet_train")
DATASET_DIR   = os.path.join(_DATASET_BASE, "asl_alphabet_train")
if not os.path.isdir(DATASET_DIR):
    DATASET_DIR = _DATASET_BASE   # fallback if already flat

OUTPUT_MODEL  = os.path.join(MODEL_DIR, "sign_language_model.h5")
BACKUP_MODEL  = os.path.join(MODEL_DIR, "sign_language_model_backup.h5")
META_PATH     = os.path.join(MODEL_DIR, "model_meta.json")
REPORT_PATH   = os.path.join(MODEL_DIR, "training_report.txt")

LETTERS          = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGES_PER_CLASS = 2000           # use up to 2000 of the 3000 available per letter
IMG_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".bmp"}


# ─── Step 1: Extract landmarks from real images ─────────────────────────────

def extract_features_from_image(img_bgr, hands):
    """
    Run MediaPipe on one BGR image.
    Returns a 63-dim feature vector or None if no hand detected.

    Normalization (MUST match cnn_model.extract_landmark_features exactly):
      1. BGR -> RGB
      2. MediaPipe Hands
      3. pts -= pts[0]           (wrist-relative)
      4. flat = pts.flatten()    (63-D)
      5. flat /= L2_norm(flat)   (scale-invariant)
    """
    import cv2
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results  = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None

    lms  = results.multi_hand_landmarks[0]
    pts  = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark],
                    dtype=np.float32)   # (21, 3)

    pts  -= pts[0]                      # wrist-relative
    flat  = pts.flatten()               # (63,)
    norm  = np.linalg.norm(flat)
    if norm > 1e-6:
        flat = flat / norm              # L2-normalize

    return flat


def collect_real_landmarks():
    import cv2
    import mediapipe as mp

    print("=" * 60)
    print("Phase 1 — Extracting real MediaPipe landmarks from images")
    print("=" * 60)
    print(f"Dataset : {DATASET_DIR}")
    print(f"Images  : up to {IMAGES_PER_CLASS} per class")
    print()

    if not os.path.isdir(DATASET_DIR):
        print(f"[ERROR] Dataset folder not found: {DATASET_DIR}")
        print("Make sure the dataset is extracted inside signai/dataset/")
        sys.exit(1)

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.4,
    )

    X_list, y_list = [], []
    detection_stats = {}
    t_start = time.time()

    for label_idx, letter in enumerate(LETTERS):
        letter_dir = os.path.join(DATASET_DIR, letter)
        if not os.path.isdir(letter_dir):
            print(f"  [WARN] Missing folder: {letter_dir}")
            detection_stats[letter] = 0.0
            continue

        files = [
            f for f in os.listdir(letter_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ][:IMAGES_PER_CLASS]

        ok = 0
        for fname in files:
            img = cv2.imread(os.path.join(letter_dir, fname))
            if img is None:
                continue
            feat = extract_features_from_image(img, hands)
            if feat is not None:
                X_list.append(feat)
                y_list.append(label_idx)
                ok += 1

        rate = ok / max(len(files), 1) * 100
        detection_stats[letter] = round(rate, 1)
        bar  = "█" * int(rate / 5)
        print(f"  {letter}: {ok:>4}/{len(files):<4}  ({rate:5.1f}%)  {bar}")

    hands.close()

    elapsed = time.time() - t_start
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"\nExtraction done in {elapsed:.1f}s")
    print(f"Total samples   : {len(X):,}")
    print(f"Feature shape   : {X.shape}")
    mean_rate = np.mean(list(detection_stats.values()))
    print(f"Mean detect rate: {mean_rate:.1f}%")

    return X, y, detection_stats


# ─── Step 2: Build DNN ──────────────────────────────────────────────────────

def build_model(input_dim=63, num_classes=26):
    """
    Residual Dense Network — identical architecture to SignAI_CNN_Training.ipynb
    Input: (None, 63)  Output: (None, 26) softmax
    """
    import tensorflow as tf

    reg = tf.keras.regularizers.l2(5e-5)
    inp = tf.keras.Input(shape=(input_dim,), name="landmarks")

    # Input BN
    x = tf.keras.layers.BatchNormalization()(inp)

    # Block 1 — 512
    x1 = tf.keras.layers.Dense(512, kernel_regularizer=reg)(x)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation("relu")(x1)
    x1 = tf.keras.layers.Dropout(0.35)(x1)

    # Block 2 — 512 + residual skip
    x2 = tf.keras.layers.Dense(512, kernel_regularizer=reg)(x1)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Activation("relu")(x2)
    x2 = tf.keras.layers.Dropout(0.35)(x2)
    x2 = tf.keras.layers.Add()([x1, x2])

    # Block 3 — 256
    x3 = tf.keras.layers.Dense(256, kernel_regularizer=reg)(x2)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Activation("relu")(x3)
    x3 = tf.keras.layers.Dropout(0.30)(x3)

    # Block 4 — 128 + residual skip
    x4 = tf.keras.layers.Dense(128, kernel_regularizer=reg)(x3)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.Activation("relu")(x4)
    x4 = tf.keras.layers.Dropout(0.25)(x4)

    # Block 5 — 64
    x5 = tf.keras.layers.Dense(64, kernel_regularizer=reg)(x4)
    x5 = tf.keras.layers.BatchNormalization()(x5)
    x5 = tf.keras.layers.Activation("relu")(x5)

    out = tf.keras.layers.Dense(num_classes, activation="softmax",
                                name="predictions")(x5)

    model = tf.keras.Model(inputs=inp, outputs=out, name="SignAI_RealLandmarkDNN")
    return model


# ─── Step 3: Train ──────────────────────────────────────────────────────────

def train(X, y):
    import tensorflow as tf

    print("\n" + "=" * 60)
    print("Phase 2 — Training DNN on real landmark features")
    print("=" * 60)
    print(f"  Samples  : {len(X):,}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Classes  : {len(LETTERS)}")

    idx  = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split     = int(len(X) * 0.85)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    print(f"  Train    : {len(X_tr):,}  |  Val: {len(X_val):,}")

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    best_ckpt = os.path.join(MODEL_DIR, "real_landmark_best.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_ckpt,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.4,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0

    # Load best checkpoint
    if os.path.isfile(best_ckpt):
        model = tf.keras.models.load_model(best_ckpt)

    # Per-class accuracy
    pred      = model.predict(X_val, verbose=0).argmax(axis=1)
    per_class = {}
    for i, letter in enumerate(LETTERS):
        mask = y_val == i
        if mask.sum() == 0:
            per_class[letter] = None
            continue
        per_class[letter] = round(float((pred[mask] == i).mean()) * 100, 1)

    best_val = max(history.history["val_accuracy"])

    print(f"\nTraining time  : {elapsed:.1f}s")
    print(f"Best val acc   : {best_val * 100:.2f}%")
    print("\nPer-letter accuracy:")
    for letter, acc in per_class.items():
        if acc is None:
            print(f"  {letter}: N/A (no validation samples)")
        else:
            bar  = "█" * int(acc / 5)
            mark = "✓" if acc >= 95 else ("⚠" if acc >= 85 else "✗")
            print(f"  {mark} {letter}: {acc:5.1f}%  {bar}")

    return model, best_val, elapsed, per_class, history


# ─── Step 4: Save ───────────────────────────────────────────────────────────

def save_everything(model, best_val, elapsed, per_class, detection_stats, history):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Backup the existing production model before overwriting
    if os.path.isfile(OUTPUT_MODEL):
        shutil.copy2(OUTPUT_MODEL, BACKUP_MODEL)
        print(f"\nPrevious model backed up → {BACKUP_MODEL}")

    model.save(OUTPUT_MODEL)
    print(f"New production model saved → {OUTPUT_MODEL}")

    # Update metadata (read by cnn_model.py)
    meta = {
        "model_type"         : "landmark_dnn",
        "input_shape"        : [None, 63],
        "num_classes"        : len(LETTERS),
        "classes"            : LETTERS,
        "architecture"       : "Residual Dense Network (5 blocks)",
        "normalization"      : "wrist_relative_L2",
        "training_data"      : "Real MediaPipe landmarks from local ASL dataset",
        "images_per_class"   : IMAGES_PER_CLASS,
        "final_val_accuracy" : round(best_val * 100, 2),
        "training_time_s"    : round(elapsed, 1),
        "per_class_accuracy" : per_class,
        "mediapipe_detection_rates": detection_stats,
        "trained_at"         : time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved  → {META_PATH}")

    # Training report
    with open(REPORT_PATH, "w") as f:
        f.write("SignAI Real-Landmark Model — Training Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Trained at       : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training time    : {elapsed:.1f}s\n")
        f.write(f"Best val acc     : {best_val * 100:.2f}%\n")
        f.write(f"Training data    : Real MediaPipe landmarks (local ASL dataset)\n")
        f.write(f"Images per class : {IMAGES_PER_CLASS}\n")
        f.write(f"Normalization    : Wrist-relative + L2-normalize (63-D)\n\n")
        f.write("Per-letter accuracy (validation set):\n")
        for letter, acc in per_class.items():
            if acc is None:
                f.write(f"  {letter}:   N/A\n")
            else:
                bar = "█" * int(acc / 5)
                f.write(f"  {letter}: {acc:5.1f}%  {bar}\n")
        f.write("\nMediaPipe detection rates per letter:\n")
        for letter, rate in detection_stats.items():
            f.write(f"  {letter}: {rate:.1f}%\n")

    print(f"Report saved    → {REPORT_PATH}")


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, detection_stats = collect_real_landmarks()

    if len(X) < 100:
        print(f"\n[ERROR] Only {len(X)} samples detected.")
        print("Check that MediaPipe is installed and the dataset is valid.")
        sys.exit(1)

    model, best_val, elapsed, per_class, history = train(X, y)

    save_everything(model, best_val, elapsed, per_class, detection_stats, history)

    print(f"\n{'=' * 60}")
    print("DONE — Restart the Flask server to load the new model.")
    print(f"{'=' * 60}")

    low = [(l, a) for l, a in per_class.items() if a is not None and a < 95]
    if low:
        print(f"\nLetters below 95%: {', '.join(f'{l}({a:.0f}%)' for l, a in low)}")
        print("Consider increasing IMAGES_PER_CLASS or adding more augmentation.")
    else:
        print("\nAll letters >= 95% — model ready for deployment.")
