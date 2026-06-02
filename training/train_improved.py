"""
SignAI — Improved Model Trainer
================================
Trains a MediaPipe landmark-based DNN for ASL letter recognition with:
  • 3000 samples per class (vs 600 before)
  • Rich augmentation: rotation, scale, perspective, finger-length variation
  • Improved model: residual-style skip connections + cosine LR decay
  • Better landmark templates for easily-confused letters (S/A, M/N, U/V)

Run:
    cd signai && python train_improved.py
"""

import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
# NOTE: This script trains on SYNTHETIC landmark data (augmented hand-crafted
# templates). It does NOT use real ASL images. For production use, run
# SignAI_CNN_Training.ipynb which trains on real MediaPipe landmarks.
SAMPLES_PER_CLASS = 3000
EPOCHS            = 80
BATCH_SIZE        = 512
LEARNING_RATE     = 3e-3
DROPOUT_RATE      = 0.35
VALIDATION_SPLIT  = 0.15
L2_REG            = 5e-5
MODEL_DIR         = os.path.join(os.path.dirname(__file__), '..', 'models')
# Save under dedicated names to avoid overwriting the real trained production model
LANDMARK_MODEL    = os.path.join(MODEL_DIR, 'synthetic_improved_landmark_model.h5')
PIXEL_MODEL       = os.path.join(MODEL_DIR, 'synthetic_improved_landmark_model.h5')
REPORT_PATH       = os.path.join(MODEL_DIR, 'synthetic_training_report.txt')
LETTERS           = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

print('Loading TensorFlow...', flush=True)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
print(f'  TensorFlow {tf.__version__} on CPU  ({os.cpu_count()} cores)', flush=True)

np.random.seed(42)
tf.random.set_seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _rot2d(theta):
    """2D rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def augment_landmarks(pts: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Apply realistic geometric augmentation to 21×3 landmark array.

    Augmentations applied:
      1.  Global 2-D rotation  (±25°)
      2.  Non-uniform scale    (x: 0.85-1.15, y: 0.82-1.18)
      3.  Individual finger-length jitter (±10%)
      4.  Per-landmark gaussian noise   (σ = 0.02-0.10)
      5.  Mild z-axis compression        (simulate 2-D projection)
      6.  Wrist-anchor jitter            (translate ± 0.05)
    """
    pts = pts.copy()

    # 1. Global rotation (hand tilted left/right)
    theta = rng.uniform(-0.44, 0.44)   # ±25 degrees
    R = _rot2d(theta)
    pts[:, :2] = (R @ pts[:, :2].T).T

    # 2. Non-uniform scale (hand size / camera distance)
    sx = rng.uniform(0.85, 1.15)
    sy = rng.uniform(0.82, 1.18)
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    pts[:, 2] *= rng.uniform(0.7, 1.3)   # depth scale

    # 3. Finger-length jitter (each finger segment scaled independently)
    # Fingers: thumb (1-4), index (5-8), middle (9-12), ring (13-16), pinky (17-20)
    finger_groups = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]
    for grp in finger_groups:
        scale = rng.uniform(0.90, 1.10)
        anchor = pts[grp[0] - 1]   # MCP or CMC
        for idx in grp:
            pts[idx] = anchor + (pts[idx] - anchor) * scale

    # 4. Per-landmark gaussian noise
    sigma = rng.uniform(0.02, 0.10)
    pts += rng.normal(0, sigma, pts.shape).astype(np.float32)

    # 5. Mild z-axis compression (simulate near-2D MediaPipe output)
    pts[:, 2] *= rng.uniform(0.5, 1.0)

    # 6. Global translation jitter (wrist not perfectly at origin)
    pts[:, 0] += rng.uniform(-0.05, 0.05)
    pts[:, 1] += rng.uniform(-0.05, 0.05)

    return pts.astype(np.float32)


def landmarks_to_features(pts: np.ndarray) -> np.ndarray:
    """Flatten 21×3 → 63-dim and L2-normalize."""
    flat = pts.flatten()
    norm = np.linalg.norm(flat)
    if norm > 1e-6:
        flat = flat / norm
    return flat.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK TEMPLATES  (21 × 3 per letter)
# ─────────────────────────────────────────────────────────────────────────────

from asl_landmarks import _T as _BASE_TEMPLATES, LETTERS as _LTRS

def get_template(letter: str) -> np.ndarray:
    return _BASE_TEMPLATES[letter].copy()


# ─────────────────────────────────────────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset():
    rng = np.random.RandomState(42)
    print(f'\n{"─"*60}')
    print('DATASET GENERATION')
    print(f'{"─"*60}')
    print(f'Samples per class  : {SAMPLES_PER_CLASS}')
    print(f'Total classes      : {len(LETTERS)}')
    print(f'Total samples      : {SAMPLES_PER_CLASS * len(LETTERS):,}')
    print(f'Feature dimension  : 63  (21 landmarks × 3)')
    print()

    X_list, y_list = [], []
    t0 = time.time()

    for label_idx, letter in enumerate(LETTERS):
        template = get_template(letter)
        for _ in range(SAMPLES_PER_CLASS):
            aug = augment_landmarks(template, rng)
            feat = landmarks_to_features(aug)
            X_list.append(feat)
            y_list.append(label_idx)
        print(f'  {letter}: {SAMPLES_PER_CLASS} samples generated', flush=True)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    elapsed = time.time() - t0
    print(f'\nGenerated {len(X):,} samples in {elapsed:.1f}s')
    print(f'X shape: {X.shape}  |  y shape: {y.shape}')
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

def build_model(input_dim: int = 63, num_classes: int = 26) -> tf.keras.Model:
    """
    Deep DNN with residual skip connections for ASL landmark classification.

    Architecture:
        Input(63)
        → BN → Dense(512) → BN → ReLU → Dropout(0.35)
        → Dense(512) → BN → ReLU → Dropout(0.35)  [+ skip from first Dense]
        → Dense(256) → BN → ReLU → Dropout(0.30)
        → Dense(128) → BN → ReLU → Dropout(0.25)
        → Dense(64)  → BN → ReLU
        → Dense(26, softmax)
    """
    reg = tf.keras.regularizers.l2(L2_REG)
    inputs = tf.keras.Input(shape=(input_dim,), name='landmarks')

    # Input normalization
    x = tf.keras.layers.BatchNormalization(name='bn_in')(inputs)

    # Block 1 — 512
    x1 = tf.keras.layers.Dense(512, kernel_regularizer=reg, name='d1')(x)
    x1 = tf.keras.layers.BatchNormalization(name='bn1')(x1)
    x1 = tf.keras.layers.Activation('relu', name='relu1')(x1)
    x1 = tf.keras.layers.Dropout(DROPOUT_RATE, name='drop1')(x1)

    # Block 2 — 512 with residual skip
    x2 = tf.keras.layers.Dense(512, kernel_regularizer=reg, name='d2')(x1)
    x2 = tf.keras.layers.BatchNormalization(name='bn2')(x2)
    x2 = tf.keras.layers.Activation('relu', name='relu2')(x2)
    x2 = tf.keras.layers.Dropout(DROPOUT_RATE, name='drop2')(x2)
    x2 = tf.keras.layers.Add(name='skip1')([x1, x2])   # residual

    # Block 3 — 256
    x3 = tf.keras.layers.Dense(256, kernel_regularizer=reg, name='d3')(x2)
    x3 = tf.keras.layers.BatchNormalization(name='bn3')(x3)
    x3 = tf.keras.layers.Activation('relu', name='relu3')(x3)
    x3 = tf.keras.layers.Dropout(DROPOUT_RATE - 0.05, name='drop3')(x3)

    # Block 4 — 128
    x4 = tf.keras.layers.Dense(128, kernel_regularizer=reg, name='d4')(x3)
    x4 = tf.keras.layers.BatchNormalization(name='bn4')(x4)
    x4 = tf.keras.layers.Activation('relu', name='relu4')(x4)
    x4 = tf.keras.layers.Dropout(DROPOUT_RATE - 0.10, name='drop4')(x4)

    # Block 5 — 64
    x5 = tf.keras.layers.Dense(64, kernel_regularizer=reg, name='d5')(x4)
    x5 = tf.keras.layers.BatchNormalization(name='bn5')(x5)
    x5 = tf.keras.layers.Activation('relu', name='relu5')(x5)

    # Output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x5)

    model = tf.keras.Model(inputs, outputs, name='SignAI_LandmarkDNN_v2')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_val, y_val):
    y_pred = np.argmax(model.predict(X_val, batch_size=1024, verbose=0), axis=1)
    overall_acc = np.mean(y_pred == y_val)
    per_letter = {}
    for i, letter in enumerate(LETTERS):
        mask = y_val == i
        if mask.sum() > 0:
            per_letter[letter] = np.mean(y_pred[mask] == i)
    return overall_acc, per_letter


def save_report(overall_acc, per_letter, history, elapsed):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write('SignAI Landmark Model — Training Report\n')
        f.write('=======================================================\n\n')
        f.write(f'Training time    : {elapsed:.1f}s\n')
        f.write(f'Overall accuracy : {overall_acc * 100:.2f}%\n')
        f.write(f'Epochs trained   : {len(history.history["val_accuracy"])}\n')
        f.write(f'Best val_acc     : {max(history.history["val_accuracy"]) * 100:.2f}%\n\n')
        f.write('Per-letter accuracy:\n')
        for letter, acc in sorted(per_letter.items()):
            bar = '█' * int(acc * 20)
            f.write(f'  {letter}: {acc * 100:6.1f}%  {bar}\n')

    print(f'\nReport saved → {REPORT_PATH}')


def save_metadata(model_type='landmark_dnn'):
    meta = {
        'model_type': model_type,
        'input_shape': [None, 63],
        'labels': LETTERS,
        'trained_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'training_data': 'SYNTHETIC — augmented hand-crafted templates (not real images)',
        'accuracy': None,
    }
    path = os.path.join(MODEL_DIR, 'synthetic_improved_model_meta.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # ── Step 1: Generate dataset ───────────────────────────────────────────────
    X, y = build_dataset()

    # ── Step 2: Split ──────────────────────────────────────────────────────────
    n_val = int(len(X) * VALIDATION_SPLIT)
    X_val, y_val = X[:n_val], y[:n_val]
    X_tr,  y_tr  = X[n_val:], y[n_val:]

    print(f'\nTrain: {len(X_tr):,}  |  Val: {len(X_val):,}')

    # ── Step 3: Build model ───────────────────────────────────────────────────
    print(f'\n{"─"*60}')
    print('MODEL ARCHITECTURE')
    print(f'{"─"*60}')
    model = build_model()
    model.summary()

    # ── Step 4: Compile ───────────────────────────────────────────────────────
    total_steps = (len(X_tr) // BATCH_SIZE) * EPOCHS
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=total_steps,
        alpha=5e-4,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # ── Step 5: Callbacks ──────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=LANDMARK_MODEL,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    print(f'\n{"─"*60}')
    print('TRAINING')
    print(f'{"─"*60}')
    print(f'Epochs     : {EPOCHS}')
    print(f'Batch size : {BATCH_SIZE}')
    print(f'LR (start) : {LEARNING_RATE}  (cosine decay)')
    print()

    history = model.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # ── Step 7: Load best & evaluate ──────────────────────────────────────────
    if os.path.exists(LANDMARK_MODEL):
        model = tf.keras.models.load_model(LANDMARK_MODEL)
        print('\nLoaded best checkpoint.')

    overall_acc, per_letter = evaluate_model(model, X_val, y_val)

    # ── Step 8: Print per-letter results ──────────────────────────────────────
    print(f'\n{"═"*60}')
    print('PER-LETTER ACCURACY (validation set)')
    print(f'{"═"*60}')
    for letter, acc in sorted(per_letter.items()):
        bar = '█' * int(acc * 30)
        mark = '✓' if acc >= 0.95 else ('⚠' if acc >= 0.85 else '✗')
        print(f'  {mark} {letter}: {acc * 100:5.1f}%  {bar}')

    print(f'\n  OVERALL: {overall_acc * 100:.2f}%')

    # ── Step 9: Save synthetic model (dedicated filename, does not touch production) ──
    model.save(LANDMARK_MODEL)
    print(f'\nSynthetic improved model → {LANDMARK_MODEL}')
    print(f'NOTE: sign_language_model.h5 (production model) was NOT modified.')
    print(f'      Run SignAI_CNN_Training.ipynb to train/update the production model.')

    # ── Step 10: Save metadata & report ───────────────────────────────────────
    save_metadata('landmark_dnn')
    elapsed = time.time() - t_start
    save_report(overall_acc, per_letter, history, elapsed)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"═"*60}')
    print('TRAINING COMPLETE')
    print(f'{"═"*60}')
    print(f'Overall accuracy  : {overall_acc * 100:.2f}%')
    print(f'Total time        : {elapsed:.1f}s  ({elapsed/60:.1f} min)')

    low = [(l, a) for l, a in per_letter.items() if a < 0.95]
    if low:
        print(f'\n⚠  Letters below 95%: {", ".join(f"{l}({a*100:.0f}%)" for l,a in low)}')
    else:
        print('\n✓  All letters ≥ 95%  — model ready for deployment.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nTraining interrupted.')
        sys.exit(0)
    except Exception as e:
        print(f'\nFATAL: {e}')
        traceback.print_exc()
        sys.exit(1)
