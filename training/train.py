"""
SignAI Model Trainer
====================
Trains a MediaPipe landmark-based DNN for ASL letter recognition.

Usage:
    python train.py

Output:
    models/sign_language_model.h5   — Keras model (loaded by cnn_model.py)
    models/landmark_model.h5        — Dedicated landmark model (high accuracy)
    models/training_report.txt      — Accuracy report and confusion matrix
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
# NOTE: This script trains on SYNTHETIC landmark data generated from hand-crafted
# templates. It is useful for quick offline testing but should NOT replace a model
# trained on real images. Use SignAI_CNN_Training.ipynb for the production model.
SAMPLES_PER_CLASS  = 600    # Synthetic samples per letter (augmented noise variations)
EXTRA_NOISE_PASSES = 4      # Number of noise-level augmentation passes
EPOCHS             = 60
BATCH_SIZE         = 256
LEARNING_RATE      = 0.002
DROPOUT_RATE       = 0.30
VALIDATION_SPLIT   = 0.15
MODEL_DIR          = os.path.join(os.path.dirname(__file__), '..', 'models')
# Save under its own name to avoid overwriting the real trained model
# (sign_language_model.h5 is produced by SignAI_CNN_Training.ipynb on real data)
LANDMARK_MODEL     = os.path.join(MODEL_DIR, 'synthetic_landmark_model.h5')
PIXEL_MODEL        = os.path.join(MODEL_DIR, 'synthetic_landmark_model.h5')
LETTERS            = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# ── Import TF ─────────────────────────────────────────────────────────────────
print('Loading TensorFlow...', flush=True)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
print(f'  TensorFlow {tf.__version__} on CPU', flush=True)


# ── Build Dataset ─────────────────────────────────────────────────────────────

def build_dataset():
    """Generate synthetic ASL landmark dataset with multiple noise levels."""
    from asl_landmarks import LETTERS as LTRS, generate_training_sample

    print(f'\n{"─"*55}')
    print('DATASET GENERATION')
    print(f'{"─"*55}')
    print(f'Samples per class : {SAMPLES_PER_CLASS}')
    print(f'Total classes     : {len(LTRS)}')
    print(f'Total samples     : ~{SAMPLES_PER_CLASS * len(LTRS)}')

    X_list, y_list = [], []
    noise_levels = np.linspace(0.015, 0.08, EXTRA_NOISE_PASSES)

    t0 = time.time()
    for label_idx, letter in enumerate(LTRS):
        per_level = SAMPLES_PER_CLASS // len(noise_levels)
        for noise in noise_levels:
            for _ in range(per_level):
                feat = generate_training_sample(letter, noise_level=float(noise))
                X_list.append(feat)
                y_list.append(label_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    print(f'Generated {len(X)} samples in {time.time()-t0:.1f}s')
    print(f'Feature shape  : {X.shape}  (21 landmarks × 3 coords = 63)')
    return X, y


# ── Build Landmark DNN ─────────────────────────────────────────────────────────

def build_landmark_model(input_dim: int = 63, num_classes: int = 26) -> tf.keras.Model:
    """
    Deep neural network for landmark-based ASL classification.
    Architecture: BN → Dense(512) → BN → Dropout → Dense(512) → BN → Dropout
               → Dense(256) → BN → Dropout → Dense(128) → BN → Dense(26 softmax)
    """
    inputs = tf.keras.Input(shape=(input_dim,), name='landmarks')

    x = tf.keras.layers.BatchNormalization(name='bn_in')(inputs)

    # Block 1
    x = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='d1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.Activation('relu', name='relu1')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE, name='drop1')(x)

    # Block 2
    x = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='d2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.Activation('relu', name='relu2')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE, name='drop2')(x)

    # Block 3
    x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='d3')(x)
    x = tf.keras.layers.BatchNormalization(name='bn3')(x)
    x = tf.keras.layers.Activation('relu', name='relu3')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE * 0.8, name='drop3')(x)

    # Block 4
    x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='d4')(x)
    x = tf.keras.layers.BatchNormalization(name='bn4')(x)
    x = tf.keras.layers.Activation('relu', name='relu4')(x)

    # Output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs, outputs, name='SignAI_LandmarkDNN')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            weight_decay=1e-5,
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Training Callbacks ────────────────────────────────────────────────────────

def get_callbacks(model_path: str):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f'  Epoch {epoch+1:3d}/{EPOCHS}: '
                f'loss={logs["loss"]:.4f}  acc={logs["accuracy"]:.4f}  '
                f'val_loss={logs["val_loss"]:.4f}  val_acc={logs["val_accuracy"]:.4f}',
                flush=True
            ) if (epoch + 1) % 5 == 0 else None
        ),
    ]


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """Print detailed accuracy report and per-letter scores."""
    print(f'\n{"─"*55}')
    print('EVALUATION REPORT')
    print(f'{"─"*55}')

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    overall_acc = np.mean(y_pred == y_test)
    print(f'Overall Accuracy : {overall_acc * 100:.2f}%')

    print(f'\n{"Letter":>8}  {"Correct":>7}  {"Total":>7}  {"Accuracy":>9}')
    print('─' * 45)
    per_letter_acc = {}
    for i, letter in enumerate(LETTERS):
        mask = y_test == i
        total = mask.sum()
        if total == 0:
            continue
        correct = (y_pred[mask] == i).sum()
        acc = correct / total
        per_letter_acc[letter] = acc
        status = '' if acc >= 0.9 else '  ← LOW'
        print(f'{letter:>8}  {correct:>7d}  {total:>7d}  {acc*100:>8.1f}%{status}')

    # Confusion matrix snippet (worst 5 letters)
    worst = sorted(per_letter_acc.items(), key=lambda x: x[1])[:5]
    if worst:
        print(f'\nLowest accuracy letters: {", ".join(f"{l}({a*100:.0f}%)" for l, a in worst)}')

    return overall_acc, per_letter_acc


# ── Save Landmark Metadata ─────────────────────────────────────────────────────

def save_landmark_metadata():
    """Save a flag file indicating this is a synthetic landmark-based model."""
    meta = {
        'model_type': 'landmark_dnn',
        'input_dim': 63,
        'num_classes': 26,
        'letters': LETTERS,
        'features': 'mediapipe_21_landmarks_normalized',
        'training_data': 'SYNTHETIC — generated from hand-crafted templates (not real images)',
        'description': 'DNN trained on synthetic MediaPipe hand landmark coordinates (63 features)',
    }
    meta_path = os.path.join(MODEL_DIR, 'synthetic_model_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Saved metadata → {meta_path}')


# ── Save Report ────────────────────────────────────────────────────────────────

def save_report(overall_acc, per_letter_acc, history, elapsed):
    report_path = os.path.join(MODEL_DIR, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write('SignAI Landmark Model — Training Report\n')
        f.write('=' * 55 + '\n\n')
        f.write(f'Training time    : {elapsed:.1f}s\n')
        f.write(f'Overall accuracy : {overall_acc * 100:.2f}%\n')
        f.write(f'Epochs trained   : {len(history.history["accuracy"])}\n')
        f.write(f'Best val_acc     : {max(history.history["val_accuracy"])*100:.2f}%\n\n')
        f.write('Per-letter accuracy:\n')
        for l, a in per_letter_acc.items():
            f.write(f'  {l}: {a*100:.1f}%\n')
    print(f'\nReport saved → {report_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '═' * 55)
    print(' SignAI — Model Training Pipeline')
    print('═' * 55)

    os.makedirs(MODEL_DIR, exist_ok=True)
    t_start = time.time()

    # ── Step 1: Generate data ─────────────────────────────────────────────────
    X, y = build_dataset()

    # Train/test split (85/15)
    split = int(len(X) * (1 - VALIDATION_SPLIT))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f'Train set : {len(X_train)} samples')
    print(f'Val set   : {len(X_val)} samples')

    # ── Step 2: Build model ───────────────────────────────────────────────────
    print(f'\n{"─"*55}')
    print('MODEL ARCHITECTURE')
    print(f'{"─"*55}')
    model = build_landmark_model(input_dim=X.shape[1], num_classes=26)
    model.summary(print_fn=lambda s: print('  ' + s))

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print(f'\n{"─"*55}')
    print('TRAINING')
    print(f'{"─"*55}')
    print(f'Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}  |  Dropout: {DROPOUT_RATE}')
    print('(Progress shown every 5 epochs)\n')

    callbacks = get_callbacks(LANDMARK_MODEL)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=0,
    )

    # ── Step 4: Load best weights & evaluate ──────────────────────────────────
    if os.path.exists(LANDMARK_MODEL):
        model = tf.keras.models.load_model(LANDMARK_MODEL)

    overall_acc, per_letter_acc = evaluate_model(model, X_val, y_val)

    # ── Step 5: Save synthetic model (separate filename, never overwrites real model) ──
    model.save(PIXEL_MODEL)
    print(f'\nSynthetic model saved → {PIXEL_MODEL}')
    print(f'NOTE: The production model sign_language_model.h5 was NOT modified.')
    print(f'      Run SignAI_CNN_Training.ipynb to train/update the production model.')

    save_landmark_metadata()

    elapsed = time.time() - t_start
    save_report(overall_acc, per_letter_acc, history, elapsed)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"═"*55}')
    print('TRAINING COMPLETE')
    print(f'{"═"*55}')
    print(f'Overall val accuracy : {overall_acc * 100:.2f}%')
    print(f'Total time           : {elapsed:.1f}s')
    print(f'Model saved to       : {PIXEL_MODEL}')

    if overall_acc >= 0.95:
        print('\n✓ Excellent! Model accuracy ≥ 95%. Ready for deployment.')
    elif overall_acc >= 0.90:
        print('\n✓ Good. Model accuracy ≥ 90%.')
    else:
        print('\n⚠ Model accuracy below 90%. Consider increasing SAMPLES_PER_CLASS.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nTraining interrupted by user.')
        sys.exit(0)
    except Exception as e:
        print(f'\nFATAL ERROR: {e}')
        traceback.print_exc()
        sys.exit(1)
