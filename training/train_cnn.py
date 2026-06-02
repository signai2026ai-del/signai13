"""
SignAI — CNN Training (MobileNetV2 Transfer Learning)
======================================================
Phase 1 : Freeze base, train head only        → 8 epochs
Phase 2 : Unfreeze all, fine-tune full net    → up to 27 epochs (EarlyStopping)

Dataset : signai/dataset/asl_alphabet_train/asl_alphabet_train/
          26 classes A-Z  (del / space / nothing are skipped)
Output  : signai/models/signai_cnn_model.h5
          signai/models/model_meta.json
"""

import os, json, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger,
)

# ─── Config ───────────────────────────────────────────────────────────────────
_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))   # signai/training/
_PROJECT_DIR  = os.path.normpath(os.path.join(_TRAINING_DIR, ".."))  # signai/
_DS_BASE      = os.path.join(_PROJECT_DIR, "dataset", "asl_alphabet_train")
_DS_NESTED    = os.path.join(_DS_BASE, "asl_alphabet_train")
DATASET_DIR   = _DS_NESTED if os.path.isdir(_DS_NESTED) else _DS_BASE
MODEL_DIR     = os.path.join(_PROJECT_DIR, "models")
MODEL_PATH    = os.path.join(MODEL_DIR, "signai_cnn_model.h5")
LOG_PATH      = os.path.join(MODEL_DIR, "training_log.csv")
META_PATH     = os.path.join(MODEL_DIR, "model_meta.json")

IMG_SIZE      = 64          # MobileNetV2 @ 64×64 — fast on CPU, high accuracy
BATCH_SIZE    = 64
CLASSES       = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES   = 26

PHASE1_EPOCHS = 8           # head-only
PHASE2_EPOCHS = 27          # full fine-tune (EarlyStopping patience=6)
TOTAL_EPOCHS  = PHASE1_EPOCHS + PHASE2_EPOCHS

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Startup ──────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("   SignAI CNN Training  —  MobileNetV2 Transfer Learning")
print("="*62)
print(f"  Dataset  : {DATASET_DIR}")
print(f"  IMG_SIZE : {IMG_SIZE}×{IMG_SIZE}")
print(f"  Batch    : {BATCH_SIZE}")
print(f"  Classes  : {NUM_CLASSES}  {CLASSES}")
print("="*62 + "\n")

for cls in CLASSES:
    p = os.path.join(DATASET_DIR, cls)
    if not os.path.isdir(p):
        raise RuntimeError(f"Missing class folder: {p}")
print("✓  All 26 class folders verified\n")

# ─── Data Generators ──────────────────────────────────────────────────────────
print("Creating data generators …")

train_datagen = ImageDataGenerator(
    rescale            = 1.0 / 255,
    validation_split   = 0.15,
    rotation_range     = 18,
    width_shift_range  = 0.15,
    height_shift_range = 0.15,
    shear_range        = 0.12,
    zoom_range         = 0.18,
    brightness_range   = [0.75, 1.30],
    horizontal_flip    = False,          # ASL is chiral — do NOT flip
    fill_mode          = "nearest",
)

val_datagen = ImageDataGenerator(
    rescale          = 1.0 / 255,
    validation_split = 0.15,
)

flow_kw = dict(
    directory   = DATASET_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = "categorical",
    classes     = CLASSES,
    seed        = 42,
)

train_gen = train_datagen.flow_from_directory(subset="training",  shuffle=True,  **flow_kw)
val_gen   = val_datagen.flow_from_directory(  subset="validation", shuffle=False, **flow_kw)

print(f"  Train : {train_gen.samples:,}  images  |  {len(train_gen)} steps/epoch")
print(f"  Val   : {val_gen.samples:,}   images  |  {len(val_gen)} steps\n")

# ─── Model ────────────────────────────────────────────────────────────────────
print("Building model …")

base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                   include_top=False, weights="imagenet")
base.trainable = False

inp  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x    = base(inp, training=False)
x    = layers.GlobalAveragePooling2D()(x)
x    = layers.BatchNormalization()(x)
x    = layers.Dense(512, activation="relu")(x)
x    = layers.Dropout(0.45)(x)
x    = layers.Dense(256, activation="relu")(x)
x    = layers.Dropout(0.30)(x)
out  = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inp, out, name="SignAI_MobileNetV2")

n_train = sum(tf.size(w).numpy() for w in model.trainable_weights)
n_total = sum(tf.size(w).numpy() for w in model.weights)
print(f"  Total params     : {n_total:,}")
print(f"  Trainable        : {n_train:,}  (phase 1 — head only)")

# ─── Phase 1 ─────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print(f"  PHASE 1 — Head Training  ({PHASE1_EPOCHS} epochs)")
print("="*62)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

cb1 = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                      patience=2, verbose=1, min_lr=1e-6),
    CSVLogger(LOG_PATH, append=False),
]

h1 = model.fit(train_gen, epochs=PHASE1_EPOCHS,
               validation_data=val_gen, callbacks=cb1, verbose=1)

best_p1 = max(h1.history.get("val_accuracy", [0]))
print(f"\n✓  Phase 1 complete — best val_acc = {best_p1*100:.2f}%\n")

# ─── Phase 2 ─────────────────────────────────────────────────────────────────
print("="*62)
print(f"  PHASE 2 — Full Fine-tune  (up to {PHASE2_EPOCHS} epochs + EarlyStopping)")
print("="*62)

base.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

n_train2 = sum(tf.size(w).numpy() for w in model.trainable_weights)
print(f"  Trainable params : {n_train2:,}  (all layers unfrozen)\n")

cb2 = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.4,
                      patience=3, verbose=1, min_lr=1e-7),
    EarlyStopping(monitor="val_accuracy", patience=6,
                  restore_best_weights=True, verbose=1),
    CSVLogger(LOG_PATH, append=True),
]

h2 = model.fit(train_gen, initial_epoch=PHASE1_EPOCHS,
               epochs=TOTAL_EPOCHS, validation_data=val_gen,
               callbacks=cb2, verbose=1)

# ─── Evaluation ───────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("  FINAL EVALUATION")
print("="*62)

best_model = tf.keras.models.load_model(MODEL_PATH)

val_eval_gen = val_datagen.flow_from_directory(
    subset="validation", shuffle=False, **flow_kw)

loss, acc = best_model.evaluate(val_eval_gen, verbose=1)
print(f"\n  Validation Accuracy : {acc*100:.2f}%")
print(f"  Validation Loss     : {loss:.4f}")

# Per-letter accuracy
print("\n  Per-letter breakdown:")
val_eval_gen.reset()
y_true, y_pred = [], []
for _ in range(len(val_eval_gen)):
    xb, yb = next(val_eval_gen)
    y_true.extend(np.argmax(yb, axis=1))
    y_pred.extend(np.argmax(best_model.predict(xb, verbose=0), axis=1))

y_true, y_pred = np.array(y_true), np.array(y_pred)
per_class = []
for i, cls in enumerate(CLASSES):
    mask = y_true == i
    if not mask.any():
        continue
    ca = (y_pred[mask] == i).mean() * 100
    per_class.append(ca)
    sym = "✓" if ca >= 90 else "✗"
    print(f"    {sym} {cls}: {ca:.1f}%")

print(f"\n  Mean : {np.mean(per_class):.2f}%  |  "
      f"Min : {np.min(per_class):.2f}%  |  Max : {np.max(per_class):.2f}%")

# ─── Save Metadata ────────────────────────────────────────────────────────────
def to_list(d):
    return {k: [float(v) for v in vals] for k, vals in d.items()}

all_hist = to_list(h1.history)
for k, v in to_list(h2.history).items():
    all_hist.setdefault(k, [])
    all_hist[k] += v

meta = dict(
    model_type         = "pixel_cnn",
    input_shape        = [IMG_SIZE, IMG_SIZE, 3],
    num_classes        = NUM_CLASSES,
    classes            = CLASSES,
    architecture       = "MobileNetV2",
    final_val_accuracy = round(float(acc) * 100, 2),
    final_val_loss     = round(float(loss), 4),
    mean_per_class_acc = round(float(np.mean(per_class)), 2),
    min_per_class_acc  = round(float(np.min(per_class)), 2),
    history            = all_hist,
    dataset            = DATASET_DIR,
    batch_size         = BATCH_SIZE,
    phase1_epochs      = PHASE1_EPOCHS,
    phase2_epochs      = PHASE2_EPOCHS,
)
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✓  Model saved   → {MODEL_PATH}")
print(f"✓  Meta saved    → {META_PATH}")
print(f"✓  Training log  → {LOG_PATH}")
print("\n" + "="*62)
print("  TRAINING COMPLETE")
print("="*62 + "\n")
