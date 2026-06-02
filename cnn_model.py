"""
SignAI Model Inference Engine
==============================
Supports two model types:
  1. landmark_dnn  — MediaPipe 63-feature DNN (high accuracy, lighting-invariant)
  2. pixel_cnn     — Raw pixel CNN (fallback, requires trained pixel model)

Optimizations:
  - model(x, training=False) instead of model.predict() → eliminates tf.function retracing
  - Model warm-up on load with dummy input
  - EMA score smoothing across frames in PredictionBuffer
"""

import collections
import json
import logging
import os
import threading
import time
import traceback

import numpy as np

logger = logging.getLogger(__name__)

LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

_model       = None
_model_lock  = threading.Lock()
_model_type  = 'unknown'
_is_demo     = False
_call_fn     = None   # compiled callable (avoids retracing)

_latency_buffer: collections.deque = collections.deque(maxlen=1000)


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """Load model from disk. Detects type via model_meta.json. Thread-safe."""
    global _model, _model_type, _is_demo, _call_fn

    with _model_lock:
        if _model is not None:
            return _model

        if not os.path.isfile(model_path):
            logger.warning("Model file not found at '%s'. Running in demo mode.", model_path)
            _is_demo    = True
            _model_type = 'demo'
            return None

        try:
            import tensorflow as tf
            logger.info("Loading model from %s ...", model_path)
            _model   = tf.keras.models.load_model(model_path, compile=False)
            _is_demo = False

            # ── Auto-detect model type from input shape ────────────────────
            input_shape = tuple(_model.input_shape)
            if len(input_shape) == 2 and input_shape[-1] == 63:
                _model_type = 'landmark_dnn'
                logger.info("Auto-detected: landmark_dnn  (input %s)", input_shape)
            elif len(input_shape) == 4 and input_shape[3] == 3:
                # Any H×W×3 image CNN (50×50, 64×64, 96×96, etc.)
                _model_type = 'pixel_cnn'
                logger.info("Auto-detected: pixel_cnn  (input %s)", input_shape)
            else:
                # Try reading meta as fallback
                meta_path = os.path.join(os.path.dirname(model_path), 'model_meta.json')
                if os.path.isfile(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    _model_type = meta.get('model_type', 'pixel_cnn')
                else:
                    _model_type = 'pixel_cnn'
                logger.info("Unknown input shape %s — using model_type: %s", input_shape, _model_type)

            logger.info("Model loaded. Type=%s  Input=%s  Output=%s",
                        _model_type, _model.input_shape, _model.output_shape)

            # ── Warm-up: compile the graph once with a real-shaped input ──────
            try:
                if _model_type == 'landmark_dnn':
                    dummy = tf.constant(np.zeros((1, 63), dtype=np.float32))
                else:
                    h, w = input_shape[1], input_shape[2]
                    dummy = tf.constant(np.zeros((1, h, w, 3), dtype=np.float32))

                # Build a tf.function that fixes the input signature — eliminates retracing
                @tf.function(input_signature=[tf.TensorSpec(shape=dummy.shape, dtype=tf.float32)])
                def _fixed_call(x):
                    return _model(x, training=False)

                _call_fn = _fixed_call
                _call_fn(dummy)   # actual warm-up run
                logger.info("Model warm-up complete. Graph compiled.")
            except Exception as wu_exc:
                logger.warning("Warm-up failed (will use fallback): %s", wu_exc)
                _call_fn = None

            return _model

        except Exception as exc:
            logger.error("Failed to load model: %s\n%s", exc, traceback.format_exc())
            _is_demo    = True
            _model_type = 'demo'
            return None


def get_model():
    with _model_lock:
        return _model


def is_demo_mode() -> bool:
    return _is_demo


def get_model_type() -> str:
    return _model_type


def reload_model(model_path: str):
    """
    Hot-reload: reset state and load a new model from disk.
    Safe to call while the server is running.
    """
    global _model, _model_type, _is_demo, _call_fn
    with _model_lock:
        _model      = None
        _call_fn    = None
        _model_type = 'unknown'
        _is_demo    = False
    load_model(model_path)


# ─── Feature Extraction ───────────────────────────────────────────────────────

def extract_landmark_features(mp_results) -> np.ndarray | None:
    """
    Extract 63-dimensional feature vector from MediaPipe results.

    Normalization MUST match train_improved.py → landmarks_to_features():
      1. Subtract wrist (make wrist-relative, since templates start at origin)
      2. Flatten to 63-D vector
      3. L2-normalize the full vector  ← this is what training used
    """
    try:
        if not mp_results or not mp_results.multi_hand_landmarks:
            return None

        landmarks = mp_results.multi_hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                       dtype=np.float32)  # (21, 3)

        # Step 1: wrist-relative (landmarks_to_features receives wrist-at-origin data)
        pts -= pts[0]

        # Step 2+3: flatten then L2-normalize  (matches training exactly)
        flat = pts.flatten()
        norm = np.linalg.norm(flat)
        if norm > 1e-6:
            flat = flat / norm

        return flat.reshape(1, -1)   # (1, 63)
    except Exception as exc:
        logger.warning("extract_landmark_features error: %s", exc)
        return None


# ─── Inference ────────────────────────────────────────────────────────────────

def _run_model(x: np.ndarray) -> np.ndarray:
    """
    Run the model on input x.
    Uses the pre-compiled tf.function (_call_fn) when available
    to avoid tf.function retracing on every call.
    Falls back to model(x, training=False) which is still better than model.predict().
    """
    import tensorflow as tf

    if _call_fn is not None:
        return _call_fn(tf.constant(x, dtype=tf.float32)).numpy()

    return _model(tf.constant(x, dtype=tf.float32), training=False).numpy()


def cnn_predict(preprocessed_frame: np.ndarray, mp_results=None) -> dict:
    """
    Run inference. Chooses landmark or pixel path automatically.

    Args:
        preprocessed_frame: (1, 50, 50, 3) for pixel CNN (ignored in landmark mode)
        mp_results: MediaPipe Hands results object (used in landmark mode)
    Returns:
        dict with letter, confidence, top5, letter_scores, latency_ms
    """
    t0    = time.perf_counter()
    model = get_model()

    # ── Demo mode ─────────────────────────────────────────────────────────────
    if _is_demo or model is None:
        scores  = np.random.dirichlet(np.ones(26)).tolist()
        elapsed = (time.perf_counter() - t0) * 1000
        _latency_buffer.append(elapsed)
        return _format_result(scores, elapsed)

    # ── Landmark DNN ──────────────────────────────────────────────────────────
    if _model_type == 'landmark_dnn':
        features = extract_landmark_features(mp_results)
        if features is None:
            scores  = [1.0 / 26.0] * 26
            elapsed = (time.perf_counter() - t0) * 1000
            return _format_result(scores, elapsed, hand_detected=False)
        try:
            raw     = _run_model(features)
            scores  = raw[0].tolist()
            elapsed = (time.perf_counter() - t0) * 1000
            _latency_buffer.append(elapsed)
            return _format_result(scores, elapsed)
        except Exception as exc:
            logger.error("Landmark DNN inference error: %s", exc)
            raise RuntimeError(f"Inference error: {exc}") from exc

    # ── Pixel CNN ─────────────────────────────────────────────────────────────
    try:
        raw     = _run_model(preprocessed_frame)
        scores  = raw[0].tolist()
        elapsed = (time.perf_counter() - t0) * 1000
        _latency_buffer.append(elapsed)
        return _format_result(scores, elapsed)
    except Exception as exc:
        logger.error("Pixel CNN inference error: %s", exc)
        raise RuntimeError(f"CNN inference error: {exc}") from exc


def _format_result(scores: list, latency_ms: float, hand_detected: bool = True) -> dict:
    indexed  = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_idx, top_conf = indexed[0]
    top5 = [
        {"letter": LETTERS[i], "confidence": round(c * 100, 2)}
        for i, c in indexed[:5]
    ]
    letter_scores = [
        {"letter": LETTERS[i], "score": round(scores[i] * 100, 4)}
        for i in range(26)
    ]
    return {
        "letter":        LETTERS[top_idx],
        "confidence":    round(top_conf * 100, 2),
        "top5":          top5,
        "letter_scores": letter_scores,
        "latency_ms":    round(latency_ms, 2),
        "hand_detected": hand_detected,
        "model_type":    _model_type,
    }


# ─── Latency ──────────────────────────────────────────────────────────────────

def get_avg_latency() -> float:
    if not _latency_buffer:
        return 0.0
    return round(sum(_latency_buffer) / len(_latency_buffer), 2)


# ─── Prediction Buffer (temporal smoothing + EMA) ─────────────────────────────

class PredictionBuffer:
    """
    Stabilise predictions over a sliding window of N frames.

    Two-layer smoothing:
      1. Vote majority  — the letter that appears most often wins
      2. EMA on scores  — exponential moving average of raw softmax scores
                          across frames, so confidence values don't flicker

    A prediction is "stable" (= appended to the word) only when:
      - The winning letter has >= min_votes out of window_size frames
      - Its average confidence >= min_confidence %
    """

    def __init__(self,
                 window_size:    int   = 10,
                 min_votes:      int   = 7,
                 min_confidence: float = 75.0,
                 ema_alpha:      float = 0.35):
        """
        Args:
            window_size:    Number of frames to consider (larger → more stable, slower)
            min_votes:      Minimum frame-votes needed to confirm a letter
            min_confidence: Minimum average confidence (%) to confirm
            ema_alpha:      EMA smoothing factor (0 = full smoothing, 1 = no smoothing)
        """
        self.window_size    = window_size
        self.min_votes      = min_votes
        self.min_confidence = min_confidence
        self.ema_alpha      = ema_alpha

        self._buffer: collections.deque = collections.deque(maxlen=window_size)
        self._ema_conf: dict[str, float] = {}   # per-letter smoothed confidence

    def add(self, letter: str, confidence: float):
        self._buffer.append((letter, confidence))

        # Update EMA for this letter
        prev = self._ema_conf.get(letter, confidence)
        self._ema_conf[letter] = (self.ema_alpha * confidence
                                  + (1.0 - self.ema_alpha) * prev)

        # Decay all other letters' EMA toward 0 slowly
        for l in list(self._ema_conf):
            if l != letter:
                self._ema_conf[l] = self._ema_conf[l] * (1.0 - self.ema_alpha * 0.3)

    def get_smoothed_confidence(self, letter: str) -> float:
        """Return EMA-smoothed confidence for a letter."""
        return round(self._ema_conf.get(letter, 0.0), 2)

    def get_stable(self) -> dict:
        buf_len = len(self._buffer)

        if buf_len == 0:
            return {'stable': False, 'letter': None,
                    'stable_letter': None, 'buffer_votes': {}, 'fill': 0}

        vote_counts: dict[str, int]   = {}
        conf_sums:   dict[str, float] = {}

        for letter, conf in self._buffer:
            vote_counts[letter] = vote_counts.get(letter, 0) + 1
            conf_sums[letter]   = conf_sums.get(letter, 0.0) + conf

        best_letter   = max(vote_counts, key=vote_counts.get)
        best_votes    = vote_counts[best_letter]
        raw_avg_conf  = conf_sums[best_letter] / best_votes
        smoothed_conf = self.get_smoothed_confidence(best_letter)

        # Use smoothed confidence for display, raw average for stability gate
        stable = (buf_len >= self.window_size
                  and best_votes >= self.min_votes
                  and raw_avg_conf >= self.min_confidence)

        return {
            'stable':        stable,
            'letter':        best_letter if stable else None,
            'stable_letter': best_letter,
            'buffer_votes':  vote_counts,
            'fill':          buf_len,
            'avg_confidence': round(smoothed_conf, 2),
        }

    def reset(self):
        self._buffer.clear()
        self._ema_conf.clear()


# ─── Model Building Helpers ───────────────────────────────────────────────────

def build_pixel_cnn(input_shape=(50, 50, 3), num_classes=26):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ], name='SignAI_PixelCNN')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
