import base64
import logging
import traceback

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SIZE = (64, 64)   # matches MobileNetV2 training; fallback for landmark DNN


def _get_target_size() -> tuple[int, int]:
    """Read the loaded model's spatial input size dynamically."""
    try:
        import cnn_model
        m = cnn_model.get_model()
        if m is not None:
            shape = m.input_shape   # e.g. (None, 64, 64, 3) or (None, 50, 50, 3)
            if len(shape) == 4:
                return (int(shape[2]), int(shape[1]))  # (W, H) for cv2.resize
    except Exception:
        pass
    return DEFAULT_SIZE


def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode a Base64-encoded image string to a BGR numpy array."""
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        raw = base64.b64decode(b64_string, validate=True)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None — image data is corrupt or unsupported.")
        return img
    except (base64.binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid Base64 image data: {exc}") from exc
    except Exception as exc:
        logger.error("decode_base64_image unexpected error: %s\n%s", exc, traceback.format_exc())
        raise ValueError(f"Failed to decode image: {exc}") from exc


def preprocess_frame(frame: np.ndarray, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """Convert a BGR/grayscale frame to a normalized (1, H, W, 3) float32 array.

    target_size: (W, H) for cv2.resize — defaults to the loaded model's input size.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Empty or null frame passed to preprocess_frame.")

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")

    size    = target_size or _get_target_size()
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)


def preprocess_crop(frame: np.ndarray, bbox: tuple, padding: int = 20,
                    target_size: tuple[int, int] | None = None) -> np.ndarray:
    """Crop the hand bounding box (with padding) then preprocess."""
    h, w = frame.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    if x_max <= x_min or y_max <= y_min:
        return preprocess_frame(frame, target_size)

    cropped = frame[y_min:y_max, x_min:x_max]
    return preprocess_frame(cropped, target_size)
