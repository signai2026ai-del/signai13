import base64
import io
import json
import logging
import os
import threading
import time
import traceback
import uuid
from datetime import timedelta
from functools import wraps

from dotenv import load_dotenv

load_dotenv(override=True)

from flask import (
    Flask, Response, jsonify, redirect, render_template,
    request, session, url_for,
)
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import check_password_hash, generate_password_hash

import cnn_model
import db_manager
from config import get_config
from preprocess import decode_base64_image, preprocess_crop, preprocess_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

cfg = get_config()

app = Flask(__name__)
app.config.from_object(cfg)

CORS(app, supports_credentials=True)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=cfg.RATELIMIT_STORAGE_URI,
)

db_manager.init_db_config(cfg)
_db_ok = db_manager.init_db()
if not _db_ok:
    logger.warning("Database initialization failed. Running without persistence.")

cnn_model.load_model(cfg.MODEL_PATH)

# Per-user PredictionBuffer instances keyed by session ID
# TTL-based cleanup: buffers unused for more than BUFFER_TTL_SECONDS are evicted
_pred_buffers: dict[str, cnn_model.PredictionBuffer] = {}
_buffer_last_access: dict[str, float] = {}
_buffers_lock = threading.Lock()
BUFFER_TTL_SECONDS = 3600  # evict buffers idle for more than 1 hour


def _get_buffer(session_id: str) -> cnn_model.PredictionBuffer:
    now = time.time()
    with _buffers_lock:
        # Evict stale buffers to prevent unbounded memory growth
        stale = [sid for sid, t in _buffer_last_access.items()
                 if now - t > BUFFER_TTL_SECONDS]
        for sid in stale:
            _pred_buffers.pop(sid, None)
            _buffer_last_access.pop(sid, None)

        if session_id not in _pred_buffers:
            _pred_buffers[session_id] = cnn_model.PredictionBuffer()
        _buffer_last_access[session_id] = now
        return _pred_buffers[session_id]


# ─── CAMERA MANAGER ──────────────────────────────────────────────────────────

class CameraManager:
    def __init__(self):
        self._cap = None
        self._lock = threading.Lock()
        self._running = False
        self._latest_frame = None
        self._thread: threading.Thread | None = None

    def open(self, index: int = 0) -> bool:
        with self._lock:
            if self._cap and self._cap.isOpened():
                return True
            import cv2, sys
            # On Windows, DirectShow (CAP_DSHOW) is more stable than
            # the default MSMF backend which causes "can't grab frame" errors
            if sys.platform == "win32":
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                logger.error("Failed to open camera index %d", index)
                return False
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap = cap
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return True

    def _capture_loop(self):
        import cv2
        while True:
            # Check running flag inside lock, then release before blocking read()
            with self._lock:
                if not self._running or self._cap is None:
                    break
                cap = self._cap
            ret, frame = cap.read()          # blocking — must be outside the lock
            if ret:
                with self._lock:
                    self._latest_frame = cv2.flip(frame, 1)
            time.sleep(0.01)                 # small yield — also outside the lock

    def read(self):
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None

    def close(self):
        with self._lock:
            self._running = False
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self._latest_frame = None

    def is_open(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()


camera_mgr = CameraManager()


@app.teardown_appcontext
def _close_camera(_exc):
    pass  # Camera is released by /api/camera/close or process exit


import atexit
atexit.register(camera_mgr.close)

# ─── MEDIAPIPE HAND DETECTOR ─────────────────────────────────────────────────

_mp_hands = None
_mp_drawing = None
_mp_lock = threading.Lock()


def _get_mp_hands():
    global _mp_hands, _mp_drawing
    with _mp_lock:
        if _mp_hands is None:
            try:
                import mediapipe as mp
                _mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=True,   # HTTP frames are not a continuous stream
                    max_num_hands=1,
                    min_detection_confidence=0.6,
                    # min_tracking_confidence is ignored when static_image_mode=True
                )
                _mp_drawing = mp.solutions.drawing_utils
                logger.info("MediaPipe Hands initialized.")
            except Exception as exc:
                logger.warning("MediaPipe not available: %s. Skipping hand detection.", exc)
        return _mp_hands, _mp_drawing


def detect_hand(frame_rgb):
    """Run MediaPipe hand detection. Returns (bbox, landmarks_result) or (None, None)."""
    hands, _ = _get_mp_hands()
    if hands is None:
        return None, None
    try:
        with _mp_lock:
            results = hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return None, None
        landmarks = results.multi_hand_landmarks[0]
        h, w = frame_rgb.shape[:2]
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        return bbox, results
    except Exception as exc:
        logger.warning("Hand detection error: %s", exc)
        return None, None


def draw_landmarks(frame_bgr, mp_results):
    """Draw hand skeleton on the frame."""
    _, mp_drawing = _get_mp_hands()
    if mp_drawing is None or mp_results is None:
        return frame_bgr
    try:
        import mediapipe as mp
        for hand_lm in mp_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr, hand_lm, mp.solutions.hands.HAND_CONNECTIONS
            )
    except Exception:
        pass
    return frame_bgr


# ─── AUTH HELPERS ─────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"success": False, "data": None, "error": "Authentication required"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"success": False, "data": None, "error": "Authentication required"}), 401
        user = db_manager.get_user_by_id(session["user_id"])
        if not user or not user.get("is_admin"):
            return jsonify({"success": False, "data": None, "error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated


def _ok(data=None):
    return jsonify({"success": True, "data": data, "error": None})


def _err(msg: str, code: int = 400):
    return jsonify({"success": False, "data": None, "error": msg}), code


def _ensure_recognition_session():
    if "recognition_session_id" not in session:
        session["recognition_session_id"] = str(uuid.uuid4())
    return session["recognition_session_id"]


# ─── PAGE ROUTES ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", user=_current_user())


@app.route("/about")
def about():
    return render_template("about.html", user=_current_user())


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        data = request.form if not request.is_json else request.json
        email = (data.get("email") or "").strip().lower()
        password = (data.get("password") or "").strip()

        if not email or not password:
            error = "Email and password are required."
            if request.is_json:
                return _err(error)
            return render_template("login.html", error=error, user=None)

        user = db_manager.get_user_by_email(email)
        if not user or not check_password_hash(user["password_hash"], password):
            error = "Invalid email or password."
            if request.is_json:
                return _err(error, 401)
            return render_template("login.html", error=error, user=None)

        session.clear()
        session["user_id"] = user["id"]
        session["user_name"] = user["fullname"]
        session.permanent = True
        db_manager.update_last_login(user["id"])
        _ensure_recognition_session()

        if request.is_json:
            return _ok({"redirect": url_for("dashboard")})
        return redirect(url_for("dashboard"))

    return render_template("login.html", user=None)


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        data = request.form if not request.is_json else request.json
        fullname = (data.get("fullname") or "").strip()
        email = (data.get("email") or "").strip().lower()
        password = (data.get("password") or "").strip()

        errors = []
        if not fullname:
            errors.append("Full name is required.")
        if not email or "@" not in email:
            errors.append("A valid email is required.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters.")

        if errors:
            msg = " ".join(errors)
            if request.is_json:
                return _err(msg)
            return render_template("register.html", error=msg, user=None)

        if db_manager.get_user_by_email(email):
            msg = "An account with this email already exists."
            if request.is_json:
                return _err(msg, 409)
            return render_template("register.html", error=msg, user=None)

        pw_hash = generate_password_hash(password)
        user_id = db_manager.create_user(fullname, email, pw_hash)
        if not user_id:
            msg = "Registration failed. Please try again."
            if request.is_json:
                return _err(msg, 500)
            return render_template("register.html", error=msg, user=None)

        session.clear()
        session["user_id"] = user_id
        session["user_name"] = fullname
        session.permanent = True
        _ensure_recognition_session()

        if request.is_json:
            return _ok({"redirect": url_for("dashboard")}), 201
        return redirect(url_for("dashboard"))

    return render_template("register.html", user=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=_current_user())


@app.route("/history")
@login_required
def history():
    return render_template("history.html", user=_current_user())


@app.route("/statistics")
@login_required
def statistics():
    return render_template("statistics.html", user=_current_user())


@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", user=_current_user())


@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html", user=_current_user())


@app.route("/recognition")
@login_required
def recognition():
    _ensure_recognition_session()
    return render_template("recognition.html", user=_current_user())


def _current_user():
    if "user_id" not in session:
        return None
    user = db_manager.get_user_by_id(session["user_id"])
    if user is None:
        return {
            "id": session["user_id"],
            "fullname": session.get("user_name", "User"),
            "email": "",
            "confidence_threshold": 0.70,
            "is_admin": 0,
        }
    return user


# ─── CAMERA API ───────────────────────────────────────────────────────────────

@app.route("/api/camera/open", methods=["POST"])
@login_required
def api_camera_open():
    data = request.get_json(silent=True) or {}
    index = int(data.get("index", 0))
    ok = camera_mgr.open(index)
    if ok:
        return _ok({"message": "Camera opened"})
    return _err("Could not open camera.", 500)


@app.route("/api/camera/close", methods=["POST"])
@login_required
def api_camera_close():
    camera_mgr.close()
    return _ok({"message": "Camera closed"})


@app.route("/video_feed")
@login_required
def video_feed():
    def generate():
        import cv2
        while True:
            frame = camera_mgr.read()
            if frame is None:
                time.sleep(0.05)
                continue

            try:
                import cv2 as cv2_
                frame_rgb = cv2_.cvtColor(frame, cv2_.COLOR_BGR2RGB)
                bbox, mp_results = detect_hand(frame_rgb)

                if mp_results:
                    frame = draw_landmarks(frame, mp_results)
                    # Green border when hand detected
                    cv2_.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1),
                                  (0, 255, 0), 3)
                else:
                    cv2_.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1),
                                  (0, 0, 255), 3)

                t_encode = time.perf_counter()
                ret, buf = cv2_.imencode(".jpg", frame, [cv2_.IMWRITE_JPEG_QUALITY, 80])
                elapsed = (time.perf_counter() - t_encode) * 1000

                # Adaptive quality
                quality = 60 if elapsed > 40 else 80
                if quality != 80:
                    _, buf = cv2_.imencode(".jpg", frame, [cv2_.IMWRITE_JPEG_QUALITY, quality])

                if not ret:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            except BrokenPipeError:
                break
            except Exception as exc:
                logger.warning("video_feed frame error: %s", exc)
                break

            time.sleep(1.0 / cfg.FRAME_RATE)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ─── PREDICTION API ────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
@login_required
@limiter.limit("120 per minute")
def api_predict():
    session_id = _ensure_recognition_session()
    user_id = session["user_id"]
    user = db_manager.get_user_by_id(user_id)
    threshold = float(user.get("confidence_threshold", cfg.CONFIDENCE_THRESHOLD)) * 100

    try:
        # Handle file upload
        if "file" in request.files:
            f = request.files["file"]
            raw = f.read()
            arr = __import__("numpy").frombuffer(raw, dtype=__import__("numpy").uint8)
            import cv2
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return _err("Could not decode uploaded image.")
        else:
            data = request.get_json(silent=True) or {}
            b64 = data.get("image")
            if not b64:
                return _err("Missing 'image' field (Base64) or file upload.")
            try:
                frame = decode_base64_image(b64)
            except ValueError as exc:
                return _err(str(exc))

        import cv2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox, mp_results = detect_hand(frame_rgb)

        if bbox is None:
            # No hand detected
            buf = _get_buffer(session_id)
            buf.reset()
            return _ok({
                "hand_detected": False,
                "letter": None,
                "confidence": 0,
                "stable": False,
                "buffer_votes": {},
            })

        preprocessed = preprocess_crop(frame, bbox, padding=20)
        result = cnn_model.cnn_predict(preprocessed, mp_results=mp_results)

        # Temporal smoothing
        buf = _get_buffer(session_id)
        buf.add(result["letter"], result["confidence"])
        stability = buf.get_stable()

        # Save to DB only if stable and above threshold
        auto_save = data.get("auto_save", True) if not ("file" in request.files) else True
        if stability["stable"] and result["confidence"] >= threshold and auto_save:
            import json as _json
            db_manager.save_prediction(
                user_id=user_id,
                session_id=session_id,
                letter=result["letter"],
                confidence=result["confidence"],
                top5_json=_json.dumps(result["top5"]),
                letter_scores_json=_json.dumps(result["letter_scores"]),
            )

        return _ok({
            "hand_detected": True,
            "letter": result["letter"],
            "confidence": result["confidence"],
            "top5": result["top5"],
            "letter_scores": result["letter_scores"],
            "latency_ms": result["latency_ms"],
            "stable": stability["stable"],
            "stable_letter": stability.get("stable_letter"),
            "buffer_votes": stability["buffer_votes"],
            "buffer_fill": stability["fill"],
            "low_confidence_warning": result["confidence"] < 50,
        })

    except Exception as exc:
        logger.error("api_predict error: %s\n%s", exc, traceback.format_exc())
        return _err("Internal prediction error.", 500)


@app.route("/api/predict/frame", methods=["POST"])
@login_required
@limiter.limit("200 per minute")
def api_predict_frame():
    """
    Server-side prediction: reads the latest frame directly from the server
    camera (OpenCV). No image upload needed from the browser.
    Works over plain HTTP — no HTTPS / getUserMedia required.
    """
    session_id = _ensure_recognition_session()
    user_id = session["user_id"]
    user = db_manager.get_user_by_id(user_id)
    threshold = float((user or {}).get("confidence_threshold", cfg.CONFIDENCE_THRESHOLD)) * 100
    data = request.get_json(silent=True) or {}
    auto_save = bool(data.get("auto_save", True))

    try:
        frame = camera_mgr.read()
        if frame is None:
            return _ok({
                "hand_detected": False,
                "letter": None,
                "confidence": 0,
                "stable": False,
                "buffer_votes": {},
                "camera_ready": False,
            })

        import cv2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox, mp_results = detect_hand(frame_rgb)

        buf = _get_buffer(session_id)
        if bbox is None:
            buf.reset()
            return _ok({
                "hand_detected": False,
                "letter": None,
                "confidence": 0,
                "stable": False,
                "buffer_votes": {},
                "camera_ready": True,
            })

        preprocessed = preprocess_crop(frame, bbox, padding=20)
        result = cnn_model.cnn_predict(preprocessed, mp_results=mp_results)

        buf.add(result["letter"], result["confidence"])
        stability = buf.get_stable()

        if stability["stable"] and result["confidence"] >= threshold and auto_save:
            import json as _json
            db_manager.save_prediction(
                user_id=user_id,
                session_id=session_id,
                letter=result["letter"],
                confidence=result["confidence"],
                top5_json=_json.dumps(result["top5"]),
                letter_scores_json=_json.dumps(result["letter_scores"]),
            )

        return _ok({
            "hand_detected": True,
            "letter": result["letter"],
            "confidence": result["confidence"],
            "top5": result["top5"],
            "letter_scores": result["letter_scores"],
            "latency_ms": result["latency_ms"],
            "stable": stability["stable"],
            "stable_letter": stability.get("stable_letter"),
            "buffer_votes": stability["buffer_votes"],
            "buffer_fill": stability["fill"],
            "low_confidence_warning": result["confidence"] < 50,
            "camera_ready": True,
        })

    except Exception as exc:
        logger.error("api_predict_frame error: %s\n%s", exc, traceback.format_exc())
        return _err("Internal prediction error.", 500)


@app.route("/api/session/reset", methods=["POST"])
@login_required
def api_session_reset():
    session["recognition_session_id"] = str(uuid.uuid4())
    buf = _get_buffer(session["recognition_session_id"])
    buf.reset()
    return _ok({"session_id": session["recognition_session_id"]})


# ─── WORD / SENTENCE BUILDER API ─────────────────────────────────────────────

def _get_word_session():
    """Get or create a word-building session for the current user, cached in Flask session."""
    user_id = session["user_id"]
    ws = db_manager.get_or_create_word_session(user_id)
    return ws


@app.route("/api/word/current", methods=["GET"])
@login_required
def api_word_current():
    ws = _get_word_session()
    return _ok({"sentence": ws.get("sentence", ""), "session_id": ws.get("id")})


@app.route("/api/word/append", methods=["POST"])
@login_required
def api_word_append():
    data = request.get_json(silent=True) or {}
    letter = (data.get("letter") or "").strip().upper()
    if not letter or len(letter) != 1 or not letter.isalpha():
        return _err("'letter' must be a single A-Z character.")
    ws = _get_word_session()
    new_sentence = ws.get("sentence", "") + letter
    db_manager.update_word_session(ws["id"], new_sentence)
    return _ok({"sentence": new_sentence})


@app.route("/api/word/space", methods=["POST"])
@login_required
def api_word_space():
    ws = _get_word_session()
    new_sentence = ws.get("sentence", "") + " "
    db_manager.update_word_session(ws["id"], new_sentence)
    return _ok({"sentence": new_sentence})


@app.route("/api/word/backspace", methods=["POST"])
@login_required
def api_word_backspace():
    ws = _get_word_session()
    current = ws.get("sentence", "")
    new_sentence = current[:-1] if current else ""
    db_manager.update_word_session(ws["id"], new_sentence)
    return _ok({"sentence": new_sentence})


@app.route("/api/word/clear", methods=["POST"])
@login_required
def api_word_clear():
    ws = _get_word_session()
    db_manager.update_word_session(ws["id"], "")
    return _ok({"sentence": ""})


@app.route("/api/word/history", methods=["GET"])
@login_required
def api_word_history():
    rows = db_manager.get_sentence_history(session["user_id"], limit=20)
    return _ok({"history": rows})


# ─── TTS API ──────────────────────────────────────────────────────────────────

@app.route("/api/tts/speak", methods=["POST"])
@login_required
def api_tts_speak():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return _err("'text' field is required.")
    if len(text) > 500:
        return _err("Text too long (max 500 characters).")

    try:
        audio_b64 = _generate_tts(text)
        return _ok({"audio_base64": audio_b64, "format": "mp3"})
    except Exception as exc:
        logger.error("TTS error: %s\n%s", exc, traceback.format_exc())
        return _err("TTS generation failed.", 500)


def _generate_tts(text: str) -> str:
    """Generate TTS audio and return as Base64 MP3 string."""
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        tts = gTTS(text=text, lang="en", slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as exc:
        logger.warning("gTTS failed (%s), trying pyttsx3 fallback.", exc)

    # pyttsx3 fallback — saves to temp file then reads
    try:
        import pyttsx3
        import tempfile
        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        engine.save_to_file(text, path)
        engine.runAndWait()
        with open(path, "rb") as f:
            data = f.read()
        os.unlink(path)
        return base64.b64encode(data).decode()
    except Exception as exc2:
        raise RuntimeError(f"Both TTS engines failed: {exc2}") from exc2


# ─── USER / STATS API ─────────────────────────────────────────────────────────

@app.route("/api/user/me", methods=["GET"])
@login_required
def api_user_me():
    user = db_manager.get_user_by_id(session["user_id"])
    if not user:
        return _err("User not found.", 404)
    return _ok({
        "id": user["id"],
        "fullname": user["fullname"],
        "email": user["email"],
        "is_admin": bool(user.get("is_admin")),
        "confidence_threshold": user.get("confidence_threshold", 0.70),
        "created_at": user["created_at"].isoformat() if user.get("created_at") else None,
        "last_login": user["last_login"].isoformat() if user.get("last_login") else None,
    })


@app.route("/api/stats", methods=["GET"])
@login_required
def api_stats():
    date_range = request.args.get("range", "all")
    if date_range not in ("all", "1d", "7d", "30d"):
        return _err("Invalid range. Use one of: all, 1d, 7d, 30d")
    stats = db_manager.get_user_stats(session["user_id"], date_range=date_range)
    return _ok(stats)


@app.route("/api/history", methods=["GET"])
@login_required
def api_history():
    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(100, max(1, int(request.args.get("per_page", 20))))
    except (TypeError, ValueError):
        return _err("'page' and 'per_page' must be integers.")
    result = db_manager.get_history(session["user_id"], page=page, per_page=per_page)
    return _ok(result)


@app.route("/api/history/clear", methods=["DELETE"])
@login_required
def api_history_clear():
    ok = db_manager.delete_all_predictions(session["user_id"])
    if ok:
        return _ok({"message": "History cleared."})
    return _err("Failed to clear history.", 500)


@app.route("/api/history/delete", methods=["DELETE"])
@login_required
def api_history_delete_selected():
    data = request.get_json(silent=True) or {}
    ids = data.get("ids")
    if not isinstance(ids, list) or not ids:
        return _err("'ids' must be a non-empty list of prediction IDs.")
    try:
        ids = [int(i) for i in ids]
    except (ValueError, TypeError):
        return _err("All IDs must be integers.")
    ok = db_manager.delete_predictions_by_ids(session["user_id"], ids)
    if ok:
        return _ok({"deleted": len(ids)})
    return _err("Delete failed.", 500)


@app.route("/api/profile/update", methods=["POST"])
@login_required
def api_profile_update():
    data = request.get_json(silent=True) or {}
    fullname = (data.get("fullname") or "").strip()
    if not fullname:
        return _err("'fullname' is required.")
    if len(fullname) > 120:
        return _err("Name too long (max 120 characters).")
    ok = db_manager.update_user_name(session["user_id"], fullname)
    if ok:
        session["user_name"] = fullname
        return _ok({"fullname": fullname})
    return _err("Update failed.", 500)


@app.route("/api/profile/change-password", methods=["POST"])
@login_required
def api_change_password():
    data = request.get_json(silent=True) or {}
    current = (data.get("current_password") or "").strip()
    new_pw = (data.get("new_password") or "").strip()

    if not current or not new_pw:
        return _err("Both 'current_password' and 'new_password' are required.")
    if len(new_pw) < 8:
        return _err("New password must be at least 8 characters.")

    user = db_manager.get_user_by_id(session["user_id"])
    if not user or not check_password_hash(user["password_hash"], current):
        return _err("Current password is incorrect.", 403)

    new_hash = generate_password_hash(new_pw)
    ok = db_manager.update_user_password(session["user_id"], new_hash)
    if ok:
        return _ok({"message": "Password updated successfully."})
    return _err("Password update failed.", 500)


@app.route("/api/profile/threshold", methods=["POST"])
@login_required
def api_update_threshold():
    data = request.get_json(silent=True) or {}
    try:
        threshold = float(data.get("confidence_threshold", -1))
    except (TypeError, ValueError):
        return _err("'confidence_threshold' must be a float.")
    if not (0.40 <= threshold <= 0.95):
        return _err("Threshold must be between 0.40 and 0.95.")
    ok = db_manager.update_confidence_threshold(session["user_id"], threshold)
    if ok:
        return _ok({"confidence_threshold": threshold})
    return _err("Update failed.", 500)


@app.route("/api/account/delete", methods=["DELETE"])
@login_required
def api_delete_account():
    user_id = session["user_id"]
    ok = db_manager.delete_user(user_id)
    if ok:
        session.clear()
        return _ok({"message": "Account deleted."})
    return _err("Account deletion failed.", 500)


# ─── ADMIN API ────────────────────────────────────────────────────────────────

@app.route("/api/model_info", methods=["GET"])
def api_model_info():
    """Public endpoint — returns the current model type (no auth needed)."""
    return jsonify({
        "model_type": cnn_model.get_model_type(),
        "demo_mode":  cnn_model.is_demo_mode(),
    })


@app.route("/api/admin/model_stats", methods=["GET"])
@admin_required
def api_admin_model_stats():
    global_stats = db_manager.get_model_stats_global()
    avg_latency = cnn_model.get_avg_latency()
    return _ok({
        **global_stats,
        "avg_latency_ms": avg_latency,
        "demo_mode": cnn_model.is_demo_mode(),
        "model_type": cnn_model.get_model_type(),
        "model_path": cfg.MODEL_PATH,
    })


@app.route("/api/admin/upload_model", methods=["POST"])
@login_required
def api_upload_model():
    """
    Upload a trained Keras model (.h5 or .keras) to replace the current model.
    The model type (landmark_dnn / pixel_cnn) is auto-detected from input shape.
    """
    import shutil
    import tempfile
    if "model" not in request.files:
        return _err("No model file provided. Include it as 'model' in the form-data.")

    f = request.files["model"]
    fname = f.filename or ""
    if not (fname.endswith(".h5") or fname.endswith(".keras")):
        return _err("Unsupported file format. Upload a .h5 or .keras Keras model file.")

    try:
        import tensorflow as tf
        # Save to temp file first to validate
        suffix = ".keras" if fname.endswith(".keras") else ".h5"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        # Validate: try to load
        test_model = tf.keras.models.load_model(tmp_path, compile=False)
        in_shape = tuple(test_model.input_shape)
        out_shape = tuple(test_model.output_shape)

        # Determine type
        if len(in_shape) == 2 and in_shape[-1] == 63:
            detected_type = 'landmark_dnn'
        elif len(in_shape) == 4 and in_shape[3] == 3:
            # Any H×W×3 image CNN (50×50, 64×64, 96×96, …)
            detected_type = 'pixel_cnn'
        else:
            os.unlink(tmp_path)
            return _err(f"Unsupported model input shape {in_shape}. "
                        "Expected (None,63) for landmark DNN or (None,H,W,3) for image CNN.")

        if out_shape[-1] != 26:
            os.unlink(tmp_path)
            return _err(f"Model must output 26 classes (A-Z). Got output shape {out_shape}.")

        del test_model   # free memory before reload

        # Backup current model
        model_dest = os.path.join(os.path.dirname(__file__), cfg.MODEL_PATH)
        if os.path.isfile(model_dest):
            backup = model_dest + ".bak"
            shutil.copy2(model_dest, backup)
            logger.info("Backed up old model to %s", backup)

        # Move validated model into place
        shutil.move(tmp_path, model_dest)
        logger.info("New model installed at %s (type=%s)", model_dest, detected_type)

        # Hot-reload: reset and reload the model safely
        cnn_model.reload_model(cfg.MODEL_PATH)

        return _ok({
            "message": "Model uploaded and loaded successfully.",
            "detected_type": detected_type,
            "input_shape": list(in_shape),
            "output_shape": list(out_shape),
        })

    except Exception as exc:
        logger.error("Model upload error: %s\n%s", exc, traceback.format_exc())
        return _err(f"Failed to load uploaded model: {exc}", 500)


@app.route("/api/admin/start_training", methods=["POST"])
@login_required
def api_start_training():
    """
    Launch fast_train.py as an independent subprocess.
    Returns immediately; poll /api/admin/training_status for progress.
    """
    import subprocess, sys as _sys
    status_path = os.path.join(os.path.dirname(__file__), "models", "training_status.json")
    script_path = os.path.join(os.path.dirname(__file__), "fast_train.py")

    # Check if training is already running
    if os.path.isfile(status_path):
        try:
            with open(status_path) as f:
                st = json.load(f)
            if not st.get("done") and st.get("phase") not in ("error", "done", None):
                age = time.time() - st.get("timestamp", 0)
                if age < 60:
                    return _ok({"message": "Training already running.", "status": st})
        except Exception:
            pass

    try:
        proc = subprocess.Popen(
            [_sys.executable, script_path],
            stdout=open(os.path.join(os.path.dirname(__file__),
                        "models", "train_stdout.log"), "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        # Write an initial status
        with open(status_path, "w") as f:
            json.dump({"phase": "starting", "message": "Training process launched.",
                       "percent": 0, "done": False, "pid": proc.pid,
                       "timestamp": time.time()}, f)
        logger.info("Training subprocess started PID=%s", proc.pid)
        return _ok({"message": "Training started.", "pid": proc.pid})
    except Exception as exc:
        logger.error("Failed to start training: %s", exc)
        return _err(f"Failed to start training: {exc}", 500)


@app.route("/api/admin/training_status", methods=["GET"])
@login_required
def api_training_status():
    """Return the current training status from the JSON status file."""
    status_path = os.path.join(os.path.dirname(__file__), "models", "training_status.json")
    if not os.path.isfile(status_path):
        return _ok({"phase": "idle", "message": "No training started yet.",
                    "percent": 0, "done": False})
    try:
        with open(status_path) as f:
            st = json.load(f)
        return _ok(st)
    except Exception as exc:
        return _err(f"Could not read training status: {exc}", 500)


# ─── NEW PAGES ────────────────────────────────────────────────────────────────

@app.route("/dictionary")
@login_required
def dictionary():
    return render_template("dictionary.html", user=_current_user())


@app.route("/leaderboard")
@login_required
def leaderboard():
    return render_template("leaderboard.html", user=_current_user())


@app.route("/flashcards")
@login_required
def flashcards():
    return render_template("flashcards.html", user=_current_user())


@app.route("/help")
@login_required
def help_page():
    return render_template("help.html", user=_current_user())


@app.route("/practice")
@login_required
def practice():
    return render_template("practice.html", user=_current_user())


@app.route("/achievements")
@login_required
def achievements():
    return render_template("achievements.html", user=_current_user())


@app.route("/progress")
@login_required
def progress():
    return render_template("progress.html", user=_current_user())


# ─── NEW APIS ──────────────────────────────────────────────────────────────────

@app.route("/api/leaderboard", methods=["GET"])
@login_required
def api_leaderboard():
    period   = request.args.get("period", "all")
    user_id  = session["user_id"]

    # Try to get real data from DB; fall back to demo
    try:
        my_stats = db_manager.get_user_stats(user_id, date_range="all") or {}
        my_total = my_stats.get("total_predictions", 0)
        my_freq  = my_stats.get("letter_frequency", {})
        my_unique = len(my_freq)

        # Build demo leaderboard with the current user + synthetic rivals
        import random
        random.seed(42)
        NAMES = ["Alex M.","Jordan K.","Sam R.","Taylor B.","Casey N.","Riley P.","Quinn A.","Morgan L.","Avery T.","Charlie S."]
        base_counts = [random.randint(50, 3000) for _ in NAMES]

        # Insert current user at a realistic rank
        all_entries = [
            {
                "name": n,
                "total": c,
                "unique_letters": random.randint(10, 26),
                "days_active": random.randint(1, 60),
                "is_me": False,
            }
            for n, c in zip(NAMES, base_counts)
        ]
        me_entry = {
            "name": session.get("user_name", "You"),
            "total": my_total,
            "unique_letters": my_unique,
            "days_active": len(my_stats.get("daily_counts", [])),
            "is_me": True,
        }
        all_entries.append(me_entry)
        all_entries.sort(key=lambda x: x["total"], reverse=True)

        my_rank = next((i+1 for i, e in enumerate(all_entries) if e["is_me"]), None)
        return _ok({"rankings": all_entries, "my_rank": my_rank, "my_total": my_total})
    except Exception as exc:
        logger.error("Leaderboard error: %s", exc)
        return _ok({"rankings": [], "my_rank": None, "my_total": 0})


@app.route("/api/achievements", methods=["GET"])
@login_required
def api_achievements():
    user_id = session["user_id"]
    stats   = db_manager.get_user_stats(user_id, date_range="all") or {}
    freq    = stats.get("letter_frequency", {})
    total   = stats.get("total_predictions", 0)
    unique  = len(freq)
    best_c  = stats.get("personal_best", {}).get("confidence", 0) if stats.get("personal_best") else 0
    days    = stats.get("daily_counts", [])
    streak  = 0
    if days:
        from datetime import date, timedelta
        today = date.today()
        for i in range(len(days)):
            d = today - timedelta(days=i)
            if any(r.get("day", "") == d.isoformat() for r in days):
                streak += 1
            else:
                break

    badges = [
        {
            "id":      "first_sign",
            "title":   "First Sign",
            "desc":    "Make your very first prediction",
            "icon":    "hand",
            "color":   "#6c63ff",
            "earned":  total >= 1,
            "progress": min(1, total),
            "goal":    1,
        },
        {
            "id":      "century",
            "title":   "Century",
            "desc":    "Reach 100 total predictions",
            "icon":    "star",
            "color":   "#f59e0b",
            "earned":  total >= 100,
            "progress": min(100, total),
            "goal":    100,
        },
        {
            "id":      "millennium",
            "title":   "Millennium",
            "desc":    "Reach 1,000 total predictions",
            "icon":    "zap",
            "color":   "#ef4444",
            "earned":  total >= 1000,
            "progress": min(1000, total),
            "goal":    1000,
        },
        {
            "id":      "explorer",
            "title":   "Explorer",
            "desc":    "Sign at least 10 different letters",
            "icon":    "compass",
            "color":   "#10b981",
            "earned":  unique >= 10,
            "progress": min(10, unique),
            "goal":    10,
        },
        {
            "id":      "alphabet",
            "title":   "Alphabet Master",
            "desc":    "Sign all 26 letters of the alphabet",
            "icon":    "trophy",
            "color":   "#f59e0b",
            "earned":  unique >= 26,
            "progress": min(26, unique),
            "goal":    26,
        },
        {
            "id":      "precision",
            "title":   "Precision",
            "desc":    "Achieve 90%+ confidence on any letter",
            "icon":    "target",
            "color":   "#06b6d4",
            "earned":  best_c >= 90,
            "progress": min(90, round(best_c)),
            "goal":    90,
        },
        {
            "id":      "perfectionist",
            "title":   "Perfectionist",
            "desc":    "Achieve 99%+ confidence on any letter",
            "icon":    "award",
            "color":   "#8b5cf6",
            "earned":  best_c >= 99,
            "progress": min(99, round(best_c)),
            "goal":    99,
        },
        {
            "id":      "streak3",
            "title":   "On Fire",
            "desc":    "Practice 3 days in a row",
            "icon":    "flame",
            "color":   "#f97316",
            "earned":  streak >= 3,
            "progress": min(3, streak),
            "goal":    3,
        },
        {
            "id":      "streak7",
            "title":   "Week Warrior",
            "desc":    "Practice 7 days in a row",
            "icon":    "calendar",
            "color":   "#ec4899",
            "earned":  streak >= 7,
            "progress": min(7, streak),
            "goal":    7,
        },
    ]
    earned_count = sum(1 for b in badges if b["earned"])
    return _ok({"badges": badges, "streak": streak, "earned": earned_count, "total_badges": len(badges)})


@app.route("/api/progress", methods=["GET"])
@login_required
def api_progress():
    user_id = session["user_id"]
    stats   = db_manager.get_user_stats(user_id, date_range="all") or {}
    freq    = stats.get("letter_frequency", {})
    total   = stats.get("total_predictions", 0)

    letters_data = []
    all_letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    max_count   = max(freq.values()) if freq else 1

    for letter in all_letters:
        count = freq.get(letter, 0)
        letters_data.append({
            "letter":   letter,
            "count":    count,
            "percent":  round(count / max_count * 100) if max_count else 0,
            "rel_pct":  round(count / total * 100, 1) if total else 0,
            "practiced": count > 0,
        })

    practiced = sum(1 for d in letters_data if d["practiced"])
    return _ok({
        "letters": letters_data,
        "total":   total,
        "practiced_count": practiced,
        "remaining_count": 26 - practiced,
    })


# ─── ERROR HANDLERS ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_e):
    if request.is_json or request.path.startswith("/api/"):
        return _err("Not found.", 404)
    return render_template("index.html", error="Page not found.", user=_current_user()), 404


@app.errorhandler(429)
def rate_limit_exceeded(_e):
    return _err("Rate limit exceeded. Please slow down.", 429)


@app.errorhandler(500)
def internal_error(_e):
    logger.error("Internal server error: %s", traceback.format_exc())
    if request.is_json or request.path.startswith("/api/"):
        return _err("Internal server error.", 500)
    return render_template("index.html", error="Internal server error.", user=_current_user()), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") == "development"

    # Disable the watchdog/stat reloader to prevent SystemExit errors when
    # running inside IDEs (Spyder, PyCharm, Jupyter) on Windows and Linux.
    # Set FLASK_USE_RELOADER=true in your environment to enable hot-reload.
    use_reloader = os.environ.get("FLASK_USE_RELOADER", "false").lower() == "true"

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=use_reloader,
    )
