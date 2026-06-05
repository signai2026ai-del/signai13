import logging
import traceback
import sqlite3
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

_db_path: str = "site.db"

def init_db_config(cfg):
    global _db_path
    _db_path = cfg.DB_PATH

@contextmanager
def _conn():
    """Context manager to handle SQLite connections and transactions."""
    conn = None
    try:
        conn = sqlite3.connect(_db_path)
        # تعيين row_factory لكي تظهر النتائج كقاموس (Dictionary) مثل pymysql
        conn.row_factory = sqlite3.Row 
        yield conn
        # Commit implicitly if no error occurred (mimicking autocommit behavior for DDL/DML)
        conn.commit()
    except sqlite3.Error as exc:
        logger.error("Database error [%s]: %s", type(exc).__name__, exc)
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def init_db():
    """Create tables if they do not exist."""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            
            # جدول المستخدمين
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fullname VARCHAR(120) NOT NULL,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash VARCHAR(256) NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    confidence_threshold REAL NOT NULL DEFAULT 0.70,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME NULL
                )
            """)
            
            # جدول التوقعات
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NULL,
                    session_id VARCHAR(64) NOT NULL,
                    letter CHAR(1) NOT NULL,
                    confidence REAL NOT NULL,
                    top5_json TEXT NULL,
                    letter_scores_json TEXT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            # مؤشرات (Indexes)
            try:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_predictions_user_created "
                    "ON predictions (user_id, created_at)"
                )
            except sqlite3.Error:
                pass

            # جدول الجلسات (الجمل)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    sentence TEXT NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            # محاولة إضافة العمود إذا لم يكن موجوداً (Migration بسيطة)
            try:
                cur.execute("ALTER TABLE users ADD COLUMN confidence_threshold REAL NOT NULL DEFAULT 0.70")
                logger.info("Migrated: added confidence_threshold column to users table.")
            except sqlite3.OperationalError:
                # العمود موجود بالفعل
                pass
                
        logger.info("Database schema initialized successfully.")
        return True
    except sqlite3.Error as exc:
        logger.error("Failed to initialize database schema: %s\n%s", exc, traceback.format_exc())
        return False


def create_user(fullname: str, email: str, password_hash: str) -> int | None:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (fullname, email, password_hash) VALUES (?, ?, ?)",
                (fullname.strip(), email.strip().lower(), password_hash),
            )
            return cur.lastrowid
    except sqlite3.Error as exc:
        logger.error("create_user failed: %s", exc)
        return None


def get_user_by_email(email: str) -> dict | None:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM users WHERE email = ? LIMIT 1",
                (email.strip().lower(),),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    except sqlite3.Error as exc:
        logger.error("get_user_by_email failed: %s", exc)
        return None


def get_user_by_id(user_id: int) -> dict | None:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE id = ? LIMIT 1", (user_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    except sqlite3.Error as exc:
        logger.error("get_user_by_id failed: %s", exc)
        return None


def update_last_login(user_id: int):
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), user_id),
            )
    except sqlite3.Error as exc:
        logger.error("update_last_login failed: %s", exc)


def update_user_name(user_id: int, fullname: str) -> bool:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE users SET fullname = ? WHERE id = ?",
                (fullname.strip(), user_id),
            )
            return True
    except sqlite3.Error as exc:
        logger.error("update_user_name failed: %s", exc)
        return False


def update_user_password(user_id: int, new_hash: str) -> bool:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (new_hash, user_id),
            )
            return True
    except sqlite3.Error as exc:
        logger.error("update_user_password failed: %s", exc)
        return False


def update_confidence_threshold(user_id: int, threshold: float) -> bool:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE users SET confidence_threshold = ? WHERE id = ?",
                (float(threshold), user_id),
            )
            return True
    except sqlite3.Error as exc:
        logger.error("update_confidence_threshold failed: %s", exc)
        return False


def delete_user(user_id: int) -> bool:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return True
    except sqlite3.Error as exc:
        logger.error("delete_user failed: %s", exc)
        return False


def save_prediction(user_id, session_id: str, letter: str, confidence: float,
                    top5_json: str, letter_scores_json: str = None) -> int | None:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO predictions
                   (user_id, session_id, letter, confidence, top5_json, letter_scores_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, session_id, letter, confidence, top5_json, letter_scores_json),
            )
            return cur.lastrowid
    except sqlite3.Error as exc:
        logger.error("save_prediction failed: %s\n%s", exc, traceback.format_exc())
        return None


def get_history(user_id: int, page: int = 1, per_page: int = 20) -> dict:
    offset = (page - 1) * per_page
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) AS total FROM predictions WHERE user_id = ?",
                (user_id,),
            )
            total = cur.fetchone()["total"]
            
            cur.execute(
                """SELECT id, letter, confidence, top5_json, created_at
                   FROM predictions WHERE user_id = ?
                   ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (user_id, per_page, offset),
            )
            rows = [dict(row) for row in cur.fetchall()]
            
            # تحويل التواريخ إلى نص ISO
            for row in rows:
                if isinstance(row.get("created_at"), str):
                    pass # Already stored as ISO string in SQLite usually
                elif row.get("created_at"):
                    row["created_at"] = row["created_at"].isoformat()
                    
        return {
            "rows": rows,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": max(1, -(-total // per_page)),
        }
    except sqlite3.Error as exc:
        logger.error("get_history failed: %s", exc)
        return {"rows": [], "total": 0, "page": page, "per_page": per_page, "pages": 1}


def delete_all_predictions(user_id: int) -> bool:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
            return True
    except sqlite3.Error as exc:
        logger.error("delete_all_predictions failed: %s", exc)
        return False


def delete_predictions_by_ids(user_id: int, ids: list) -> bool:
    if not ids:
        return True
    try:
        placeholders = ", ".join(["?"] * len(ids))
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"DELETE FROM predictions WHERE user_id = ? AND id IN ({placeholders})",
                [user_id] + list(ids),
            )
            return True
    except sqlite3.Error as exc:
        logger.error("delete_predictions_by_ids failed: %s", exc)
        return False


def get_user_stats(user_id: int, date_range: str = "all") -> dict:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            
            # دوال الوقت في SQLite
            time_filter = ""
            if date_range == "1d":
                time_filter = " AND created_at >= datetime('now', '-1 day')"
            elif date_range == "7d":
                time_filter = " AND created_at >= datetime('now', '-7 days')"
            elif date_range == "30d":
                time_filter = " AND created_at >= datetime('now', '-30 days')"

            cur.execute(
                f"""SELECT COUNT(*) AS total,
                           AVG(confidence) AS avg_confidence,
                           COUNT(DISTINCT letter) AS unique_letters,
                           MAX(confidence) AS best_confidence
                    FROM predictions
                    WHERE user_id = ?{time_filter}""",
                (user_id,),
            )
            row = cur.fetchone()

            cur.execute(
                f"""SELECT letter, COUNT(*) AS cnt
                    FROM predictions WHERE user_id = ?{time_filter}
                    GROUP BY letter ORDER BY cnt DESC LIMIT 1""",
                (user_id,),
            )
            top_row = cur.fetchone()

            cur.execute(
                f"""SELECT letter, COUNT(*) AS cnt
                    FROM predictions WHERE user_id = ?{time_filter}
                    GROUP BY letter ORDER BY cnt DESC""",
                (user_id,),
            )
            freq_rows = cur.fetchall()

            cur.execute(
                f"""SELECT date(created_at) AS day, COUNT(*) AS cnt
                    FROM predictions WHERE user_id = ?{time_filter}
                    GROUP BY date(created_at) ORDER BY day DESC""",
                (user_id,),
            )
            daily_rows = cur.fetchall()
            daily_rows = [dict(r) for r in daily_rows]

            cur.execute(
                f"""SELECT date(created_at) AS day, AVG(confidence) AS avg_conf
                    FROM predictions WHERE user_id = ?{time_filter}
                    GROUP BY date(created_at) ORDER BY day ASC""",
                (user_id,),
            )
            conf_over_time = cur.fetchall()
            conf_over_time = [dict(r) for r in conf_over_time]

            cur.execute(
                f"""SELECT strftime('%H', created_at) AS hr, COUNT(*) AS cnt
                    FROM predictions WHERE user_id = ?{time_filter}
                    GROUP BY strftime('%H', created_at) ORDER BY hr ASC""",
                (user_id,),
            )
            hourly = cur.fetchall()
            hourly = [dict(r) for r in hourly]

            cur.execute(
                f"""SELECT letter, confidence, created_at
                    FROM predictions WHERE user_id = ?{time_filter}
                    ORDER BY confidence DESC LIMIT 1""",
                (user_id,),
            )
            best_row = cur.fetchone()
            if best_row:
                best_row = dict(best_row)

        total = int(row["total"] or 0)
        avg_conf = float(row["avg_confidence"] or 0.0)

        return {
            "total_predictions": total,
            "avg_confidence": round(avg_conf, 2),
            "unique_letters": int(row["unique_letters"] or 0),
            "most_predicted": top_row["letter"] if top_row else None,
            "best_confidence": float(row["best_confidence"] or 0.0),
            "letter_frequency": {r["letter"]: r["cnt"] for r in freq_rows},
            "daily_counts": daily_rows,
            "confidence_over_time": conf_over_time,
            "hourly_distribution": hourly,
            "personal_best": best_row,
        }
    except sqlite3.Error as exc:
        logger.error("get_user_stats failed: %s", exc)
        return {}


def get_or_create_word_session(user_id: int) -> dict:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                cur.execute(
                    "INSERT INTO sessions (user_id, sentence) VALUES (?, ?)",
                    (user_id, ""),
                )
                sess_id = cur.lastrowid
                return {"id": sess_id, "user_id": user_id, "sentence": ""}
            return dict(row)
    except sqlite3.Error as exc:
        logger.error("get_or_create_word_session failed: %s", exc)
        return {"id": None, "user_id": user_id, "sentence": ""}


def update_word_session(session_id: int, sentence: str) -> bool:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            # تحديث updated_at يدوياً لأن SQLite لا يدعم ON UPDATE CURRENT_TIMESTAMP
            now = datetime.utcnow().isoformat()
            cur.execute(
                "UPDATE sessions SET sentence = ?, updated_at = ? WHERE id = ?",
                (sentence, now, session_id),
            )
            return True
    except sqlite3.Error as exc:
        logger.error("update_word_session failed: %s", exc)
        return False


def get_sentence_history(user_id: int, limit: int = 20) -> list:
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """SELECT sentence, created_at, updated_at FROM sessions
                   WHERE user_id = ? AND sentence != ''
                   ORDER BY updated_at DESC LIMIT ?""",
                (user_id, limit),
            )
            rows = [dict(r) for r in cur.fetchall()]
            return rows
    except sqlite3.Error as exc:
        logger.error("get_sentence_history failed: %s", exc)
        return []


def get_model_stats_global() -> dict:
    """Return aggregate stats across all users for admin model analytics."""
    try:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS total FROM predictions")
            total = cur.fetchone()["total"]
            cur.execute(
                """SELECT letter, COUNT(*) AS cnt FROM predictions
                   GROUP BY letter ORDER BY cnt DESC LIMIT 5"""
            )
            top5 = cur.fetchall()
            cur.execute(
                """SELECT letter, COUNT(*) AS cnt FROM predictions
                   GROUP BY letter ORDER BY cnt ASC LIMIT 5"""
            )
            bottom5 = cur.fetchall()
        return {"total_inferences": total, "top5_letters": top5, "bottom5_letters": bottom5}
    except sqlite3.Error as exc:
        logger.error("get_model_stats_global failed: %s", exc)
        return {}
