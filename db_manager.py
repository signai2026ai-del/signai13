import logging
import traceback
from contextlib import contextmanager
from datetime import datetime
import pymysql
import pymysql.cursors

logger = logging.getLogger(__name__)

_db_config: dict = {}


def init_db_config(cfg):
    global _db_config
    _db_config = {
        "host": cfg.DB_HOST,
        "port": cfg.DB_PORT,
        "user": cfg.DB_USER,
        "password":cfg.DB_PASSWORD,
        "db": cfg.DB_NAME,
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
        "autocommit": True,
        "connect_timeout": 10,
    }
@contextmanager
def _conn():
    conn = None
    try:
        conn = pymysql.connect(**_db_config)
        yield conn
    except pymysql.MySQLError as exc:
        logger.error(
            "Database error [%s]: %s\nConfig host=%s db=%s",
            type(exc).__name__,
            exc,
            _db_config.get("host"),
            _db_config.get("db"),
        )
        with open("db_debug.log", "a") as f:
            f.write(f"Conn error: {exc} | config: {_db_config}\n")
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def init_db():
    """Create database and all tables if they do not exist. Logs and returns False on failure."""
    try:
        root_cfg = dict(_db_config)
        root_cfg.pop("db", None)
        conn = pymysql.connect(**root_cfg)
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{_db_config['db']}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.close()
    except pymysql.MySQLError as exc:
        logger.error("Cannot reach MySQL to create database: %s", exc)
        return False

    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        fullname VARCHAR(120) NOT NULL,
                        email VARCHAR(255) NOT NULL UNIQUE,
                        password_hash VARCHAR(256) NOT NULL,
                        is_admin TINYINT(1) NOT NULL DEFAULT 0,
                        confidence_threshold FLOAT NOT NULL DEFAULT 0.70,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        last_login DATETIME NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id INT NULL,
                        session_id VARCHAR(64) NOT NULL,
                        letter CHAR(1) NOT NULL,
                        confidence FLOAT NOT NULL,
                        top5_json TEXT NULL,
                        letter_scores_json TEXT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                try:
                    cur.execute(
                        "CREATE INDEX idx_predictions_user_created "
                        "ON predictions (user_id, created_at)"
                    )
                except pymysql.MySQLError:
                    pass
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id INT NOT NULL,
                        sentence TEXT NOT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                try:
                    cur.execute(
                        "ALTER TABLE users ADD COLUMN confidence_threshold FLOAT NOT NULL DEFAULT 0.70"
                    )
                    logger.info("Migrated: added confidence_threshold column to users table.")
                except pymysql.MySQLError:
                    pass
        logger.info("Database schema initialized successfully.")
        return True
    except pymysql.MySQLError as exc:
        logger.error("Failed to initialize database schema: %s\n%s", exc, traceback.format_exc())
        return False


def create_user(fullname: str, email: str, password_hash: str) -> int | None:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (fullname, email, password_hash) VALUES (%s, %s, %s)",
                    (fullname.strip(), email.strip().lower(), password_hash),
                )
                return conn.insert_id()
    except pymysql.MySQLError as exc:
        logger.error("create_user failed: %s", exc)
        with open("db_debug.log", "a") as f:
            f.write(f"create_user failed: {exc}\n")
        return None


def get_user_by_email(email: str) -> dict | None:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM users WHERE email = %s LIMIT 1",
                    (email.strip().lower(),),
                )
                return cur.fetchone()
    except pymysql.MySQLError as exc:
        logger.error("get_user_by_email failed: %s", exc)
        return None


def get_user_by_id(user_id: int) -> dict | None:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE id = %s LIMIT 1", (user_id,))
                return cur.fetchone()
    except pymysql.MySQLError as exc:
        logger.error("get_user_by_id failed: %s", exc)
        return None


def update_last_login(user_id: int):
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET last_login = %s WHERE id = %s",
                    (datetime.utcnow(), user_id),
                )
    except pymysql.MySQLError as exc:
        logger.error("update_last_login failed: %s", exc)


def update_user_name(user_id: int, fullname: str) -> bool:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET fullname = %s WHERE id = %s",
                    (fullname.strip(), user_id),
                )
                return True
    except pymysql.MySQLError as exc:
        logger.error("update_user_name failed: %s", exc)
        return False


def update_user_password(user_id: int, new_hash: str) -> bool:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET password_hash = %s WHERE id = %s",
                    (new_hash, user_id),
                )
                return True
    except pymysql.MySQLError as exc:
        logger.error("update_user_password failed: %s", exc)
        return False


def update_confidence_threshold(user_id: int, threshold: float) -> bool:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET confidence_threshold = %s WHERE id = %s",
                    (float(threshold), user_id),
                )
                return True
    except pymysql.MySQLError as exc:
        logger.error("update_confidence_threshold failed: %s", exc)
        return False


def delete_user(user_id: int) -> bool:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
                return True
    except pymysql.MySQLError as exc:
        logger.error("delete_user failed: %s", exc)
        return False


def save_prediction(user_id, session_id: str, letter: str, confidence: float,
                    top5_json: str, letter_scores_json: str = None) -> int | None:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO predictions
                       (user_id, session_id, letter, confidence, top5_json, letter_scores_json)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (user_id, session_id, letter, confidence, top5_json, letter_scores_json),
                )
                return conn.insert_id()
    except pymysql.MySQLError as exc:
        logger.error("save_prediction failed: %s\n%s", exc, traceback.format_exc())
        return None


def get_history(user_id: int, page: int = 1, per_page: int = 20) -> dict:
    offset = (page - 1) * per_page
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) AS total FROM predictions WHERE user_id = %s",
                    (user_id,),
                )
                total = cur.fetchone()["total"]
                cur.execute(
                    """SELECT id, letter, confidence, top5_json, created_at
                       FROM predictions WHERE user_id = %s
                       ORDER BY created_at DESC LIMIT %s OFFSET %s""",
                    (user_id, per_page, offset),
                )
                rows = cur.fetchall()
                for row in rows:
                    if isinstance(row.get("created_at"), datetime):
                        row["created_at"] = row["created_at"].isoformat()
        return {
            "rows": rows,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": max(1, -(-total // per_page)),
        }
    except pymysql.MySQLError as exc:
        logger.error("get_history failed: %s", exc)
        return {"rows": [], "total": 0, "page": page, "per_page": per_page, "pages": 1}


def delete_all_predictions(user_id: int) -> bool:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM predictions WHERE user_id = %s", (user_id,))
                return True
    except pymysql.MySQLError as exc:
        logger.error("delete_all_predictions failed: %s", exc)
        return False


def delete_predictions_by_ids(user_id: int, ids: list) -> bool:
    if not ids:
        return True
    try:
        placeholders = ", ".join(["%s"] * len(ids))
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM predictions WHERE user_id = %s AND id IN ({placeholders})",
                    [user_id] + list(ids),
                )
                return True
    except pymysql.MySQLError as exc:
        logger.error("delete_predictions_by_ids failed: %s", exc)
        return False


def get_user_stats(user_id: int, date_range: str = "all") -> dict:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                where_extra = ""
                if date_range == "1d":
                    where_extra = " AND created_at >= NOW() - INTERVAL 1 DAY"
                elif date_range == "7d":
                    where_extra = " AND created_at >= NOW() - INTERVAL 7 DAY"
                elif date_range == "30d":
                    where_extra = " AND created_at >= NOW() - INTERVAL 30 DAY"

                cur.execute(
                    f"""SELECT COUNT(*) AS total,
                               AVG(confidence) AS avg_confidence,
                               COUNT(DISTINCT letter) AS unique_letters,
                               MAX(confidence) AS best_confidence
                        FROM predictions
                        WHERE user_id = %s{where_extra}""",
                    (user_id,),
                )
                row = cur.fetchone()

                cur.execute(
                    f"""SELECT letter, COUNT(*) AS cnt
                        FROM predictions WHERE user_id = %s{where_extra}
                        GROUP BY letter ORDER BY cnt DESC LIMIT 1""",
                    (user_id,),
                )
                top_row = cur.fetchone()

                cur.execute(
                    f"""SELECT letter, COUNT(*) AS cnt
                        FROM predictions WHERE user_id = %s{where_extra}
                        GROUP BY letter ORDER BY cnt DESC""",
                    (user_id,),
                )
                freq_rows = cur.fetchall()

                cur.execute(
                    f"""SELECT DATE(created_at) AS day, COUNT(*) AS cnt
                        FROM predictions WHERE user_id = %s{where_extra}
                        GROUP BY DATE(created_at) ORDER BY day DESC""",
                    (user_id,),
                )
                daily_rows = cur.fetchall()
                for r in daily_rows:
                    if hasattr(r["day"], "isoformat"):
                        r["day"] = r["day"].isoformat()

                cur.execute(
                    f"""SELECT DATE(created_at) AS day, AVG(confidence) AS avg_conf
                        FROM predictions WHERE user_id = %s{where_extra}
                        GROUP BY DATE(created_at) ORDER BY day ASC""",
                    (user_id,),
                )
                conf_over_time = cur.fetchall()
                for r in conf_over_time:
                    if hasattr(r["day"], "isoformat"):
                        r["day"] = r["day"].isoformat()

                cur.execute(
                    f"""SELECT HOUR(created_at) AS hr, COUNT(*) AS cnt
                        FROM predictions WHERE user_id = %s{where_extra}
                        GROUP BY HOUR(created_at) ORDER BY hr ASC""",
                    (user_id,),
                )
                hourly = cur.fetchall()

                cur.execute(
                    f"""SELECT letter, confidence, created_at
                        FROM predictions WHERE user_id = %s{where_extra}
                        ORDER BY confidence DESC LIMIT 1""",
                    (user_id,),
                )
                best_row = cur.fetchone()
                if best_row and isinstance(best_row.get("created_at"), datetime):
                    best_row["created_at"] = best_row["created_at"].isoformat()

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
    except pymysql.MySQLError as exc:
        logger.error("get_user_stats failed: %s", exc)
        return {}


def get_or_create_word_session(user_id: int) -> dict:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM sessions WHERE user_id = %s ORDER BY updated_at DESC LIMIT 1",
                    (user_id,),
                )
                row = cur.fetchone()
                if not row:
                    cur.execute(
                        "INSERT INTO sessions (user_id, sentence) VALUES (%s, %s)",
                        (user_id, ""),
                    )
                    sess_id = conn.insert_id()
                    return {"id": sess_id, "user_id": user_id, "sentence": ""}
                return row
    except pymysql.MySQLError as exc:
        logger.error("get_or_create_word_session failed: %s", exc)
        return {"id": None, "user_id": user_id, "sentence": ""}


def update_word_session(session_id: int, sentence: str) -> bool:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET sentence = %s WHERE id = %s",
                    (sentence, session_id),
                )
                return True
    except pymysql.MySQLError as exc:
        logger.error("update_word_session failed: %s", exc)
        return False


def get_sentence_history(user_id: int, limit: int = 20) -> list:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT sentence, created_at, updated_at FROM sessions
                       WHERE user_id = %s AND sentence != ''
                       ORDER BY updated_at DESC LIMIT %s""",
                    (user_id, limit),
                )
                rows = cur.fetchall()
                for r in rows:
                    for k in ("created_at", "updated_at"):
                        if isinstance(r.get(k), datetime):
                            r[k] = r[k].isoformat()
                return rows
    except pymysql.MySQLError as exc:
        logger.error("get_sentence_history failed: %s", exc)
        return []


def get_model_stats_global() -> dict:
    """Return aggregate stats across all users for admin model analytics."""
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
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
    except pymysql.MySQLError as exc:
        logger.error("get_model_stats_global failed: %s", exc)
        return {}
