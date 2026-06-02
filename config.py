import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv(override=True)


class BaseConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")
    PERMANENT_SESSION_LIFETIME = timedelta(
        seconds=int(os.environ.get("SESSION_LIFETIME_SECONDS", "86400"))
    )
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", str(16 * 1024 * 1024)))

    DB_USER = os.environ.get("DB_USER", "root")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "root")
    DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
    DB_PORT = int(os.environ.get("DB_PORT", "8889"))
    DB_NAME = os.environ.get("DB_NAME", "signai")

    MODEL_PATH = os.environ.get("MODEL_PATH", "models/sign_language_model.h5")
    CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
    FRAME_RATE = int(os.environ.get("FRAME_RATE", "10"))

    RATELIMIT_DEFAULT = "200 per day;50 per hour"
    RATELIMIT_STORAGE_URI = os.environ.get("REDIS_URL", "memory://")


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False


class ProductionConfig(BaseConfig):
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True

    def __init__(self):
        secret = os.environ.get("SECRET_KEY")
        if not secret or secret == "dev-secret-change-in-prod":
            raise RuntimeError(
                "ProductionConfig requires SECRET_KEY to be set via environment variable "
                "to a strong random value. Never use the default dev key in production."
            )
class TestingConfig(BaseConfig):
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False
    SESSION_COOKIE_SECURE = False


CONFIG_MAP = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config():
    env = os.environ.get("FLASK_ENV", "development").lower()
    cls = CONFIG_MAP.get(env, DevelopmentConfig)
    if cls is ProductionConfig:
        obj = cls()
        return obj
    return cls()
