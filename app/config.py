import os
from pathlib import Path

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GIGACHAT_CREDENTIALS = os.environ.get("GIGACHAT_CREDENTIALS", "")
GIGACHAT_MODEL = os.environ.get("GIGACHAT_MODEL", "GigaChat-2-Max")
LANGUAGE = os.environ.get("LANGUAGE", "ru")
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "10"))
CAT_THRESHOLD = int(os.environ.get("CAT_THRESHOLD", "50"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
SCHEMA_CACHE_TTL_SECONDS = int(os.environ.get("SCHEMA_CACHE_TTL_SECONDS", str(7 * 24 * 60 * 60)))
DIALOG_TTL_SECONDS = int(os.environ.get("DIALOG_TTL_SECONDS", str(24 * 60 * 60)))
DIALOG_HISTORY_LIMIT = int(os.environ.get("DIALOG_HISTORY_LIMIT", "10"))
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", "/data/filter-platform"))
CLIP_MODEL = os.environ.get("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.environ.get("CLIP_PRETRAINED", "laion2b_s34b_b79k")
HYBRID_ALPHA = float(os.environ.get("HYBRID_ALPHA", "0.5"))
CLIP_BUILD_ON_UPLOAD_DEFAULT = os.environ.get("CLIP_BUILD_ON_UPLOAD_DEFAULT", "false").lower() in {"1", "true", "yes", "on"}
