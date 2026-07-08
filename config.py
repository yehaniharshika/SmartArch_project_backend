"""
SmartArch — config.py
ALL settings come from here. Every other file imports Config
and reads values — nothing reads .env directly anywhere else.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Paths 
    BASE_DIR        = Path(__file__).parent
    UPLOAD_DIR      = BASE_DIR / os.getenv("UPLOAD_FOLDER",   "uploads")
    VECTORSTORE_DIR = BASE_DIR / os.getenv("VECTORSTORE_DIR", "vectorstore")
    WEIGHTS_PATH    = BASE_DIR / os.getenv("YOLO_WEIGHTS",    "dl/weights/best.pt")

    UPLOAD_DIR.mkdir(exist_ok=True)
    VECTORSTORE_DIR.mkdir(exist_ok=True)

    # Database 
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///smartarch.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    SQLALCHEMY_ENGINE_OPTIONS = {
        "connect_args": {
            "timeout": 15,          # seconds to wait for a lock before erroring
            "check_same_thread": False,
        }
    }

    # Flask 
    FLASK_PORT       = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_ENV        = os.getenv("FLASK_ENV", "development")
    MAX_UPLOAD_MB    = int(os.getenv("MAX_CONTENT_LENGTH_MB", "50"))
    ALLOWED_EXT      = {"png", "jpg", "jpeg", "pdf"}
    FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")

    # ── YOLOv8 ───────────────────────────────────────────────
    YOLO_CONF     = float(os.getenv("YOLO_CONF_THRESHOLD", "0.35"))
    YOLO_IOU      = float(os.getenv("YOLO_IOU_THRESHOLD",  "0.45"))
    YOLO_IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE",         "1280"))

    # ── OCR & PDF ────────────────────────────────────────────
    OCR_LANG      = ["en"]
    OCR_GPU       = os.getenv("OCR_GPU", "false").lower() == "true"
    DEFAULT_SCALE = float(os.getenv("DEFAULT_SCALE_M_PER_PX", "0.01"))
    PDF_DPI       = int(os.getenv("PDF_DPI", "200"))

    # ── AI Provider Switch ──────────────────────────────────
    # "gemini" now (free). Change to "openai" later by editing
    # ONLY this one .env value — no code changes needed anywhere.
    AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()

    # Gemini settings (active now)
    GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")

    # OpenAI settings (ready for later)
    OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # ── ChromaDB (for the chat/RAG step later) ──────────────
    CHROMA_PATH = str(VECTORSTORE_DIR)

    # ── JWT ──────────────────────────────────────────────────
    JWT_SECRET      = os.getenv("JWT_SECRET", "smartarch-dev-secret-CHANGE-ME")
    JWT_ALGORITHM   = "HS256"
    JWT_EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "30"))

    # ── Detection class metadata ─────────────────────────────
    CLASS_META = {
        "door":        {"emoji": "🚪", "color": "#EF5350", "type": "opening"},
        "window":      {"emoji": "🪟", "color": "#42A5F5", "type": "opening"},
        "wall":        {"emoji": "🧱", "color": "#78909C", "type": "structure"},
        "room":        {"emoji": "🏠", "color": "#FFA726", "type": "space"},
        "bedroom":     {"emoji": "🛏️",  "color": "#AB47BC", "type": "space"},
        "bathroom":    {"emoji": "🛁", "color": "#26C6DA", "type": "space"},
        "kitchen":     {"emoji": "🍳", "color": "#FF7043", "type": "space"},
        "living room": {"emoji": "🛋️",  "color": "#66BB6A", "type": "space"},
        "dining":      {"emoji": "🍽️",  "color": "#FFCA28", "type": "space"},
        "garage":      {"emoji": "🚗", "color": "#8D6E63", "type": "space"},
        "corridor":    {"emoji": "🚶", "color": "#EC407A", "type": "space"},
        "balcony":     {"emoji": "🌿", "color": "#9CCC65", "type": "space"},
        "staircase":   {"emoji": "🪜", "color": "#90A4AE", "type": "space"},
    }

    SPACE_CLASSES   = {k for k, v in CLASS_META.items() if v["type"] == "space"}
    OPENING_CLASSES = {k for k, v in CLASS_META.items() if v["type"] == "opening"}