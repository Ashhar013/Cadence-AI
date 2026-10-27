"""Centralized configuration loader for Cadence AI."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
_project_root = Path(__file__).resolve().parents[3]  # src/cadence/utils -> project root
load_dotenv(_project_root / ".env")


# ─── LLM ────────────────────────────────────────────────────────────────────

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ─── Storage ────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/processed/chroma")
POSTGRES_URI = os.getenv("POSTGRES_URI", "")

# ─── Audio ──────────────────────────────────────────────────────────────────

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# ─── Task Queue ─────────────────────────────────────────────────────────────

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# ─── Paths ──────────────────────────────────────────────────────────────────

DATA_RAW_DIR = _project_root / "data" / "raw"
DATA_PROCESSED_DIR = _project_root / "data" / "processed"
