"""All settings in one place, read from environment variables (.env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    gemini_model: str
    temperature: float
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_upload_mb: int
    data_dir: Path

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "faiss_index"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "documents.json"


settings = Settings(
    google_api_key=os.getenv("GOOGLE_API_KEY") or None,
    gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    # Low temperature = focused, consistent answers. We want facts from the document, not creativity.
    temperature=_get_float("GEMINI_TEMPERATURE", 0.2),
    embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    chunk_size=_get_int("CHUNK_SIZE", 1000),
    chunk_overlap=_get_int("CHUNK_OVERLAP", 200),
    top_k=_get_int("TOP_K", 4),
    max_upload_mb=_get_int("MAX_UPLOAD_MB", 20),
    data_dir=Path(os.getenv("DATA_DIR", BASE_DIR / "data")),
)
