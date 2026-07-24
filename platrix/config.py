"""Runtime configuration.

All settings can be provided through environment variables (optionally via a
``.env`` file) using the ``PLATRIX_`` prefix, e.g. ``PLATRIX_DETECTOR=yolo``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central application settings."""

    model_config = SettingsConfigDict(
        env_prefix="PLATRIX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General ---------------------------------------------------------
    app_name: str = "Platrix"
    debug: bool = False
    log_level: str = "INFO"

    # --- Storage ---------------------------------------------------------
    data_dir: Path = BASE_DIR / "data"
    database_url: str = ""  # derived from data_dir if empty
    snapshots_dir: Path = BASE_DIR / "data" / "snapshots"

    # --- Detection -------------------------------------------------------
    detector: str = "auto"  # "auto" | "contour" | "yolo"
    yolo_weights: Path = BASE_DIR / "models" / "plate_yolo.pt"
    detection_confidence: float = 0.35
    min_plate_area: int = 1000
    plate_aspect_min: float = 2.2
    plate_aspect_max: float = 5.0

    # --- OCR -------------------------------------------------------------
    ocr: str = "auto"  # "auto" | "crnn" | "onnx" | "cnn" | "none"
    crnn_weights: Path = BASE_DIR / "models" / "ocr_crnn.onnx"  # segmentation-free
    onnx_weights: Path = BASE_DIR / "models" / "ocr_cnn.onnx"  # per-char classifier
    ocr_weights: Path = BASE_DIR / "models" / "ocr_cnn.h5"  # keras backend
    ocr_min_confidence: float = 0.40

    # --- Pipeline / streaming -------------------------------------------
    frame_stride: int = 1  # process every Nth frame (>=1)
    max_fps: float = 30.0  # throttle source read rate
    dedupe_seconds: float = 4.0  # suppress identical plate re-logs within window
    jpeg_quality: int = 80

    # --- Server ----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Authentication --------------------------------------------------
    auth_enabled: bool = True
    auth_user: str = "admin"
    auth_password: str = "admin"  # CHANGE THIS via PLATRIX_AUTH_PASSWORD
    secret_key: str = "change-me-please-set-PLATRIX_SECRET_KEY"

    # --- Default camera source (used by the server on startup, optional) -
    default_source: str = ""  # e.g. "0", "rtsp://...", "/path/to/video.mp4"

    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(self.data_dir / 'platrix.db').as_posix()}"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
