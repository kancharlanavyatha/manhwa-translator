"""
Configuration Module for Webtoon Translator Server.

Handles runtime configuration, environment variables, and default hyperparameters.
Uses pure Python standard library for maximum portability and zero-DLL dependency issues.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Settings:
    """Application settings with environment variable override support."""

    # Server Network Settings
    host: str = field(default_factory=lambda: os.getenv("SERVER_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("SERVER_PORT", "8000")))
    cors_origins: List[str] = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(",")
    )

    # Translation Engine Settings
    translation_provider: str = field(
        default_factory=lambda: os.getenv("TRANSLATION_PROVIDER", "local_nmt").lower()
    )
    nmt_model_dir: str = field(
        default_factory=lambda: os.getenv("NMT_MODEL_DIR", "server/models/nmt")
    )
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    # Local OCR Settings
    ocr_provider: str = field(
        default_factory=lambda: os.getenv("OCR_PROVIDER", "rapidocr").lower()
    )
    ocr_rec_model_path: str = field(
        default_factory=lambda: os.getenv("OCR_REC_MODEL_PATH", "server/models/ocr/korean_rec.onnx")
    )
    ocr_dict_path: str = field(
        default_factory=lambda: os.getenv("OCR_DICT_PATH", "server/models/ocr/korean_dict.txt")
    )

    # Caching Layer Settings
    cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CACHE_DIR", ".cache"))
    )
    cache_enabled: bool = field(
        default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    max_cache_items: int = field(
        default_factory=lambda: int(os.getenv("MAX_CACHE_ITEMS", "2000"))
    )

    # Typesetting & Typography Defaults
    default_font_family: str = "sans-serif"
    min_font_size: int = 11
    max_font_size: int = 42
    bubble_padding_ratio: float = 0.15

    # Logging & Debugging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


settings = Settings()
