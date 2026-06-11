from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    nvidia_api_key: str
    nvidia_embedding_model: str = "nvidia/nv-embed-v1"
    nvidia_embedding_dimension: int = 4096
    similarity_threshold: float = 0.2
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1"
    nvidia_max_tokens: int = 1024
    nvidia_temperature: float = 0.2
    nvidia_top_p: float = 0.95
    app_env: str = "development"
    disastermind_api_key: str = ""
    allowed_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    )
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    max_request_bytes: int = 65536
    enable_backend_keepalive: bool = True
    backend_keepalive_url: str = ""
    backend_keepalive_interval_seconds: int = 300
    enable_supabase_keepalive: bool = True
    supabase_keepalive_interval_seconds: int = 86400

    @property
    def production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def api_key_required(self) -> bool:
        return self.production or bool(self.disastermind_api_key)


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    return int(raw_value)


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    return float(raw_value)


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_origins() -> tuple[str, ...]:
    raw_value = os.getenv("ALLOWED_ORIGINS", "")
    if not raw_value.strip():
        return (
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        )
    return tuple(origin.strip() for origin in raw_value.split(",") if origin.strip())


def get_settings() -> Settings:
    backend_keepalive_url = (
        os.getenv("BACKEND_KEEPALIVE_URL")
        or os.getenv("RENDER_EXTERNAL_URL")
        or ""
    ).strip()
    return Settings(
        nvidia_api_key=os.getenv("NVIDIA_API_KEY", "").strip(),
        nvidia_embedding_model=os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embed-v1").strip(),
        nvidia_embedding_dimension=_get_int("NVIDIA_EMBEDDING_DIMENSION", 4096),
        similarity_threshold=_get_float("SIMILARITY_THRESHOLD", 0.2),
        supabase_url=os.getenv("SUPABASE_URL", "").strip(),
        supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip(),
        nvidia_base_url=os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ).strip(),
        nvidia_model=os.getenv(
            "NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1"
        ).strip(),
        nvidia_max_tokens=_get_int("NVIDIA_MAX_TOKENS", 1024),
        nvidia_temperature=_get_float("NVIDIA_TEMPERATURE", 0.2),
        nvidia_top_p=_get_float("NVIDIA_TOP_P", 0.95),
        app_env=os.getenv("APP_ENV", "development").strip() or "development",
        disastermind_api_key=os.getenv("DISASTERMIND_API_KEY", "").strip(),
        allowed_origins=_get_origins(),
        rate_limit_requests=_get_int("RATE_LIMIT_REQUESTS", 60),
        rate_limit_window_seconds=_get_int("RATE_LIMIT_WINDOW_SECONDS", 60),
        max_request_bytes=_get_int("MAX_REQUEST_BYTES", 65536),
        enable_backend_keepalive=_get_bool("ENABLE_BACKEND_KEEPALIVE", True),
        backend_keepalive_url=backend_keepalive_url.rstrip("/"),
        backend_keepalive_interval_seconds=_get_int("BACKEND_KEEPALIVE_INTERVAL_SECONDS", 300),
        enable_supabase_keepalive=_get_bool("ENABLE_SUPABASE_KEEPALIVE", True),
        supabase_keepalive_interval_seconds=_get_int("SUPABASE_KEEPALIVE_INTERVAL_SECONDS", 86400),
    )
