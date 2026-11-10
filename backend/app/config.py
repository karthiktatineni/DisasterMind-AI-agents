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
    embedding_provider: str = "pinecone"
    embedding_model: str = "llama-text-embed-v2"
    embedding_dimension: int = 1024
    similarity_threshold: float = 0.3112
    pinecone_api_key: str = ""
    pinecone_index_name: str = "disastermind"
    pinecone_environment: str = ""
    pinecone_namespace: str = "v2-3d35e5131d79a79d-llama-text-embed-v2"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
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
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "pinecone").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "llama-text-embed-v2").strip(),
        embedding_dimension=_get_int("EMBEDDING_DIMENSION", 1024),
        similarity_threshold=_get_float("SIMILARITY_THRESHOLD", 0.3112),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip(),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "disastermind").strip(),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", "").strip(),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "v2-3d35e5131d79a79d-llama-text-embed-v2").strip(),
        nvidia_base_url=os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ).strip(),
        nvidia_model=os.getenv(
            "NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"
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
    )
