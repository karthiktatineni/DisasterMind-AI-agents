"""NVIDIA Build/NIM embedding service for query and document vectors."""

from __future__ import annotations

from openai import AsyncOpenAI

from backend.app.config import Settings, get_settings

EMBEDDING_DIM = 4096
INPUT_TYPE_QUERY = "query"
INPUT_TYPE_PASSAGE = "passage"


class EmbeddingConfigError(RuntimeError):
    pass


class EmbeddingService:
    def __init__(self, settings: Settings | None = None, client: AsyncOpenAI | None = None):
        self.settings = settings or get_settings()
        if self.settings.nvidia_embedding_dimension != EMBEDDING_DIM:
            raise EmbeddingConfigError(
                f"NVIDIA_EMBEDDING_DIMENSION must be {EMBEDDING_DIM} for nv-embed-v1."
            )
        if client is not None:
            self._client = client
            return
        if not self.settings.nvidia_api_key:
            raise EmbeddingConfigError("NVIDIA_API_KEY is missing.")
        self._client = AsyncOpenAI(
            api_key=self.settings.nvidia_api_key,
            base_url=self.settings.nvidia_base_url,
            timeout=120,
            max_retries=2,
        )

    @property
    def provider(self) -> str:
        return "nvidia-build"

    @property
    def model(self) -> str:
        return self.settings.nvidia_embedding_model

    async def embed_text(self, text: str, *, input_type: str = INPUT_TYPE_QUERY) -> list[float]:
        response = await self._client.embeddings.create(
            model=self.model,
            input=text,
            extra_body={
                "input_type": input_type,
                "encoding_format": "float",
                "truncate": "END",
            },
        )
        vector = response.data[0].embedding
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(f"Expected embedding dim {EMBEDDING_DIM}, got {len(vector)}")
        return vector

    @staticmethod
    def build_scenario_query(scenario: dict) -> str:
        parts = [
            f"Disaster Type: {scenario.get('disaster_type', 'unknown')}",
            f"Region: {scenario.get('region', 'unknown')}",
            f"Country: {scenario.get('country', 'unknown')}",
            f"Severity: {scenario.get('severity', 'unknown')}",
            f"Wind speed: {scenario.get('wind_speed', 'unknown')} km/h",
            f"Rainfall: {scenario.get('rainfall', 'unknown')} mm",
            f"Population: {scenario.get('population', 'unknown')}",
            f"Elevation: {scenario.get('elevation', 'unknown')} m",
            f"Notes: {scenario.get('notes', '')}",
        ]
        return "\n".join(parts)
