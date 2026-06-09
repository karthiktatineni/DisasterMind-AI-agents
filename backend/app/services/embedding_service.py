"""OpenAI embedding service for query and document vectors."""

from __future__ import annotations

from openai import AsyncOpenAI

from backend.app.config import Settings, get_settings

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class EmbeddingConfigError(RuntimeError):
    pass


class EmbeddingService:
    def __init__(self, settings: Settings | None = None, client: AsyncOpenAI | None = None):
        self.settings = settings or get_settings()
        if client is not None:
            self._client = client
            return
        if not self.settings.openai_api_key:
            raise EmbeddingConfigError("OPENAI_API_KEY is missing.")
        self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)

    async def embed_text(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(model=EMBEDDING_MODEL, input=text)
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
