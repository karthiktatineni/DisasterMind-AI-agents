"""NVIDIA Build/NIM embedding service for query and document vectors."""

from __future__ import annotations

import asyncio
import random
from typing import Sequence

import tiktoken
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from backend.app.config import Settings, get_settings

DEFAULT_EMBEDDING_DIM = 2048
EMBEDDING_DIM = DEFAULT_EMBEDDING_DIM
INPUT_TYPE_QUERY = "query"
INPUT_TYPE_PASSAGE = "passage"
MODEL_CONTEXT_TOKEN_LIMIT = 4096
SAFE_CLIENT_TOKEN_LIMIT = 3800
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens_approx(text: str) -> int:
    """Count tokens using cl100k_base (calibrated within ~5% of live NVIDIA NIM usage.prompt_tokens)."""
    return len(_TOKENIZER.encode(text))


def _truncate_to_token_limit(text: str, max_tokens: int) -> str:
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _TOKENIZER.decode(tokens[:max_tokens])


class EmbeddingConfigError(RuntimeError):
    pass


class EmbeddingService:
    def __init__(self, settings: Settings | None = None, client: AsyncOpenAI | None = None):
        self.settings = settings or get_settings()
        self.dimension = self.settings.nvidia_embedding_dimension or DEFAULT_EMBEDDING_DIM
        if client is not None:
            self._client = client
            return
        if not self.settings.nvidia_api_key:
            raise EmbeddingConfigError("NVIDIA_API_KEY is missing.")
        self._client = AsyncOpenAI(
            api_key=self.settings.nvidia_api_key,
            base_url=self.settings.nvidia_base_url,
            timeout=120,
            max_retries=0,
        )

    @property
    def provider(self) -> str:
        return "nvidia-build"

    @property
    def model(self) -> str:
        return self.settings.nvidia_embedding_model

    async def _request_with_retry(
        self,
        payload_input: str | list[str],
        *,
        input_type: str,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ):
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await self._client.embeddings.create(
                    model=self.model,
                    input=payload_input,
                    extra_body={
                        "input_type": input_type,
                        "encoding_format": "float",
                        "truncate": "END",
                    },
                )
            except APIStatusError as exc:
                last_exc = exc
                if exc.status_code not in (429, 500, 502, 503, 504) or attempt == max_retries - 1:
                    raise
            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                if attempt == max_retries - 1:
                    raise
            delay = base_delay * (2**attempt) + random.uniform(0.0, 0.5)
            await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def embed_text(self, text: str, *, input_type: str) -> list[float]:
        if input_type not in (INPUT_TYPE_QUERY, INPUT_TYPE_PASSAGE):
            raise ValueError(f"input_type must be '{INPUT_TYPE_QUERY}' or '{INPUT_TYPE_PASSAGE}', got '{input_type}'")
        safe_text = _truncate_to_token_limit(text, SAFE_CLIENT_TOKEN_LIMIT)
        response = await self._request_with_retry(safe_text, input_type=input_type)
        vector = response.data[0].embedding
        if len(vector) != self.dimension:
            raise ValueError(f"Expected embedding dim {self.dimension}, got {len(vector)}")
        return vector

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> list[list[float]]:
        if input_type not in (INPUT_TYPE_QUERY, INPUT_TYPE_PASSAGE):
            raise ValueError(f"input_type must be '{INPUT_TYPE_QUERY}' or '{INPUT_TYPE_PASSAGE}', got '{input_type}'")
        if not texts:
            return []
        safe_texts = [_truncate_to_token_limit(t, SAFE_CLIENT_TOKEN_LIMIT) for t in texts]
        response = await self._request_with_retry(
            safe_texts,
            input_type=input_type,
            max_retries=max_retries,
            base_delay=base_delay,
        )
        vectors = [item.embedding for item in response.data]
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(f"Expected embedding dim {self.dimension}, got {len(vector)}")
        return vectors

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
