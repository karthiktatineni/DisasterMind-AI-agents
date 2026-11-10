"""Embedding service supporting Pinecone Inference and NVIDIA Build/NIM models."""

from __future__ import annotations

import asyncio
import random
from typing import Any, Sequence

import tiktoken
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from pinecone import AsyncPinecone

from backend.app.config import Settings, get_settings
from backend.app.services.retry_policy import retry_after_seconds

DEFAULT_EMBEDDING_DIM = 1024
EMBEDDING_DIM = DEFAULT_EMBEDDING_DIM
INPUT_TYPE_QUERY = "query"
INPUT_TYPE_PASSAGE = "passage"
MODEL_CONTEXT_TOKEN_LIMIT = 4096
SAFE_CLIENT_TOKEN_LIMIT = 3800
PINECONE_INFERENCE_MODELS = frozenset({"llama-text-embed-v2", "multilingual-e5-large"})
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
    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
        pinecone_client: AsyncPinecone | Any | None = None,
    ):
        self.settings = settings or get_settings()
        self.dimension = self.settings.embedding_dimension or DEFAULT_EMBEDDING_DIM
        self._client: AsyncOpenAI | None = None
        self._pinecone_client: AsyncPinecone | Any | None = None
        self._owns_pinecone = False

        if client is not None:
            self._backend = "openai"
            self._client = client
            return

        use_pinecone = (
            pinecone_client is not None
            or self.model in PINECONE_INFERENCE_MODELS
            or self.settings.embedding_provider.lower() == "pinecone"
        )
        if use_pinecone:
            self._backend = "pinecone"
            if pinecone_client is not None:
                self._pinecone_client = pinecone_client
            else:
                if not self.settings.pinecone_api_key:
                    raise EmbeddingConfigError("PINECONE_API_KEY is missing. Add it to .env for Pinecone inference.")
                self._pinecone_client = AsyncPinecone(api_key=self.settings.pinecone_api_key, timeout=60.0)
                self._owns_pinecone = True
            return

        self._backend = "openai"
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
        return "pinecone-inference" if self._backend == "pinecone" else "nvidia-build"

    @property
    def model(self) -> str:
        return self.settings.embedding_model

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            await self._client.close()
        if self._owns_pinecone and self._pinecone_client is not None and hasattr(self._pinecone_client, "close"):
            await self._pinecone_client.close()
            self._owns_pinecone = False

    async def _request_with_retry(
        self,
        payload_input: str | list[str],
        *,
        input_type: str,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            retry_after_s: float | None = None
            try:
                if self._backend == "pinecone":
                    assert self._pinecone_client is not None
                    inputs_list = [payload_input] if isinstance(payload_input, str) else list(payload_input)
                    resp = await self._pinecone_client.inference.embed(
                        model=self.model,
                        inputs=inputs_list,
                        parameters={"input_type": input_type, "truncate": "END"},
                    )
                    return [list(item.values) for item in resp.data]
                assert self._client is not None
                resp = await self._client.embeddings.create(
                    model=self.model,
                    input=payload_input,
                    extra_body={
                        "input_type": input_type,
                        "encoding_format": "float",
                        "truncate": "END",
                    },
                )
                return [list(item.embedding) for item in resp.data]
            except APIStatusError as exc:
                last_exc = exc
                if self._backend == "openai" and exc.status_code == 429:
                    retry_after_s = retry_after_seconds(exc.response.headers)
                if exc.status_code not in (429, 500, 502, 503, 504) or attempt == max_retries - 1:
                    raise
            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                if attempt == max_retries - 1:
                    raise
            except Exception as exc:
                last_exc = exc
                if self._backend != "pinecone" or attempt == max_retries - 1:
                    raise
            delay = max(base_delay * (2**attempt) + random.uniform(0.0, 0.5), retry_after_s or 0.0)
            await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def embed_text(self, text: str, *, input_type: str) -> list[float]:
        if input_type not in (INPUT_TYPE_QUERY, INPUT_TYPE_PASSAGE):
            raise ValueError(f"input_type must be '{INPUT_TYPE_QUERY}' or '{INPUT_TYPE_PASSAGE}', got '{input_type}'")
        safe_text = _truncate_to_token_limit(text, SAFE_CLIENT_TOKEN_LIMIT)
        vectors = await self._request_with_retry(safe_text, input_type=input_type)
        vector = vectors[0]
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
        vectors = await self._request_with_retry(
            safe_texts,
            input_type=input_type,
            max_retries=max_retries,
            base_delay=base_delay,
        )
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
