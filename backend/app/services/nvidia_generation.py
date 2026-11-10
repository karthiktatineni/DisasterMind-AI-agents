from __future__ import annotations

import asyncio
import random

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI

from backend.app.config import Settings, get_settings
from backend.app.schemas import GenerationMessage, GenerationResponse
from backend.app.services.retry_policy import retry_after_seconds


class NvidiaGenerationConfigError(RuntimeError):
    pass


class NvidiaGenerationError(RuntimeError):
    pass


class NvidiaBuildGenerationClient:
    def __init__(self, settings: Settings | None = None, client: AsyncOpenAI | None = None):
        self.settings = settings or get_settings()
        if client is not None:
            self._client = client
            return

        if not self.settings.nvidia_api_key:
            raise NvidiaGenerationConfigError(
                "NVIDIA_API_KEY is missing. Add it to the project .env file."
            )

        self._client = AsyncOpenAI(
            api_key=self.settings.nvidia_api_key,
            base_url=self.settings.nvidia_base_url,
            timeout=60,
            max_retries=0,
        )

    async def generate(
        self,
        messages: list[GenerationMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> GenerationResponse:
        last_exc: Exception | None = None
        for attempt in range(3):
            retry_after_s: float | None = None
            try:
                completion = await self._client.chat.completions.create(
                    model=self.settings.nvidia_model,
                    messages=[message.model_dump() for message in messages],
                    max_tokens=max_tokens or self.settings.nvidia_max_tokens,
                    temperature=(
                        self.settings.nvidia_temperature
                        if temperature is None
                        else temperature
                    ),
                    top_p=self.settings.nvidia_top_p if top_p is None else top_p,
                    stream=False,
                )
                break
            except APIStatusError as exc:
                last_exc = exc
                if exc.status_code == 429:
                    retry_after_s = retry_after_seconds(exc.response.headers)
                if exc.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise NvidiaGenerationError(f"NVIDIA generation failed: {exc}") from exc
            except (APIConnectionError, APITimeoutError, APIError) as exc:
                last_exc = exc
                if attempt == 2:
                    raise NvidiaGenerationError(f"NVIDIA generation failed: {exc}") from exc
            backoff = 0.2 * (2**attempt) + random.uniform(0.0, 0.2)
            await asyncio.sleep(max(backoff, retry_after_s or 0.0))
        else:
            raise NvidiaGenerationError(f"NVIDIA generation failed: {last_exc}")

        message = completion.choices[0].message
        reasoning_content = getattr(message, "reasoning_content", None)

        return GenerationResponse(
            provider="nvidia-build",
            model=completion.model or self.settings.nvidia_model,
            content=message.content or "",
            reasoning_content=reasoning_content,
            request_id=completion.id,
        )
