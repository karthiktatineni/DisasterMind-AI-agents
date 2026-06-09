from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.app.config import Settings
from backend.app.schemas import GenerationMessage
from backend.app.services.nvidia_generation import (
    NvidiaBuildGenerationClient,
    NvidiaGenerationConfigError,
)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="chatcmpl-test",
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Generated response",
                        reasoning_content="Brief reasoning trace",
                    )
                )
            ],
        )


class FakeClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.mark.asyncio
async def test_generate_uses_nvidia_chat_completion_contract():
    fake_client = FakeClient()
    settings = Settings(
        nvidia_api_key="test-key",
        nvidia_model="nvidia/llama-3.3-nemotron-super-49b-v1",
        nvidia_max_tokens=512,
        nvidia_temperature=0.1,
        nvidia_top_p=0.9,
    )
    client = NvidiaBuildGenerationClient(settings=settings, client=fake_client)

    result = await client.generate(
        messages=[GenerationMessage(role="user", content="Plan evacuation.")],
    )

    call = fake_client.completions.calls[0]
    assert call["model"] == "nvidia/llama-3.3-nemotron-super-49b-v1"
    assert call["messages"] == [{"role": "user", "content": "Plan evacuation."}]
    assert call["max_tokens"] == 512
    assert call["temperature"] == 0.1
    assert call["top_p"] == 0.9
    assert call["stream"] is False
    assert result.provider == "nvidia-build"
    assert result.content == "Generated response"
    assert result.reasoning_content == "Brief reasoning trace"


def test_missing_api_key_requires_env_configuration():
    settings = Settings(nvidia_api_key="")

    with pytest.raises(NvidiaGenerationConfigError):
        NvidiaBuildGenerationClient(settings=settings)

