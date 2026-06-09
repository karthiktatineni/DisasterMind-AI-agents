from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.schemas import GenerationMessage
from backend.app.services.nvidia_generation import (
    NvidiaBuildGenerationClient,
    NvidiaGenerationConfigError,
    NvidiaGenerationError,
)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live NVIDIA Build API smoke test.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Create a concise disaster response planning summary for a cyclone.",
    )
    args = parser.parse_args()

    try:
        client = NvidiaBuildGenerationClient()
        response = await client.generate(
            messages=[
                GenerationMessage(
                    role="system",
                    content="You are the DisasterMind AI generation service.",
                ),
                GenerationMessage(role="user", content=args.prompt),
            ],
            max_tokens=256,
        )
    except NvidiaGenerationConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except NvidiaGenerationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(response.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
