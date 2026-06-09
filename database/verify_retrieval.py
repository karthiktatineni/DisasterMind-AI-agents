#!/usr/bin/env python3
"""Verify pgvector retrieval with a real test query."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from backend.app.services.similarity_service import SimilarityService  # noqa: E402

TEST_QUERY = "Cyclone India wind speed 180 km/h coastal flooding"


async def main() -> None:
    service = SimilarityService()
    result = await service.search_text(TEST_QUERY, top_k=10)

    lines = ["# Retrieval Validation\n", f"Query: `{TEST_QUERY}`\n", f"Status: {result.get('status')}\n"]
    if result.get("status") == "ok":
        lines.append("## Top Matches\n")
        lines.append("| Event | Country | Year | Similarity | ID |\n|---|---|---|---|---|\n")
        for match in result.get("matches", []):
            lines.append(
                f"| {match.get('event_name')} | {match.get('country')} | "
                f"{match.get('start_year', '-')} | {match.get('similarity')} | {match.get('id')} |\n"
            )
    else:
        lines.append(f"\nReason: {result.get('reason')}\n")

    output = PROJECT_ROOT / "retrieval_validation.md"
    output.write_text("".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
