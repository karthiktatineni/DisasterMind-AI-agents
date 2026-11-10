"""Untrusted input & retrieved-metadata sanitization and delimiting guards (Task 6)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

MAX_NOTES_CHARS = 500
MAX_QUESTION_CHARS = 500
MAX_FIELD_CHARS = 240

_ASCII_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]+")
_BREAKOUT_TAG_RE = re.compile(
    r"<\s*/?\s*(?:user_notes|untrusted_retrieved_metadata|system|assistant|developer|instructions)\b[^>]*>"
    r"|<\|\s*(?:im_start|im_end|system|user|assistant)\s*\|>"
    r"|\[\s*/?\s*SYSTEM\s*\]",
    re.IGNORECASE,
)


def sanitize_untrusted_text(text: Any, *, max_len: int | None = MAX_FIELD_CHARS) -> tuple[str, bool]:
    """Strip control/bidi-format characters, neutralize prompt-delimiter breakouts, and cap length."""
    if text is None:
        return "", False
    raw = str(text)
    cleaned = _ASCII_CTRL_RE.sub(" ", raw)
    cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) != "Cf")
    cleaned = cleaned.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = _BREAKOUT_TAG_RE.sub("[filtered_delimiter]", cleaned)
    cleaned = " ".join(cleaned.split()).strip()
    truncated = False
    if max_len is not None and len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
        truncated = True
    return cleaned, truncated


def delimit_user_notes(raw_notes: str | None) -> tuple[str, bool]:
    """Sanitize and wrap user notes (<= 500 chars) inside strict untrusted delimiters."""
    clean, truncated = sanitize_untrusted_text(raw_notes or "", max_len=MAX_NOTES_CHARS)
    return f"<user_notes>{clean}</user_notes>", truncated
