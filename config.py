from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env once at import time. Environment variables already set in shell keep priority.
load_dotenv(override=False)

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2-preview"
DEFAULT_DUPLICATE_THRESHOLD = 0.8


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def get_openai_api_key() -> Optional[str]:
    return _clean(os.getenv("OPENAI_API_KEY"))


def get_base_url() -> Optional[str]:
    # Support both BASE_URL and OPENAI_BASE_URL aliases.
    return _clean(os.getenv("OPENAI_BASE_URL")) or _clean(os.getenv("BASE_URL"))


def get_google_api_key() -> Optional[str]:
    return _clean(os.getenv("GOOGLE_API_KEY"))


def get_embedding_model() -> str:
    return _clean(os.getenv("EMBEDDING_MODEL")) or DEFAULT_EMBEDDING_MODEL


def get_duplicate_threshold() -> float:
    raw = _clean(os.getenv("DUPLICATE_THRESHOLD"))
    if raw is None:
        return DEFAULT_DUPLICATE_THRESHOLD
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_DUPLICATE_THRESHOLD
    if val <= 0:
        return DEFAULT_DUPLICATE_THRESHOLD
    if val > 1:
        return 1.0
    return val


def create_openai_client() -> OpenAI:
    """
    Build an OpenAI client with explicit api_key/base_url from environment.
    This ensures custom BASE_URL from .env is honored across the project.
    """
    kwargs = {}
    api_key = get_openai_api_key()
    base_url = get_base_url()

    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)
