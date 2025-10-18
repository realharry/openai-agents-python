"""Helpers for examples that want to talk to a local Ollama instance.

This centralizes env-var handling and small smoke-checks so example files don't
have duplicated code.
"""
from __future__ import annotations

import os
from typing import Optional

from openai import AsyncOpenAI

from agents import OpenAIChatCompletionsModel, set_default_openai_client


# Defaults used by the examples. Override with environment variables.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")


def make_ollama_client(url: Optional[str] = None, api_key: Optional[str] = None) -> AsyncOpenAI:
    """Create an AsyncOpenAI client configured to talk to an Ollama HTTP endpoint.

    Args:
        url: Base URL for the Ollama server (example: http://localhost:11434/v1).
        api_key: API key to send (Ollama often accepts `ollama` as a local sentinel).
    """
    return AsyncOpenAI(base_url=url or OLLAMA_URL, api_key=api_key or OLLAMA_API_KEY)


def make_chat_model(model_name: str, client: Optional[AsyncOpenAI] = None) -> OpenAIChatCompletionsModel:
    """Convenience factory for creating an OpenAIChatCompletionsModel that uses Ollama."""
    client = client or make_ollama_client()
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


async def smoke_check_models(client: Optional[AsyncOpenAI] = None) -> None:
    """Try to fetch `/models` from Ollama using a raw HTTP request and print
    available names (best-effort).

    This avoids depending on SDK internals or method signatures.
    """
    try:
        import httpx

        base = OLLAMA_URL
        # If an AsyncOpenAI client was provided, try to read a base_url attribute.
        if client is not None:
            base = getattr(client, "base_url", base)

        url = base.rstrip("/") + "/models"
        async with httpx.AsyncClient(timeout=5.0) as http:
            resp = await http.get(url)
            resp.raise_for_status()
            data = resp.json()

        names = [m.get("name") for m in data] if isinstance(data, list) else None
        print("Ollama models available:", names)
    except Exception as e:  # pragma: no cover - best-effort example helper
        print(f"Warning: couldn't reach models endpoint at {OLLAMA_URL}: {e}")


def try_set_default_client(url: Optional[str] = None, api_key: Optional[str] = None, use_for_tracing: bool = False) -> None:
    """Set the repository's default OpenAI client to an Ollama-backed AsyncOpenAI client.

    Args:
        url: Optional override for the Ollama base URL.
        api_key: Optional override for the Ollama API key.
        use_for_tracing: Whether this client should also be used for tracing. Defaults to False
            because Ollama does not accept OpenAI tracing keys.

    This is wrapped in a try/except so examples remain robust in dev environments
    that don't have the `openai` package installed.
    """
    try:
        set_default_openai_client(make_ollama_client(url=url, api_key=api_key), use_for_tracing=use_for_tracing)
    except Exception:
        # Swallow any errors to avoid breaking the examples when dependencies are missing.
        pass
