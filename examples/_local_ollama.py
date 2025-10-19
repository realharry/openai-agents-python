"""Helpers for examples that want to talk to a local Ollama instance.

This centralizes env-var handling and small smoke-checks so example files don't
have duplicated code.
"""
from __future__ import annotations

import os
from typing import Optional

from openai import AsyncOpenAI

from agents import (
    OpenAIChatCompletionsModel,
    set_default_openai_client,
    OpenAIResponsesModel,
    set_default_openai_api,
)
from typing import Tuple
import httpx
import time
from agents.items import ModelResponse
from agents.usage import Usage


# Defaults used by the examples. Override with environment variables.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")


def _normalize_base(url: str) -> str:
    # Ensure we return a base URL without a trailing `/v1` segment. The
    # OpenAI client will append the API path (e.g. `/v1/responses`), so if the
    # base already contains `/v1` we will end up with `/v1/v1/...` and 404s.
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -3]
    return base


def make_ollama_client(url: Optional[str] = None, api_key: Optional[str] = None) -> AsyncOpenAI:
    """Create an AsyncOpenAI client configured to talk to an Ollama HTTP endpoint.

    Args:
        url: Base URL for the Ollama server (example: http://localhost:11434/v1).
        api_key: API key to send (Ollama often accepts `ollama` as a local sentinel).
    """
    base = url or OLLAMA_URL
    base = _normalize_base(base)
    return AsyncOpenAI(base_url=base, api_key=api_key or OLLAMA_API_KEY)


def make_chat_model(model_name: str, client: Optional[AsyncOpenAI] = None) -> OpenAIChatCompletionsModel:
    """Convenience factory for creating an OpenAIChatCompletionsModel that uses Ollama."""
    client = client or make_ollama_client()
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


def _supports_responses(client: Optional[AsyncOpenAI] = None, timeout: float = 1.0) -> bool:
    """Probe the server to see if `/v1/responses` exists.

    Returns True when the endpoint responds (not a 404). Best-effort and fast.
    """
    try:
        base = OLLAMA_URL
        if client is not None:
            base = getattr(client, "base_url", base)
        base = _normalize_base(base)
        url = base.rstrip("/") + "/v1/responses"
        with httpx.Client(timeout=timeout) as http:
            try:
                r = http.options(url)
            except Exception:
                print(f"[debug] responses probe failed for {url}")
                return False
            # If it's 404, responses aren't supported. 204/200/405 are ok.
            try:
                print(f"[debug] responses probe {url} status={r.status_code}")
            except Exception:
                pass
            return r.status_code != 404
    except Exception:
        return False


def make_model_for_ollama(model_name: str, client: Optional[AsyncOpenAI] = None) -> Tuple[object, bool]:
    """Return a model wrapper suitable for the Ollama instance.

    Returns (model_instance, used_responses_api).
    If the Ollama server supports the Responses API, returns an OpenAIResponsesModel.
    Otherwise, returns an OpenAIChatCompletionsModel as a fallback.
    """
    client = client or make_ollama_client()
    # Allow env override to force chat mode (useful for CI or known incompatibilities)
    force_chat = os.environ.get("OLLAMA_USE_CHAT", "").lower() in ("1", "true", "yes")
    if not force_chat and _supports_responses(client=client):
        try:
            m = OpenAIResponsesModel(model=model_name, openai_client=client)
            print(f"[debug] make_model_for_ollama: returning OpenAIResponsesModel for {model_name}")
            return m, True
        except Exception:
            # If constructing fails for any reason, fall back to chat model below
            pass

    # If Ollama exposes a legacy /api/generate endpoint (some local builds), prefer
    # a small generate shim — this avoids depending on the OpenAI SDK paths that
    # may 404 on certain Ollama versions.
    try:
        base_tmp = getattr(client, "base_url", OLLAMA_URL) if client is not None else OLLAMA_URL
        base_tmp = _normalize_base(base_tmp)
        gen_url = base_tmp.rstrip("/") + "/api/generate"
        with httpx.Client(timeout=1.0) as http:
            try:
                rgen = http.post(gen_url, json={"model": model_name, "prompt": ""})
            except Exception as ex:
                rgen = None
                print(f"[debug] early generate probe exception for {gen_url}: {ex}")
            try:
                if rgen is not None:
                    body_snip = None
                    try:
                        body_snip = rgen.text[:200]
                    except Exception:
                        body_snip = str(rgen.content)[:200]
                    print(f"[debug] early generate probe {gen_url} status={rgen.status_code} body={body_snip}")
            except Exception:
                pass
            # Treat any non-404 as evidence the endpoint exists
            if rgen is not None and rgen.status_code != 404:
                base = getattr(client, "base_url", OLLAMA_URL) if client is not None else OLLAMA_URL
                base = _normalize_base(base)
                key = None
                try:
                    key = getattr(client, "api_key", None)
                except Exception:
                    key = None
                print(f"[debug] make_model_for_ollama: returning _OllamaGenerateModel for {model_name} (early probe)")
                return _OllamaGenerateModel(model_name, base, key or OLLAMA_API_KEY), False
    except Exception:
        pass

    # If chat completions are supported, return that.
    try:
        # Probe for chat support by attempting a very small POST to /v1/chat/completions.
        # Some Ollama builds respond to OPTIONS even when POST/create is not implemented,
        # so a lightweight POST gives a more accurate runtime check.
        base = getattr(client, "base_url", OLLAMA_URL) if client is not None else OLLAMA_URL
        base = _normalize_base(base)
        chat_url = base.rstrip("/") + "/v1/chat/completions"
        with httpx.Client(timeout=1.0) as http:
            # Send a minimal payload; consider chat supported only for 2xx responses.
            try:
                r = http.post(chat_url, json={"model": model_name, "messages": [{"role": "user", "content": ""}]})
            except Exception:
                r = None
            # Debug/logging to aid diagnosis in examples (best-effort).
            try:
                if r is not None:
                    print(f"[debug] chat probe {chat_url} status={r.status_code}")
            except Exception:
                pass
            if r is not None and 200 <= r.status_code < 300:
                # Create a lightweight shim that posts directly to /v1/chat/completions
                class _OllamaChatModel:
                    def __init__(self, model: str, base: str, api_key: Optional[str]):
                        self.model = model
                        self._base = base
                        self._api_key = api_key

                    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, previous_response_id=None, conversation_id=None, prompt=None):
                        # Build messages similar to the SDK's conversion
                        from agents.models.chatcmpl_converter import Converter

                        # Try to create compatible messages list
                        try:
                            # Reuse Converter.message formats where possible
                            messages = Converter.items_to_messages(input)
                        except Exception:
                            # Fallback: if input is str, wrap as user message
                            if isinstance(input, str):
                                messages = [{"role": "user", "content": input}]
                            else:
                                messages = [
                                    {"role": "user", "content": str(input)}
                                ]

                        if system_instructions:
                            messages.insert(0, {"role": "system", "content": system_instructions})

                        body = {"model": self.model, "messages": messages}
                        # Include response_format to request JSON output when examples expect it
                        try:
                            if output_schema and not output_schema.is_plain_text():
                                body["response_format"] = {
                                    "type": "json_schema",
                                    "json_schema": {"name": "final_output", "strict": output_schema.is_strict_json_schema(), "schema": output_schema.json_schema()},
                                }
                        except Exception:
                            pass

                        headers = {"Content-Type": "application/json"}
                        if self._api_key:
                            headers["Authorization"] = f"Bearer {self._api_key}"

                        url = str(self._base).rstrip("/") + "/v1/chat/completions"
                        async with httpx.AsyncClient(timeout=30.0) as http:
                            resp = await http.post(url, json=body, headers=headers)
                            resp.raise_for_status()
                            j = resp.json()

                        # Extract text from the first choice message
                        text = None
                        try:
                            choices = j.get("choices") if isinstance(j, dict) else None
                            if choices and isinstance(choices, list) and len(choices) > 0:
                                msg = choices[0].get("message")
                                # message.content may be a string or dict/list; handle common shapes
                                if isinstance(msg, dict):
                                    content = msg.get("content")
                                    if isinstance(content, str):
                                        text = content
                                    elif isinstance(content, dict) and "text" in content:
                                        text = content.get("text")
                        except Exception:
                            text = None

                        if text is None:
                            text = j.get("text") if isinstance(j, dict) and "text" in j else str(j)

                        output_item = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
                        return ModelResponse(output=[output_item], usage=Usage(), response_id=None)

                base = getattr(client, "base_url", OLLAMA_URL) if client is not None else OLLAMA_URL
                base = _normalize_base(base)
                key = None
                try:
                    key = getattr(client, "api_key", None)
                except Exception:
                    key = None
                print(f"[debug] make_model_for_ollama: returning _OllamaChatModel for {model_name}")
                return _OllamaChatModel(model_name, base, key or OLLAMA_API_KEY), False
    except Exception:
        pass

    # If an Ollama-specific /api/generate endpoint exists, provide a small shim
    # that posts to it and converts the response into a minimal ModelResponse.
    def _supports_generate(client: Optional[AsyncOpenAI] = None) -> bool:
        try:
            base = getattr(client, "base_url", OLLAMA_URL) if client is not None else OLLAMA_URL
            base = _normalize_base(base)
            gen_url = base.rstrip("/") + "/api/generate"
            with httpx.Client(timeout=1.0) as http:
                # Use a small POST probe — if the endpoint exists it will respond
                # to POST even if the payload is minimal/empty. Treat any non-404
                # as presence of the endpoint.
                try:
                    r = http.post(gen_url, json={"model": model_name, "prompt": ""})
                except Exception:
                    r = None
                try:
                    if r is not None:
                        print(f"[debug] generate probe {gen_url} status={r.status_code}")
                except Exception:
                    pass
                return (r is not None) and (r.status_code != 404)
        except Exception:
            return False

    if _supports_generate(client=client):
        # Define a minimal adapter implementing the Model interface used by examples.
        class _OllamaGenerateModel:
            def __init__(self, model: str, base: str, api_key: Optional[str]):
                self.model = model
                self._base = base
                self._api_key = api_key

            async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, previous_response_id=None, conversation_id=None, prompt=None):
                # Convert input to a single text prompt
                if isinstance(input, str):
                    prompt_text = input
                else:
                    try:
                        # input is list of items like {"content": ..., "role": ...}
                        parts = []
                        for it in input:
                            if isinstance(it, dict):
                                c = it.get("content")
                                if isinstance(c, str):
                                    parts.append(c)
                                elif isinstance(c, list):
                                    # join text parts
                                    for p in c:
                                        if isinstance(p, dict) and p.get("type") == "text":
                                            parts.append(p.get("text", ""))
                        prompt_text = "\n\n".join(parts) if parts else ""
                    except Exception:
                        prompt_text = str(input)

                base = self._base
                url = str(base).rstrip("/") + "/api/generate"
                headers = {"Content-Type": "application/json"}
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"

                body = {"model": self.model, "prompt": prompt_text}
                async with httpx.AsyncClient(timeout=30.0) as http:
                    resp = await http.post(url, json=body, headers=headers)
                    resp.raise_for_status()
                    # Try to parse various response shapes
                    try:
                        data = resp.json()
                    except Exception:
                        data = {"output": resp.text}

                # Extract text from common keys
                text = None
                if isinstance(data, dict):
                    for k in ("output", "text", "result", "response", "content"):
                        if k in data:
                            v = data[k]
                            if isinstance(v, str):
                                text = v
                                break
                            elif isinstance(v, dict) and "text" in v:
                                text = v["text"]
                                break
                    if text is None:
                        # Fallback: stringify data
                        text = str(data)
                else:
                    text = str(data)

                # Build a minimal ModelResponse compatible object
                output_item = {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
                return ModelResponse(output=[output_item], usage=Usage(), response_id=None)

        base = getattr(client, "base_url", OLLAMA_URL) if client is not None else OLLAMA_URL
        base = _normalize_base(base)
        key = None
        try:
            key = getattr(client, "api_key", None)
        except Exception:
            key = None
        print(f"[debug] make_model_for_ollama: returning _OllamaGenerateModel for {model_name}")
        return _OllamaGenerateModel(model_name, base, key or OLLAMA_API_KEY), False

    # Final fallback: return a chat completions model (best-effort)
    print(f"[debug] make_model_for_ollama: returning OpenAIChatCompletionsModel for {model_name}")
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client), False


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

        base = _normalize_base(base)
        url = base.rstrip("/") + "/v1/models"
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
        client = make_ollama_client(url=url, api_key=api_key)
        set_default_openai_client(client, use_for_tracing=use_for_tracing)

        # If Ollama doesn't support the Responses API, switch the default API to
        # chat_completions so examples use the compatible path and don't 404.
        try:
            if not _supports_responses(client=client):
                set_default_openai_api("chat_completions")
                print("[info] Ollama does not support Responses API, defaulting to chat_completions.")
        except Exception:
            # Best-effort: don't fail examples if the probe errors.
            pass
    except Exception:
        # Swallow any errors to avoid breaking the examples when dependencies are missing.
        pass
