# save as tools/check_ollama_conn.py and run: python tools/check_ollama_conn.py
import asyncio
import sys
import pathlib
import importlib.util

# Ensure repo root is on sys.path so `examples` imports work when run as a script
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from examples._local_ollama import make_ollama_client, OLLAMA_URL
except Exception:
    # Try loading the file directly
    try:
        mod_path = repo_root / "examples" / "_local_ollama.py"
        spec = importlib.util.spec_from_file_location("examples._local_ollama", str(mod_path))
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
        make_ollama_client = getattr(_mod, "make_ollama_client")
        OLLAMA_URL = getattr(_mod, "OLLAMA_URL")
    except Exception:
        raise

import httpx

async def main():
    client = make_ollama_client()
    base = getattr(client, "base_url", None)
    # client.base_url may be an httpx.URL object; convert to str for manipulation
    if base is not None:
        base = str(base)
    print("OLLAMA_URL env/default:", OLLAMA_URL)
    print("openai.AsyncOpenAI client.base_url:", base)

    # Check /v1/models
    url = (base or OLLAMA_URL).rstrip("/") + "/v1/models"
    print("Checking models endpoint:", url)
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            r = await http.get(url)
            print("models status:", r.status_code)
            print("models body:", r.text[:2000])
    except Exception as e:
        print("Error touching models endpoint:", repr(e))

    # Check /v1/responses root (minimal)
    url2 = (base or OLLAMA_URL).rstrip("/") + "/v1/responses"
    print("Checking responses endpoint (OPTIONS):", url2)
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            r = await http.options(url2)
            print("responses OPTIONS:", r.status_code, r.headers.get("allow"))
    except Exception as e:
        print("Error touching responses endpoint:", repr(e))

asyncio.run(main())
