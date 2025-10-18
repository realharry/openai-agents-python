# Basic examples — hello world (Ollama)

This folder contains small, focused demos that show the minimal code needed to create and run an agent.

This particular example (`hello_world.py`) is configured to use an OpenAI-compatible client. By default
it is set up to talk to a local Ollama server hosting `gemma3:1b`.

Quick start (local Ollama)

1. Install and run Ollama following the official docs: https://ollama.ai/docs

2. Pull or run the `gemma3:1b` model if you need to. The exact Ollama model name may vary; confirm
   available models with the Ollama CLI or the local models endpoint.

   Example (if the model is available via Ollama):

   ```bash
   # Pull or ensure the model is available (example CLI usage)
   ollama pull gemma3:1b
   ollama run gemma3:1b
   ```

3. Run the example. By default `hello_world.py` uses these defaults:

   - OLLAMA_URL=http://localhost:11434/v1
   - OLLAMA_MODEL=gemma3:1b
   - OLLAMA_API_KEY=ollama

   Run with defaults:

   ```bash
   uv run python examples/basic/hello_world.py
   ```

   Or override via environment variables:

   ```bash
   OLLAMA_URL=http://localhost:11434/v1 OLLAMA_MODEL=gemma3:1b uv run python examples/basic/hello_world.py
   ```

Notes

- If your Ollama server is on a different host or port, set `OLLAMA_URL` accordingly.
- Some Ollama setups accept `api_key=ollama` for local usage; otherwise provide the key in `OLLAMA_API_KEY`.
- If the model name differs, use `OLLAMA_MODEL` to change it to the correct name returned by your Ollama instance.

If you'd like, I can also:

- Add a small `run.sh` wrapper that sets sensible defaults.
- Add a test or smoke-run script that checks the model list endpoint and prints available models before running the example.
