import asyncio
import sys
from pathlib import Path
import os

from agents import Agent, Runner

# Import shared Ollama helpers with a sys.path fallback (when running as script).
try:
    from examples._local_ollama import make_model_for_ollama, try_set_default_client
    from examples._local_ollama import smoke_check_models, make_ollama_client
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root))
    from examples._local_ollama import make_model_for_ollama, try_set_default_client
    from examples._local_ollama import smoke_check_models, make_ollama_client


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stream or non-stream example using Ollama")
    parser.add_argument("--no-stream", action="store_true", help="Run non-streaming path")
    args = parser.parse_args()

    # Configure default client and model
    try_set_default_client()
    model, used_responses = make_model_for_ollama(os.environ.get("OLLAMA_MODEL", "gemma3:1b"))
    print(f"[info] using {'Responses' if used_responses else 'ChatCompletions'} API for model calls")

    # Smoke check the local Ollama models endpoint (best-effort)
    try:
        client = make_ollama_client()
        await smoke_check_models(client)
    except Exception:
        # Non-fatal: just continue if smoke check fails.
        pass

    agent = Agent(
        name="Joker",
        instructions="You are a helpful assistant.",
        model=model,
    )

    if args.no_stream:
        # Non-streaming path
        result = await Runner.run(agent, "Please tell me 5 jokes.")
        print(result.final_output)
    else:
        # Streaming path
        result = Runner.run_streamed(agent, input="Please tell me 5 jokes.")
        async for event in result.stream_events():
            data = getattr(event, "data", None)
            if hasattr(data, "delta"):
                print(data.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
