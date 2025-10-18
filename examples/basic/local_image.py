
import asyncio
import base64
import os
import sys
from pathlib import Path

from agents import Agent, Runner

# Attempt to import the shared Ollama helpers; if running the script directly,
# add the repo root to sys.path so the `examples` package is importable.
try:
    from examples._local_ollama import make_chat_model, try_set_default_client
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root))
    from examples._local_ollama import make_chat_model, try_set_default_client

FILEPATH = os.path.join(os.path.dirname(__file__), "media/image_bison.jpg")


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


async def main():
    # Print base64-encoded image
    b64_image = image_to_base64(FILEPATH)

    # Configure Ollama-backed client and model (if available)
    try_set_default_client()
    model = make_chat_model(os.environ.get("OLLAMA_MODEL", "gemma3:1b"))

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model=model,
    )

    # Send the image as a plain-text prompt (base64) to avoid provider-specific
    # structured message formats that some providers reject.
    prompt = (
        f"Here is an image encoded as a base64 JPEG:\n\n{b64_image}\n\n"
        "Please describe what you see in this image."
    )
    result = await Runner.run(agent, prompt)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
