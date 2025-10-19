import asyncio
import base64
import os
import sys
from pathlib import Path

from agents import Agent, Runner

# When running the example as a script (python examples/basic/local_file.py) the
# top-level package `examples` may not be importable because the script's
# directory becomes sys.path[0]. Try importing helpers from the package first;
# if that fails, add the repository root to sys.path and import again.
try:
    from examples._local_ollama import make_model_for_ollama, try_set_default_client
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root))
    from examples._local_ollama import make_model_for_ollama, try_set_default_client

FILEPATH = os.path.join(os.path.dirname(__file__), "media/partial_o3-and-o4-mini-system-card.pdf")


def file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_text_from_pdf(path: str) -> str:
    """Try to extract text from a PDF using PyPDF2. If PyPDF2 is not installed,
    raise ImportError so the caller can fall back to the base64 approach.
    """
    try:
        import PyPDF2

        text_parts: list[str] = []
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        return "\n\n".join(text_parts).strip()
    except Exception as e:
        raise ImportError("PyPDF2 not available or failed to extract PDF text") from e


async def main():
    # Set a default Ollama-backed OpenAI client if available.
    try_set_default_client()

    # Create a model configured via OLLAMA_MODEL (defaults to gemma3:1b)
    model, used_responses = make_model_for_ollama(os.environ.get("OLLAMA_MODEL", "gemma3:1b"))
    print(f"[info] using {'Responses' if used_responses else 'ChatCompletions'} API for model calls")

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model=model,
    )

    b64_file = file_to_base64(FILEPATH)
    # Prefer sending extracted text to the model. If PyPDF2 is not installed, fall
    # back to sending the base64 payload as before (some providers accept custom
    # file input objects).
    try:
        text = extract_text_from_pdf(FILEPATH)
        prompt = f"Document text:\n\n{text}\n\nQuestion: What is the first sentence of the introduction?"
        result = await Runner.run(agent, prompt)
    except ImportError:
        print("PyPDF2 not available — sending base64 PDF as a plain text prompt. To enable text extraction, install PyPDF2:")
        print("  pip install PyPDF2")
        prompt = (
            f"Here is a PDF file encoded as base64:\n\n{b64_file}\n\n"
            "Please extract the text from the PDF and answer: What is the first sentence of the introduction?"
        )
        result = await Runner.run(agent, prompt)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
