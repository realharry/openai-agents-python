import asyncio
import os
from typing import Optional

from agents import Agent, Runner
from examples._local_ollama import make_model_for_ollama


async def main():
    model, used_responses = make_model_for_ollama(os.environ.get("OLLAMA_MODEL", "gemma3:1b"))
    print(f"[info] using {'Responses' if used_responses else 'ChatCompletions'} API for model calls")

    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=model,
    )

    result = await Runner.run(agent, "Tell me about recursion in programming.")
    print(result.final_output)
    # Function calls itself,
    # Looping in smaller pieces,
    # Endless by design.


if __name__ == "__main__":
    asyncio.run(main())
