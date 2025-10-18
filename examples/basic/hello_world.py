import asyncio
import os
from typing import Optional

from agents import Agent, Runner
from examples._local_ollama import make_chat_model


async def main():
    model = make_chat_model(os.environ.get("OLLAMA_MODEL", "gemma3:1b"))

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
