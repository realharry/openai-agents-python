import asyncio
import logging
import os
from typing import Optional

from agents import Agent, Runner, set_tracing_disabled
from examples._local_ollama import make_chat_model, smoke_check_models, make_ollama_client

set_tracing_disabled(True)
logging.basicConfig(level=logging.DEBUG)


GPT_OSS_MODEL: str = os.environ.get("GPT_OSS_MODEL", "gpt-oss:20b")


gpt_oss_model = make_chat_model(GPT_OSS_MODEL)


async def main():
    client = make_ollama_client()
    await smoke_check_models(client)

    agent = Agent(
        name="Assistant",
        instructions="You're a helpful assistant. You provide a concise answer to the user's question.",
        model=gpt_oss_model,
    )

    result = await Runner.run(agent, "Tell me about recursion in programming.")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
