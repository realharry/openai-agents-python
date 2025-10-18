## Learning path: examples (simple → advanced)

This file provides a recommended sequence to explore the `examples/` tree, with a one-line learning goal and a quick "how to try" command when a runnable entrypoint was detected.

1. `basic/` — Minimal, introductory examples showing how to create and run a simple agent with the SDK.
   - Why: smallest working examples to get you comfortable with agent lifecycle and APIs.
   - Try it:
     ```bash
     uv run python examples/basic/lifecycle_example.py
     ```

2. `agent_patterns/` — Collection of higher-level agent architecture patterns and templates.
   - Why: learn common orchestration patterns and reusable building blocks.
   - Try it: open the subfolder README and run the individual pattern scripts.

3. `tools/` — Demonstrations of building and wiring tools (local or remote) that agents can call.
   - Why: tools show how to extend agents with external capabilities (search, shell, image gen).
   - Try it: open `examples/tools/` and run any small tool demo (e.g., `local_shell.py`).

4. `model_providers/` — Integration examples for different model providers or provider-specific configuration patterns.
   - Why: switch providers and see provider configuration in practice.
   - Try it: follow the provider README to run a provider-specific example.

5. `memory/` — Examples illustrating memory usage patterns and persistence strategies.
   - Why: learn stateful behavior across turns.
   - Try it: open folder README and run the demo script there.

6. `reasoning_content/` — Structured reasoning and explainable outputs (chain-of-thought patterns).
   - Why: see higher quality reasoning patterns and prompt engineering techniques.

7. `realtime/` — Streaming and low-latency interaction examples.
   - Why: understand streaming outputs and websockets.
   - Try it (web app example):
     ```bash
     cd examples/realtime/app && uv run python server.py
     ```

8. `handoffs/` — Handing off to other agents or humans, and related tooling.
   - Why: implement escalation and state transfer patterns.

9. `customer_service/` — Practical multi-component app for customer support workflows.
   - Why: combines tools, memory, and prompt flows in a domain example.

10. `research_bot/` — Research-style workflows with retrieval and multi-step reasoning.
    - Why: long-running context management and retrieval augmentation.
    - Try it (module run indicated in sample outputs):
      ```bash
      uv run python -m examples.research_bot.main
      ```

11. `financial_research_agent/` — Domain-specific example integrating external data and evaluation.
    - Why: showcases domain prompts and data integrations.

12. `mcp/` — Model Context Protocol examples (self-hosted MCP components and clients).
    - Why: infra-focused examples showing registration, streaming and prompt servers.
    - Try it (some subexamples):
      ```bash
      uv run python examples/mcp/sse_example/main.py
      uv run python examples/mcp/prompt_server/main.py
      ```

13. `hosted_mcp/` — Hosted MCP demos and connectors for managed MCP environments.
    - Why: shows hosted integrations and approval/connectors examples.
    - Try it: inspect `examples/hosted_mcp/simple.py` for a small demo.

14. `voice/` — Voice-enabled agent examples for ASR/TTS and audio I/O.
    - Why: specialized audio pipelines and real-time voice interactions.
    - Try it (static and streamed demos exist):
      ```bash
      uv run python examples/voice/static/main.py
      uv run python examples/voice/streamed/main.py
      ```

Notes
- Read each subfolder `README.md` first; some examples require API keys, services, or extra packages.
- The `uv run` prefix is used in this repository to run Python; if you use a different setup, run with your preferred interpreter.

If you'd like, I can add direct wrapper scripts for any example that lacks a clear entrypoint.
