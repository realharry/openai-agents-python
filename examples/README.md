# Examples

This folder contains curated example agents and patterns demonstrating how to use the OpenAI Agents SDK. Each subfolder contains a focused example or set of examples; see the individual subfolder README files for detailed instructions, code walkthroughs, and usage notes.

For a recommended learning order (simple → advanced) and quick "try it" commands, see `examples/ORDERED.md`.

Notes

- This README intentionally provides only brief summaries. For usage, setup and examples for each item, consult the README.md inside the corresponding subfolder.
- Most examples assume you have the repository dependencies installed and configured. Refer to the top-level `README.md` and `AGENTS.md` for environment setup and run instructions.

How to run the examples (quick)

- Read the subfolder `README.md` first — many examples list prerequisites, required env vars, or provider keys.
- Common command patterns used in this repo:

  ```bash
  # Run a single example script
  uv run python examples/<folder>/script.py

  # Run a module-style example
  uv run python -m examples.<folder>
  ```

If you'd like, I can normalize entrypoints across examples (add `main.py` wrappers) or open a PR that adds a top-level tutorial with a guided walkthrough for the first 4 examples.
# Examples

This folder contains curated example agents and patterns demonstrating how to use the OpenAI Agents SDK. Each subfolder contains a focused example or set of examples; see the individual subfolder README files for detailed instructions, code walkthroughs, and usage notes. The list below gives a short one-line summary of each examples subfolder.

- `agent_patterns/` — Collection of higher-level agent architecture patterns and reusable templates for building complex agents.
- `basic/` — Minimal, introductory examples showing how to create and run a simple agent with the SDK.
- `customer_service/` — Examples that demonstrate building agents tailored for customer support and conversational workflows.
- `financial_research_agent/` — Example agent(s) focused on financial data retrieval, analysis, and research-oriented workflows.
- `handoffs/` — Demonstrations of handoff behaviors (e.g., transferring between agents, human-in-the-loop handoffs) and related tooling.
- `hosted_mcp/` — Examples that show how to host or deploy an MCP (Model Context Protocol) server or integrate with a hosted MCP environment.
- `mcp/` — Examples and reference implementations for the Model Context Protocol, including how to register tools and connect agents.
- `memory/` — Examples illustrating memory usage patterns, persistence strategies, and memory-backed agent behaviors.
- `model_providers/` — Integration examples for different model providers or provider-specific configuration patterns.
- `realtime/` — Realtime and streaming examples demonstrating low-latency interactions, websockets or streaming model responses.
- `reasoning_content/` — Examples focused on structured reasoning, chain-of-thought patterns, and producing explainable outputs.
- `research_bot/` — Example(s) implementing research assistant behaviors, retrieval-augmented generation, and multi-step investigations.
- `tools/` — Demonstrations of building and wiring tools (local or remote) that agents can call during execution.
- `voice/` — Voice-enabled agent examples showing TTS/ASR integrations and voice interaction patterns.

Notes

- This README intentionally provides only brief summaries. For usage, setup and examples for each item above, consult the README.md inside the corresponding subfolder.
- Most examples assume you have the repository dependencies installed and configured. Refer to the top-level `README.md` and `AGENTS.md` for environment setup and run instructions.
- If you'd like, I can expand any one-line summary into a longer description or add quick "try it" commands for a particular example directory.
