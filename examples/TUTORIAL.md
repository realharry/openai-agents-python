# Quick tutorial — first 4 examples

This guided walkthrough helps a new user run the first four examples in the recommended learning order: `basic`, `agent_patterns`, `tools`, and `model_providers`.

Prerequisites

- Install repository dependencies (see top-level `AGENTS.md`). This project expects Python and uses the `uv` runner in the examples; adapt the commands to your environment if necessary.
- Set any provider API keys or environment variables required by the examples you plan to run. Each subfolder README lists requirements.

Step 1 — Basic: hello world and lifecycle

Purpose: see the smallest working agent and how lifecycle events are logged.

Commands:

```bash
# Run the module-style entrypoint (wrapper runs hello_world)
uv run python -m examples.basic

# Or run the lifecycle script directly to see more lifecycle output
uv run python examples/basic/lifecycle_example.py
```

What to expect:

- Short printed output that demonstrates an agent sending and receiving a message, and lifecycle hooks.

Step 2 — Agent patterns

Purpose: inspect architecture patterns and a deterministic demo.

Commands:

```bash
uv run python -m examples.agent_patterns
```

What to expect:

- A demonstration of a deterministic agent pattern. Explore other files in `examples/agent_patterns/` like `routing.py` or `parallelization.py` for more patterns.

Step 3 — Tools

Purpose: learn how to wire external capabilities (shell, web search, image gen) to an agent.

Commands:

```bash
uv run python -m examples.tools
```

What to expect:

- The wrapper runs a local shell tool demo. Inspect `examples/tools/` to see other tools and how the agent calls them.

Step 4 — Model providers

Purpose: switch providers and see provider-specific wiring.

Commands:

```bash
uv run python -m examples.model_providers
```

What to expect:

- The representative provider example will run. Check `examples/model_providers/README.md` and confirm any provider keys are set.

After step ideas

- Next, try `examples/memory/` to add state, then `examples/reasoning_content/` to study advanced prompt engineering. Use `examples/ORDERED.md` as a roadmap.

If you'd like, I can:

- Add small `run.sh` wrappers to each example to simplify commands.
- Add more expected-output snapshots to this tutorial.
