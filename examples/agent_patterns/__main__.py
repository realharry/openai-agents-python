"""Module entrypoint for agent_patterns examples.

Runs a representative pattern script so users can execute:

    uv run python -m examples.agent_patterns

"""
from __future__ import annotations

import runpy


def main() -> None:
    # Run a representative example (deterministic pattern).
    runpy.run_path(__file__.replace('__main__.py', 'deterministic.py'), run_name='__main__')


if __name__ == '__main__':
    main()
