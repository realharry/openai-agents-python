"""Module entrypoint for model_providers examples.

Runs a representative provider example so users can execute:

    uv run python -m examples.model_providers

"""
from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_path(__file__.replace('__main__.py', 'custom_example_provider.py'), run_name='__main__')


if __name__ == '__main__':
    main()
