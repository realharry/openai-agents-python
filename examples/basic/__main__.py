"""Module entrypoint for the basic examples.

This wrapper runs the simple hello-world example so users can execute:

    uv run python -m examples.basic

"""
from __future__ import annotations

import runpy
import sys


def main() -> None:
    # Run the canonical hello world example.
    runpy.run_path(__file__.replace('__main__.py', 'hello_world.py'), run_name='__main__')


if __name__ == '__main__':
    main()
