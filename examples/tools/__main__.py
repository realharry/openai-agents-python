"""Module entrypoint for tools examples.

Runs a simple local shell tool demo so users can execute:

    uv run python -m examples.tools

"""
from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_path(__file__.replace('__main__.py', 'local_shell.py'), run_name='__main__')


if __name__ == '__main__':
    main()
