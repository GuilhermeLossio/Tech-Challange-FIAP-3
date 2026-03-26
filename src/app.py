#!/usr/bin/env python3
"""Deployment-friendly entrypoint for Flight Advisor.

Examples
--------
python src/app.py
gunicorn -w 2 -b 0.0.0.0:$PORT src.app:app
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
root_dir_str = str(ROOT_DIR)
if root_dir_str not in sys.path:
    sys.path.insert(0, root_dir_str)

from src.api.main import app as flask_app, run_local_server  # noqa: E402

app = flask_app
application = app

__all__ = ["app", "application", "main"]


def main() -> int:
    """Run the Flask app using Railway-friendly host and port resolution."""
    return run_local_server()


if __name__ == "__main__":
    raise SystemExit(main())
