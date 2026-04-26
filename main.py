"""
Repository entrypoint.

Kept at the repo root so it can be run as:
    python main.py --image path --prompt "text"

This is additive and does not change legacy `ImageReward` behavior.
"""

from __future__ import annotations

import sys

from vlmjudge.main import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
