"""
author: Jayesh Pandey
summary: Repository entrypoint for VLMJudge, delegating to the modular vlmjudge.main function.
"""

from __future__ import annotations

import sys

from vlmjudge.main import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
