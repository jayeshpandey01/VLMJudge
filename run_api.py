"""
author: Jayesh Pandey
summary: Entrypoint for running the FastAPI server using Uvicorn.
"""

import argparse
import logging

import uvicorn

from api.app import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the VLMJudge FastAPI server.")
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--config", default="config.yaml", type=str)
    parser.add_argument("--log-level", default="INFO", type=str)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    app = create_app(config_path=args.config)
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level=str(args.log_level).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

