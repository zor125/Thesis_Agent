from __future__ import annotations

import argparse

from paper_agent.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-agent")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    app_name = config.get("app", {}).get("name", "paper-agent")
    print(f"{app_name} is ready.")
