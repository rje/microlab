from __future__ import annotations

import argparse
import json
from pathlib import Path

from microlab.evals.backends import create_backend
from microlab.evals.runner import run_eval_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Microlab evaluation suite")
    parser.add_argument("--suite", required=True, help="Path to JSONL eval suite")
    parser.add_argument("--config", required=True, help="Path to JSON eval config")
    parser.add_argument("--output-dir", required=True, help="Directory for run artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    backend = create_backend(config["backend"])
    summary = run_eval_suite(
        suite_path=args.suite,
        backend=backend,
        output_dir=args.output_dir,
        run_config=config,
    )
    print(
        f"total={summary['total']} passed={summary['passed']} "
        f"failed={summary['failed']} pass_rate={summary['pass_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
