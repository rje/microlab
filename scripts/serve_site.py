from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.console.app import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Microlab Console.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()
    app = create_app(args.project_root)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
