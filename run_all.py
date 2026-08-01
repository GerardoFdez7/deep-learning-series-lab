"""Run the full Lab 2 pipeline: LSTM tuning, catch22 analysis and catch22 LSTM.

Works from any working directory: every path is resolved from this file's location.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

PIPELINE = [
    ("avance", PROJECT_ROOT / "experiments" / "run_avance.py"),
    ("catch22", PROJECT_ROOT / "experiments" / "run_catch22.py"),
    ("catch22_lstm", PROJECT_ROOT / "experiments" / "run_catch22_lstm.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=[name for name, _ in PIPELINE],
        help="Pipeline steps to skip (e.g. --skip avance to reuse existing LSTM results).",
    )
    return parser.parse_args()


def run_script(path: Path) -> None:
    print(f"\n{'=' * 80}\n>>> Running: {path.relative_to(PROJECT_ROOT)}\n{'=' * 80}")
    result = subprocess.run([sys.executable, str(path)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline stopped: {path.name} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    args = parse_args()

    steps = [(name, path) for name, path in PIPELINE if name not in args.skip]

    missing = [path for _, path in steps if not path.exists()]
    if missing:
        listed = "\n".join(f"- {path}" for path in missing)
        print(f"[ERROR] Missing pipeline scripts:\n{listed}")
        sys.exit(1)

    print("Starting the Lab 2 pipeline...")
    for _, path in steps:
        run_script(path)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print(f"Figures: {PROJECT_ROOT / 'reports' / 'figures'}")
    print(f"Tables:  {PROJECT_ROOT / 'reports' / 'tables'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
