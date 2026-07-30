from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MAX_EPOCHS, SERIES_TO_MODEL
from src.training import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run first delivery: LSTM models with hyperparameter tuning."
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=SERIES_TO_MODEL,
        help="Series keys to model (default: total_viajeros via_aerea)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
        help="Max training epochs per trial.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(output_dir=PROJECT_ROOT, series_to_model=args.series, max_epochs=args.epochs)


if __name__ == "__main__":
    main()
