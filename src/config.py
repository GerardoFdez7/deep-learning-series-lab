from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_CANDIDATES = [
    PROJECT_ROOT / "Base_Migracion_2009-2026jun.xlsx",
    PROJECT_ROOT / "data" / "raw" / "Base_Migracion_2009-2026jun.xlsx",
]

TRAIN_RATIO = 0.70
VALID_RATIO_WITHIN_TRAIN = 0.15
RANDOM_SEED = 42

SERIES_TO_MODEL = ["total_viajeros", "via_aerea"]

MODEL_FAMILIES = {
    "lstm_simple": [
        {"window": 6, "units": 32, "dropout": 0.0, "lr": 1e-3, "batch_size": 16},
        {"window": 12, "units": 32, "dropout": 0.2, "lr": 1e-3, "batch_size": 16},
        {"window": 12, "units": 64, "dropout": 0.2, "lr": 5e-4, "batch_size": 16},
    ],
    "lstm_stacked": [
        {"window": 6, "units": 32, "dropout": 0.2, "lr": 1e-3, "batch_size": 16},
        {"window": 12, "units": 32, "dropout": 0.2, "lr": 1e-3, "batch_size": 16},
        {"window": 12, "units": 64, "dropout": 0.3, "lr": 5e-4, "batch_size": 32},
    ],
}

MAX_EPOCHS = 80
PATIENCE = 10
