"""Exercise 2.14: LSTM fed with catch22 features, compared against the best plain LSTM."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from src.catch22_analysis import extract_catch22_features
from src.config import PATIENCE, RANDOM_SEED, TRAIN_RATIO, VALID_RATIO_WITHIN_TRAIN
from src.data import build_selected_series, load_clean_data
from src.evaluation import regression_metrics
from src.modeling import build_lstm_catch22, set_global_seed
from src.preprocessing import (
    inverse_transform,
    prepare_windowed_data,
    split_train_test,
    train_val_split,
)

FEATURES_PATH = PROJECT_ROOT / "reports" / "tables" / "catch22_features.csv"
BEST_MODELS_PATH = PROJECT_ROOT / "reports" / "tables" / "best_models_summary.csv"
DEFAULT_TARGET = "total_viajeros"
DEFAULT_WINDOW = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        default=DEFAULT_TARGET,
        help=f"Series used for the head-to-head comparison (default: {DEFAULT_TARGET}).",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Max training epochs.")
    return parser.parse_args()


def load_or_extract_features(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Reuse the matrix produced by run_catch22.py instead of recomputing it."""
    if FEATURES_PATH.exists():
        df_features = pd.read_csv(FEATURES_PATH, index_col=0)
        print(f"[catch22] Reusing feature matrix from {FEATURES_PATH}")
        return df_features

    print(f"[catch22] {FEATURES_PATH.name} not found; extracting features now.")
    return extract_catch22_features(series_dict)


def best_lstm_row(series_name: str) -> pd.Series | None:
    """Row of the best plain LSTM for this series, as reported by run_avance.py."""
    if not BEST_MODELS_PATH.exists():
        print(f"[warn] {BEST_MODELS_PATH} not found. Run experiments/run_avance.py first.")
        return None

    summary = pd.read_csv(BEST_MODELS_PATH)
    match = summary[summary["series_name"] == series_name]
    if match.empty:
        print(f"[warn] '{series_name}' is not in {BEST_MODELS_PATH.name}.")
        return None
    return match.iloc[0]


def resolve_window(best_row: pd.Series | None) -> int:
    """Match the window of the reference model so both are scored on the same test set."""
    if best_row is None:
        return DEFAULT_WINDOW
    try:
        return int(ast.literal_eval(best_row["params"])["window"])
    except (ValueError, SyntaxError, KeyError, TypeError):
        print(f"[warn] Could not read the window from best_models_summary; using {DEFAULT_WINDOW}.")
        return DEFAULT_WINDOW


def main() -> None:
    args = parse_args()
    set_global_seed(RANDOM_SEED)

    print("Loading data...")
    series_dict = build_selected_series(load_clean_data())

    df_features = load_or_extract_features(series_dict)

    best_row = best_lstm_row(args.series)
    window = resolve_window(best_row)
    print(f"[setup] Target series: {args.series} | window: {window}")

    if args.series not in df_features.index:
        raise SystemExit(
            f"Series '{args.series}' has no catch22 features "
            f"(available: {list(df_features.index)})."
        )

    scaler_c22 = StandardScaler()
    scaled_c22 = scaler_c22.fit_transform(df_features)
    c22_dim = scaled_c22.shape[1]
    c22_by_series = {name: scaled_c22[i] for i, name in enumerate(df_features.index)}

    x_train_parts, y_train_parts, c22_train_parts = [], [], []
    x_val_parts, y_val_parts, c22_val_parts = [], [], []
    windowed_by_series: dict[str, object] = {}

    for name, series in series_dict.items():
        if name not in c22_by_series:
            continue

        train, test = split_train_test(series, TRAIN_RATIO)
        train_sub, val = train_val_split(train, VALID_RATIO_WITHIN_TRAIN)

        try:
            data = prepare_windowed_data(train_sub, val, test, window)
        except ValueError as exc:
            print(f"[skip] {name}: {exc}")
            continue

        windowed_by_series[name] = data
        features = c22_by_series[name]

        x_train_parts.append(data.x_train)
        y_train_parts.append(data.y_train)
        c22_train_parts.append(np.tile(features, (data.x_train.shape[0], 1)))

        x_val_parts.append(data.x_val)
        y_val_parts.append(data.y_val)
        c22_val_parts.append(np.tile(features, (data.x_val.shape[0], 1)))

    if args.series not in windowed_by_series:
        raise SystemExit(f"Series '{args.series}' could not be windowed with window={window}.")

    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    c22_train = np.concatenate(c22_train_parts, axis=0)
    x_val = np.concatenate(x_val_parts, axis=0)
    y_val = np.concatenate(y_val_parts, axis=0)
    c22_val = np.concatenate(c22_val_parts, axis=0)

    print(
        f"[setup] Pooled {len(windowed_by_series)} series -> "
        f"{x_train.shape[0]} train windows, {x_val.shape[0]} validation windows"
    )

    print("Building catch22-conditioned LSTM...")
    model = build_lstm_catch22(
        input_shape=(window, 1),
        catch22_dim=c22_dim,
        units=64,
        dropout=0.2,
        lr=1e-3,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )

    print("Training...")
    model.fit(
        {"ts_input": x_train, "catch22_input": c22_train},
        y_train,
        validation_data=({"ts_input": x_val, "catch22_input": c22_val}, y_val),
        epochs=args.epochs,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    data = windowed_by_series[args.series]
    c22_test = np.tile(c22_by_series[args.series], (data.x_test.shape[0], 1))

    y_pred = inverse_transform(
        data.scaler,
        model.predict({"ts_input": data.x_test, "catch22_input": c22_test}, verbose=0),
    )
    y_true = inverse_transform(data.scaler, data.y_test)
    metrics = regression_metrics(y_true, y_pred)

    print(f"\n--- catch22 LSTM on {args.series} ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    figures_dir = PROJECT_ROOT / "reports" / "figures"
    tables_dir = PROJECT_ROOT / "reports" / "tables"
    models_dir = PROJECT_ROOT / "artifacts" / "models"
    for directory in (figures_dir, tables_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="real", linewidth=2)
    plt.plot(y_pred, label="catch22 LSTM", linewidth=2)
    plt.title(f"catch22 LSTM forecast vs real ({args.series})")
    plt.xlabel("test step")
    plt.ylabel("travelers")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    figure_path = figures_dir / f"forecast_catch22_global_{args.series}.png"
    plt.savefig(figure_path, dpi=140)
    plt.close()
    print(f"[saved] {figure_path}")

    results = pd.DataFrame([{"series": args.series, "model": "catch22_lstm", **metrics}])
    results_path = tables_dir / "catch22_global_results.csv"
    results.to_csv(results_path, index=False)
    print(f"[saved] {results_path}")

    model_path = models_dir / f"catch22_lstm_{args.series}.keras"
    model.save(model_path)
    print(f"[saved] {model_path}")

    if best_row is None:
        print("[warn] No reference model available; skipping the head-to-head comparison.")
        return

    comparison = pd.DataFrame(
        [
            {
                "series": args.series,
                "model": f"best_lstm ({best_row['best_family']})",
                "MAE": float(best_row["MAE"]),
                "RMSE": float(best_row["RMSE"]),
                "MAPE": float(best_row["MAPE"]),
            },
            {"series": args.series, "model": "catch22_lstm", **metrics},
        ]
    )
    comparison["MAE_vs_best_pct"] = (
        (comparison["MAE"] / float(best_row["MAE"]) - 1.0) * 100.0
    ).round(2)

    comparison_path = tables_dir / "catch22_vs_best_lstm.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"[saved] {comparison_path}")

    print(f"\n--- catch22 LSTM vs best plain LSTM ({args.series}) ---")
    print(comparison.to_string(index=False))

    winner = "catch22 LSTM" if metrics["MAE"] < float(best_row["MAE"]) else "best plain LSTM"
    print(f"Lower test MAE: {winner}")


if __name__ == "__main__":
    main()
