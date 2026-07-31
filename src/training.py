from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .config import (
    MAX_EPOCHS,
    MODEL_FAMILIES,
    PATIENCE,
    RANDOM_SEED,
    SERIES_TO_MODEL,
    TRAIN_RATIO,
    VALID_RATIO_WITHIN_TRAIN,
)
from .data import build_selected_series, load_clean_data
from .evaluation import regression_metrics
from .modeling import build_lstm_simple, build_lstm_stacked, set_global_seed
from .preprocessing import inverse_transform, prepare_windowed_data, split_train_test, train_val_split


@dataclass
class TrialResult:
    series_name: str
    family: str
    window: int
    units: int
    dropout: float
    lr: float
    batch_size: int
    val_loss: float


@dataclass
class BestModelResult:
    series_name: str
    family: str
    params: dict[str, float | int]
    test_metrics: dict[str, float]
    y_test_real: np.ndarray
    y_pred_real: np.ndarray


def _build_model(family: str, input_shape: tuple[int, int], params: dict) -> keras.Model:
    if family == "lstm_simple":
        return build_lstm_simple(input_shape, units=params["units"], dropout=params["dropout"], lr=params["lr"])
    if family == "lstm_stacked":
        return build_lstm_stacked(input_shape, units=params["units"], dropout=params["dropout"], lr=params["lr"])
    raise ValueError(f"Unknown family: {family}")


def _fit_one_model(
    family: str,
    params: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int,
    patience: int,
) -> tuple[keras.Model, float]:
    model = _build_model(family=family, input_shape=(x_train.shape[1], x_train.shape[2]), params=params)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=max_epochs,
        batch_size=params["batch_size"],
        verbose=0,
        callbacks=callbacks,
    )

    best_val = float(np.min(history.history["val_loss"]))
    return model, best_val


def _retrain_for_test(
    family: str,
    params: dict,
    train_full: pd.Series,
    val: pd.Series,
    test: pd.Series,
    max_epochs: int,
    patience: int,
) -> tuple[keras.Model, np.ndarray, np.ndarray]:
    data = prepare_windowed_data(train_full, val, test, window=params["window"])

    model = _build_model(family=family, input_shape=(data.x_train.shape[1], data.x_train.shape[2]), params=params)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    ]

    model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_val, data.y_val),
        epochs=max_epochs,
        batch_size=params["batch_size"],
        verbose=0,
        callbacks=callbacks,
    )

    y_pred_scaled = model.predict(data.x_test, verbose=0).flatten()
    y_test_scaled = data.y_test.flatten()

    y_pred_real = inverse_transform(data.scaler, y_pred_scaled)
    y_test_real = inverse_transform(data.scaler, y_test_scaled)

    return model, y_test_real, y_pred_real


def tune_and_train_for_series(
    series_name: str,
    series: pd.Series,
    output_dir: Path,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> tuple[list[TrialResult], BestModelResult]:
    print("=" * 72)
    print(f"Series: {series_name}")
    print("=" * 72)

    train_full, test = split_train_test(series, TRAIN_RATIO)
    train_sub, val = train_val_split(train_full, VALID_RATIO_WITHIN_TRAIN)

    trial_results: list[TrialResult] = []

    for family, configs in MODEL_FAMILIES.items():
        print(f"[tuning] Family: {family}")
        for params in configs:
            try:
                data = prepare_windowed_data(train_sub, val, test, window=params["window"])
            except ValueError as exc:
                print(f"  Skipped params {params}: {exc}")
                continue

            if len(data.x_train) == 0 or len(data.x_val) == 0:
                print(f"  Skipped params due to insufficient data: {params}")
                continue

            _, best_val = _fit_one_model(
                family=family,
                params=params,
                x_train=data.x_train,
                y_train=data.y_train,
                x_val=data.x_val,
                y_val=data.y_val,
                max_epochs=max_epochs,
                patience=patience,
            )

            trial = TrialResult(
                series_name=series_name,
                family=family,
                window=params["window"],
                units=params["units"],
                dropout=params["dropout"],
                lr=params["lr"],
                batch_size=params["batch_size"],
                val_loss=best_val,
            )
            trial_results.append(trial)
            print(f"  Params {params} -> best val_loss={best_val:.6f}")

    if not trial_results:
        print(f"Warning: No successful trials for series {series_name}. Skipping.")
        return [], None

    best_trial = min(trial_results, key=lambda t: t.val_loss)
    best_params = {
        "window": best_trial.window,
        "units": best_trial.units,
        "dropout": best_trial.dropout,
        "lr": best_trial.lr,
        "batch_size": best_trial.batch_size,
    }

    print(f"[best] {best_trial.family} with params={best_params} val_loss={best_trial.val_loss:.6f}")

    best_model, y_test_real, y_pred_real = _retrain_for_test(
        family=best_trial.family,
        params=best_params,
        train_full=train_sub,
        val=val,
        test=test,
        max_epochs=max_epochs,
        patience=patience,
    )

    metrics = regression_metrics(y_test_real, y_pred_real)
    print(
        "[test] "
        f"MAE={metrics['MAE']:.2f} | RMSE={metrics['RMSE']:.2f} | MAPE={metrics['MAPE']:.2f}%"
    )

    model_dir = output_dir / "artifacts" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"best_{series_name}.keras"
    best_model.save(model_path)

    return trial_results, BestModelResult(
        series_name=series_name,
        family=best_trial.family,
        params=best_params,
        test_metrics=metrics,
        y_test_real=y_test_real,
        y_pred_real=y_pred_real,
    )


def _save_trials(trials: list[TrialResult], output_dir: Path) -> None:
    rows = [t.__dict__ for t in trials]
    df = pd.DataFrame(rows).sort_values(["series_name", "val_loss"])
    out = output_dir / "reports" / "tables" / "tuning_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[saved] {out}")


def _save_best_summary(results: list[BestModelResult], output_dir: Path) -> None:
    rows = []
    for item in results:
        rows.append(
            {
                "series_name": item.series_name,
                "best_family": item.family,
                "params": item.params,
                "MAE": item.test_metrics["MAE"],
                "RMSE": item.test_metrics["RMSE"],
                "MAPE": item.test_metrics["MAPE"],
            }
        )
    df = pd.DataFrame(rows).sort_values("MAE")
    out = output_dir / "reports" / "tables" / "best_models_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[saved] {out}")


def _save_forecast_figure(result: BestModelResult, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(result.y_test_real, label="real", linewidth=2)
    ax.plot(result.y_pred_real, label="pred", linewidth=2)
    ax.set_title(f"Best LSTM forecast - {result.series_name}")
    ax.set_xlabel("test step")
    ax.set_ylabel("travelers")
    ax.legend()
    ax.grid(alpha=0.3)

    out = output_dir / "reports" / "figures" / f"forecast_{result.series_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[saved] {out}")


def run(output_dir: Path, series_to_model: list[str] | None = None, max_epochs: int = MAX_EPOCHS) -> None:
    set_global_seed(RANDOM_SEED)
    tf.get_logger().setLevel("ERROR")

    if series_to_model is None:
        series_to_model = SERIES_TO_MODEL

    df = load_clean_data()
    selected = build_selected_series(df)

    all_trials: list[TrialResult] = []
    best_results: list[BestModelResult] = []

    for series_name in series_to_model:
        if series_name not in selected:
            raise KeyError(f"Series '{series_name}' not found in selected series.")

        trials, best = tune_and_train_for_series(
            series_name=series_name,
            series=selected[series_name],
            output_dir=output_dir,
            max_epochs=max_epochs,
            patience=PATIENCE,
        )
        if best is not None:
            all_trials.extend(trials)
            best_results.append(best)
            _save_forecast_figure(best, output_dir)

    _save_trials(all_trials, output_dir)
    _save_best_summary(best_results, output_dir)
