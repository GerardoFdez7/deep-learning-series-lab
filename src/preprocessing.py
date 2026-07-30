from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class WindowedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler


def split_train_test(series: pd.Series, train_ratio: float) -> tuple[pd.Series, pd.Series]:
    n_train = int(len(series) * train_ratio)
    train = series.iloc[:n_train].copy()
    test = series.iloc[n_train:].copy()
    return train, test


def train_val_split(train: pd.Series, valid_ratio_within_train: float) -> tuple[pd.Series, pd.Series]:
    n_train_sub = int(len(train) * (1.0 - valid_ratio_within_train))
    train_sub = train.iloc[:n_train_sub].copy()
    val = train.iloc[n_train_sub:].copy()
    return train_sub, val


def make_windows(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(window, len(values)):
        x.append(values[i - window : i])
        y.append(values[i])

    x_arr = np.array(x)
    y_arr = np.array(y)

    if x_arr.size == 0:
        return np.empty((0, window, 1)), np.empty((0, 1))

    x_arr = x_arr.reshape((x_arr.shape[0], x_arr.shape[1], 1))
    y_arr = y_arr.reshape((-1, 1))
    return x_arr, y_arr


def prepare_windowed_data(
    train_sub: pd.Series,
    val: pd.Series,
    test: pd.Series,
    window: int,
) -> WindowedData:
    if len(train_sub) == 0:
        raise ValueError("train_sub has 0 samples. Check series construction and split settings.")
    if len(train_sub) <= window:
        raise ValueError(
            f"train_sub length ({len(train_sub)}) must be greater than window ({window})."
        )
    if len(val) == 0 or len(test) == 0:
        raise ValueError("val/test split is empty. Check train/test ratio and series length.")

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))

    train_sub_values = train_sub.values.reshape(-1, 1)
    val_values = val.values.reshape(-1, 1)
    test_values = test.values.reshape(-1, 1)

    scaled_train_sub = scaler.fit_transform(train_sub_values).flatten()

    # Validation and test use recent history from previous split to preserve temporal continuity.
    val_context = np.concatenate([scaled_train_sub[-window:], scaler.transform(val_values).flatten()])
    test_context = np.concatenate([val_context[-window:], scaler.transform(test_values).flatten()])

    x_train, y_train = make_windows(scaled_train_sub, window)
    x_val, y_val = make_windows(val_context, window)
    x_test, y_test = make_windows(test_context, window)

    return WindowedData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        scaler=scaler,
    )


def inverse_transform(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()
