from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lstm_simple(input_shape: tuple[int, int], units: int, dropout: float, lr: float) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(units),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def build_lstm_stacked(input_shape: tuple[int, int], units: int, dropout: float, lr: float) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(units, return_sequences=True),
            keras.layers.Dropout(dropout),
            keras.layers.LSTM(max(16, units // 2)),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model
