from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_lstm_catch22(
    input_shape: tuple[int, int],
    catch22_dim: int,
    units: int,
    dropout: float,
    lr: float
) -> keras.Model:
    ts_input = keras.layers.Input(shape=input_shape, name="ts_input")
    c22_input = keras.layers.Input(shape=(catch22_dim,), name="catch22_input")

    lstm_out = keras.layers.LSTM(units)(ts_input)
    lstm_out = keras.layers.Dropout(dropout)(lstm_out)

    c22_out = keras.layers.Dense(16, activation="relu")(c22_input)

    concat = keras.layers.Concatenate()([lstm_out, c22_out])
    dense_out = keras.layers.Dense(16, activation="relu")(concat)
    dense_out = keras.layers.Dropout(dropout)(dense_out)
    out = keras.layers.Dense(1, name="prediction")(dense_out)

    model = keras.Model(inputs=[ts_input, c22_input], outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


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
