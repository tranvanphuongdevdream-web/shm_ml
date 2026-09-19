"""Adapted 1DCNN-LSTM-ResNet model for the cleaned Z24 PDT data.

The architecture is based on Le-Xuan Thang et al., Structures 59 (2024),
article 105784, DOI 10.1016/j.istruc.2023.105784. The input convention is
corrected to Keras' ``(time_steps, channels)`` layout.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    LSTM,
)


def residual_block(
    inputs,
    filters=(64, 64, 256),
    kernel_size=3,
    stride=1,
    dilation_rate=1,
    projection=False,
):
    """Return a bottleneck residual block for one-dimensional signals."""
    first, middle, output = filters
    shortcut = inputs

    x = Conv1D(first, 1, strides=stride, padding="valid", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(
        middle,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(output, 1, padding="valid", use_bias=False)(x)
    x = BatchNormalization()(x)

    if projection or stride != 1 or inputs.shape[-1] != output:
        shortcut = Conv1D(output, 1, strides=stride, padding="valid", use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    return Activation("relu")(x)


def build_dcnn_lstm_resnet(
    input_shape,
    num_classes=17,
    lstm_units=128,
    dropout=0.30,
):
    """Build and compile the Z24 condition-classification model.

    Args:
        input_shape: ``(time_steps, channels)``, for example ``(1000, 5)``.
        num_classes: Number of output condition classes.
        lstm_units: Number of LSTM units.
        dropout: Dropout applied before the final classification layer.
    """
    inputs = Input(shape=input_shape, name="sensor_time_series")

    x = Conv1D(64, 7, strides=2, padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = residual_block(x, projection=True)
    dilated = residual_block(x, dilation_rate=2)
    x = Add()([x, dilated])

    sequence = LSTM(lstm_units, return_sequences=True, name="temporal_lstm")(x)

    input_shortcut = Conv1D(lstm_units, 1, strides=2, padding="same", use_bias=False)(inputs)
    input_shortcut = BatchNormalization()(input_shortcut)
    feature_shortcut = Conv1D(lstm_units, 1, padding="same", use_bias=False)(x)
    feature_shortcut = BatchNormalization()(feature_shortcut)

    x = Concatenate(axis=-1)([sequence, input_shortcut, feature_shortcut])
    x = Activation("relu")(x)
    average = GlobalAveragePooling1D()(x)
    maximum = GlobalMaxPooling1D()(x)
    x = Concatenate()([average, maximum])
    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation="softmax", name="condition")(x)

    model = Model(inputs=inputs, outputs=outputs, name="DCNN_LSTM_ResNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
