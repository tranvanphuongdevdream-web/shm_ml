"""Regularized 1DCNN-LSTM-ResNet model for Z24 time-series classification.

The residual and recurrent blocks follow the teacher's architecture. The
training version adds dropout, L2 regularization, and AdamW to reduce
overfitting on the recording-level Z24 split. Keras input is time-major:
``(time_samples, sensor_channels)``.
"""

from __future__ import annotations

import tensorflow as tf
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
from tensorflow.keras.regularizers import l2


def resnet_block(
    x,
    filters,
    kernel_size=3,
    stride=1,
    dilation_rate=1,
    use_projection_shortcut=False,
    use_layer_norm=False,
):
    """Build the ResNet block used in the teacher's source code."""
    first_filters, middle_filters, output_filters = filters
    shortcut = x

    x = Conv1D(
        filters=first_filters,
        kernel_size=1,
        strides=stride,
        dilation_rate=dilation_rate,
        padding="valid",
    )(x)
    if use_layer_norm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(
        filters=middle_filters,
        kernel_size=kernel_size,
        strides=1,
        dilation_rate=dilation_rate,
        padding="same",
    )(x)
    if use_layer_norm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(
        filters=output_filters,
        kernel_size=1,
        strides=1,
        dilation_rate=dilation_rate,
        padding="valid",
    )(x)
    if use_layer_norm:
        x = BatchNormalization()(x)

    if use_projection_shortcut:
        shortcut = Conv1D(
            filters=output_filters,
            kernel_size=1,
            strides=stride,
            dilation_rate=dilation_rate,
            padding="valid",
        )(shortcut)
        if use_layer_norm:
            shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    return Activation("relu")(x)


def build_model(input_shape, num_classes):
    """Build and compile the regularized model for the selected Z24 data."""
    input_tensor = Input(shape=input_shape)

    x = Conv1D(filters=64, kernel_size=7, padding="same", strides=2)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = resnet_block(
        x,
        [64, 64, 256],
        use_projection_shortcut=True,
    )
    x_dilated = resnet_block(
        x,
        [64, 64, 256],
        use_projection_shortcut=True,
        dilation_rate=2,
    )
    x = Add()([x, x_dilated])

    lstm = LSTM(
        128,
        return_sequences=True,
        recurrent_activation="softmax",
    )(x)
    lstm = Dropout(0.30)(lstm)

    shortcut1 = Conv1D(
        filters=lstm.shape[-1],
        kernel_size=1,
        padding="same",
        strides=2,
    )(input_tensor)
    shortcut1 = BatchNormalization()(shortcut1)
    shortcut2 = Conv1D(
        filters=lstm.shape[-1],
        kernel_size=1,
        padding="same",
        strides=1,
    )(x)
    shortcut2 = BatchNormalization()(shortcut2)

    x = Concatenate(axis=-1)([lstm, shortcut1, shortcut2])
    x = Activation("relu")(x)

    x_avg = GlobalAveragePooling1D()(x)
    x_max = GlobalMaxPooling1D()(x)
    x = Concatenate(axis=-1)([x_avg, x_max])

    x = Dense(
        128,
        activation="relu",
        kernel_regularizer=l2(1e-4),
    )(x)
    x = Dropout(0.50)(x)
    output_tensor = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-4,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Backward-compatible name used by earlier versions of this project.
build_dcnn_lstm_resnet = build_model
