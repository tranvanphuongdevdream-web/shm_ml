"""dcnn_002: model architecture, training, prediction, and artifacts."""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Concatenate, Conv1D, Dense, Dropout,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Input, LSTM,
)
from tensorflow.keras.regularizers import l2

from src.evaluation import evaluate_predictions, save_evaluation, save_json


def resnet_block(
    x, filters, kernel_size=3, stride=1, dilation_rate=1,
    use_projection_shortcut=False, use_batch_norm=False,
):
    first_filters, middle_filters, output_filters = filters
    shortcut = x
    x = Conv1D(first_filters, 1, strides=stride, dilation_rate=dilation_rate, padding="valid")(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(middle_filters, kernel_size, dilation_rate=dilation_rate, padding="same")(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(output_filters, 1, padding="valid")(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    if use_projection_shortcut:
        shortcut = Conv1D(
            output_filters, 1, strides=stride, dilation_rate=dilation_rate, padding="valid"
        )(shortcut)
        if use_batch_norm:
            shortcut = BatchNormalization()(shortcut)
    return Activation("relu")(Add()([x, shortcut]))


def build_dcnn_lstm_resnet(input_shape, num_classes, config):
    """Build the shared architecture with behavior selected entirely by config."""
    input_tensor = Input(shape=input_shape)
    x = Conv1D(64, 7, padding="same", strides=2)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = resnet_block(x, [64, 64, 256], use_projection_shortcut=True)
    x_dilated = resnet_block(
        x, [64, 64, 256], use_projection_shortcut=True, dilation_rate=2
    )
    x = Add()([x, x_dilated])

    lstm_output = LSTM(
        128,
        return_sequences=True,
        recurrent_activation=config["recurrent_activation"],
    )(x)
    if float(config.get("dropout_lstm", 0.0)):
        lstm_output = Dropout(float(config["dropout_lstm"]))(lstm_output)

    shortcut1 = Conv1D(lstm_output.shape[-1], 1, padding="same", strides=2)(input_tensor)
    shortcut1 = BatchNormalization()(shortcut1)
    shortcut2 = Conv1D(lstm_output.shape[-1], 1, padding="same")(x)
    shortcut2 = BatchNormalization()(shortcut2)
    x = Concatenate(axis=-1)([lstm_output, shortcut1, shortcut2])
    x = Activation("relu")(x)
    x = Concatenate(axis=-1)([GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)])

    l2_strength = float(config.get("l2", 0.0))
    regularizer = l2(l2_strength) if l2_strength else None
    x = Dense(128, activation="relu", kernel_regularizer=regularizer)(x)
    if float(config.get("dropout_dense", 0.0)):
        x = Dropout(float(config["dropout_dense"]))(x)
    output_tensor = Dense(
        num_classes,
        activation="softmax",
        dtype="float32" if config.get("mixed_precision") else None,
    )(x)

    model = tf.keras.Model(input_tensor, output_tensor)
    optimizer_name = config["optimizer"].lower()
    learning_rate = float(config["learning_rate"])
    if optimizer_name == "adamw":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    elif optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _configure_tensorflow(config):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(config["seed"]))
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    use_mixed = bool(config.get("mixed_precision")) and bool(gpus)
    # clear_session resets Keras global state, so set the policy afterwards.
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if use_mixed else "float32"
    )
    strategy = (
        tf.distribute.MirroredStrategy()
        if config.get("use_all_gpus") and len(gpus) > 1
        else tf.distribute.get_strategy()
    )
    gpu_names = [
        tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name)
        for gpu in gpus
    ]
    return tf, strategy, gpu_names


def _make_tf_datasets(tf, prepared, config):
    batch_size = int(config["batch_size"])
    seed = int(config["seed"])
    x_train, y_train = prepared["train"]
    x_validation, y_validation = prepared["validation"]
    options = tf.data.Options()
    options.experimental_deterministic = False
    train = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .with_options(options)
    )
    validation = (
        tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train, validation


def run(config, prepared, artifact_dir):
    if config["experiment"] != "dcnn_002":
        raise ValueError("dcnn_002.run received the wrong experiment config")
    tf, strategy, gpu_names = _configure_tensorflow(config)
    artifact_dir = Path(artifact_dir)
    x_train, y_train = prepared["train"]
    x_validation, y_validation = prepared["validation"]
    input_shape = tuple(x_train.shape[1:])
    num_classes = int(config["num_classes"])

    with strategy.scope():
        model = build_dcnn_lstm_resnet(input_shape, num_classes, config)
    model.summary()

    class EpochTimer(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.epoch_seconds = []

        def on_epoch_begin(self, epoch, logs=None):
            self.started = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_seconds.append(time.perf_counter() - self.started)

    timer = EpochTimer()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=config["early_stopping_monitor"],
            mode=config["early_stopping_mode"],
            patience=int(config["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            artifact_dir / "best_model.keras",
            monitor=config["early_stopping_monitor"],
            mode=config["early_stopping_mode"],
            save_best_only=True,
        ),
        timer,
    ]

    started = time.perf_counter()
    if config.get("data_pipeline") == "tf_data":
        train_data, validation_data = _make_tf_datasets(tf, prepared, config)
        history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=int(config["epochs"]),
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            batch_size=int(config["batch_size"]),
            epochs=int(config["epochs"]),
            shuffle=True,
            callbacks=callbacks,
            verbose=1,
        )
    training_seconds = time.perf_counter() - started

    predictions = {}
    targets = {}
    for name in ("train", "validation", "test"):
        x_part, y_part = prepared[name]
        probabilities = model.predict(
            x_part, batch_size=int(config["batch_size"]), verbose=0
        )
        predictions[name] = probabilities.argmax(axis=1)
        targets[name] = y_part
    metrics, report, matrix = evaluate_predictions(predictions, targets, num_classes)
    print("Final metrics (macro averages):")
    print(metrics.to_string(float_format=lambda value: f"{value:.4f}"))

    model.save(artifact_dir / "final_model.keras")
    pd.DataFrame(history.history).to_csv(artifact_dir / "history.csv", index=False)
    save_evaluation(artifact_dir, metrics, report, matrix, num_classes)
    np.savez(
        artifact_dir / "preprocessing_and_splits.npz",
        sensor_mean=prepared["sensor_mean"],
        sensor_std=prepared["sensor_std"],
        train_indexes=prepared["indexes"]["train"],
        validation_indexes=prepared["indexes"]["validation"],
        test_indexes=prepared["indexes"]["test"],
    )

    epoch_seconds = np.asarray(timer.epoch_seconds, dtype=np.float64)
    steady = epoch_seconds[1:] if len(epoch_seconds) > 1 else epoch_seconds
    benchmark = {
        "pipeline": config["experiment"],
        "training_seconds_total": float(training_seconds),
        "epochs_completed": int(len(epoch_seconds)),
        "mean_epoch_seconds": float(epoch_seconds.mean()),
        "mean_epoch_seconds_excluding_first": float(steady.mean()),
        "effective_train_samples_per_second": float(len(x_train) / steady.mean()),
        "model_parameters": int(model.count_params()),
        "batch_size": int(config["batch_size"]),
        "replicas": int(strategy.num_replicas_in_sync),
        "gpu_names": gpu_names,
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "precision_policy": str(tf.keras.mixed_precision.global_policy()),
        "model_input_shape": list(input_shape),
        "input_layout": config["input_layout"],
        "normalization": "train_sensor_zscore",
        "optimizer": config["optimizer"],
        "split_strategy": "setup_grouped_6_1_2",
        "metrics": metrics.to_dict(orient="index"),
    }
    save_json(artifact_dir / "benchmark_summary.json", benchmark)
    pd.json_normalize(benchmark, sep=".").to_csv(
        artifact_dir / "benchmark_summary.csv", index=False
    )
    return benchmark
