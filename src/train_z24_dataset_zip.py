"""Train the Z24 classifier directly from ``raw_data/dataset.zip``.

This is the primary entry point for the processed Hugging Face dataset.  It
uses setup-grouped splits and the standard Keras Conv1D layout
``[sample, time, sensor]``.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from src.data.z24_dataset_zip import (
    materialize_split,
    normalize_from_train,
    prepare_dataset_zip,
    load_dataset_arrays,
    split_by_setup,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Validate/extract dataset.zip and show the split without training.",
    )
    return parser.parse_args()


def project_root():
    return Path(__file__).resolve().parents[1]


def main():
    args = parse_args()
    root = project_root()

    cache_dir = prepare_dataset_zip(root, args.archive)
    inputs, labels = load_dataset_arrays(cache_dir)
    splits, setup_split, metadata = split_by_setup(labels, seed=args.seed)

    print(f"Archive cache: {cache_dir}")
    print(f"Stored inputs: {inputs.shape} {inputs.dtype} [samples, sensors, time]")
    for name, indexes in splits.items():
        recording_count = len(set(metadata["recording"][indexes].tolist()))
        print(
            f"{name:10s}: {len(indexes):4d} segments, "
            f"{recording_count:3d} recordings, setups={setup_split[name].tolist()}"
        )

    if args.cache_only:
        return

    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    from src.models.dcnn_lstm_resnet import build_model

    tf.keras.utils.set_random_seed(args.seed)

    arrays = {}
    for name in ("train", "validation", "test"):
        arrays[name] = materialize_split(
            inputs, labels, splits[name], channels_last=True
        )
        print(
            f"Loaded {name:10s}: X={arrays[name][0].shape}, "
            f"memory={arrays[name][0].nbytes / 1024**2:.1f} MiB"
        )

    x_train, y_train = arrays["train"]
    x_validation, y_validation = arrays["validation"]
    x_test, y_test = arrays["test"]
    x_train, x_validation, x_test, sensor_mean, sensor_std = normalize_from_train(
        x_train, x_validation, x_test, channels_last=True
    )

    num_classes = len(np.unique(labels))
    input_shape = x_train.shape[1:]
    model = build_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    artifact_dir = root / "artifacts" / "dataset_zip_dcnn_lstm_resnet" / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            artifact_dir / "best_model.keras",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, batch_size=args.batch_size, verbose=0
    )
    probabilities = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(num_classes),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=np.arange(num_classes))

    model.save(artifact_dir / "final_model.keras")
    pd.DataFrame(history.history).to_csv(artifact_dir / "history.csv", index=False)
    pd.DataFrame(report).T.to_csv(artifact_dir / "classification_report.csv")
    np.savetxt(artifact_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    np.savez(
        artifact_dir / "preprocessing.npz",
        sensor_mean=sensor_mean,
        sensor_std=sensor_std,
        train_indexes=splits["train"],
        validation_indexes=splits["validation"],
        test_indexes=splits["test"],
    )

    experiment = {
        "archive": str((args.archive or root / "raw_data" / "dataset.zip").resolve()),
        "stored_shape": list(inputs.shape),
        "model_input_shape": list(input_shape),
        "model_input_layout": "time_samples x sensors",
        "num_classes": num_classes,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "maximum_epochs": args.epochs,
        "completed_epochs": len(history.history["loss"]),
        "setup_split": {name: values.tolist() for name, values in setup_split.items()},
        "segment_counts": {name: len(values) for name, values in splits.items()},
        "sensor_mean": sensor_mean.tolist(),
        "sensor_std": sensor_std.tolist(),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    (artifact_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()
