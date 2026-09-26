"""tsai_001: ResNet using the same split and normalization as DCNN."""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import evaluate_predictions, save_evaluation, save_json


def training_history_frame(values, metric_names):
    """Snapshot fastai's training rows with columns that match their width."""
    rows = [list(row) for row in values]
    columns = list(metric_names[1:-1])
    value_width = max((len(row) for row in rows), default=0)
    if len(columns) != value_width:
        print(
            f"Recorder has {value_width} values per epoch but "
            f"{len(columns)} metric names; using generic names.",
            flush=True,
        )
        columns = [f"metric_{index + 1}" for index in range(value_width)]
    history = pd.DataFrame(rows, columns=columns)
    history.insert(0, "epoch", np.arange(1, len(history) + 1))
    return history


def run(config, prepared, artifact_dir):
    try:
        import torch
        import fastai
        import tsai
        from fastai.callback.core import Callback
        from fastai.metrics import accuracy
        from tsai.data.core import TSClassification
        from tsai.data.validation import combine_split_data
        from tsai.tslearner import TSClassifier
        import tsai.inference  # noqa: F401 - registers inference helpers
    except ImportError as error:
        raise RuntimeError(
            "The tsai experiment needs the optional local/Kaggle tsai dependencies. "
            "See README.md for the installation command."
        ) from error

    artifact_dir = Path(artifact_dir)
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_train, y_train = prepared["train"]
    x_validation, y_validation = prepared["validation"]
    x_test, y_test = prepared["test"]
    x_combined, y_combined, tsai_splits = combine_split_data(
        [x_train, x_validation], [y_train, y_validation]
    )
    learner = TSClassifier(
        x_combined,
        y_combined,
        splits=tsai_splits,
        tfms=[None, TSClassification()],
        arch=config["architecture"],
        metrics=accuracy,
        bs=int(config["batch_size"]),
        wd=float(config["weight_decay"]),
        path=artifact_dir,
        model_dir="models",
        seed=seed,
        device=device,
        verbose=True,
    )
    learner.summary()

    def synchronize():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    class EpochTimer(Callback):
        order = 100

        def before_fit(self):
            self.epoch_seconds = []

        def before_epoch(self):
            synchronize()
            self.started = time.perf_counter()

        def after_epoch(self):
            synchronize()
            self.epoch_seconds.append(time.perf_counter() - self.started)

    timer = EpochTimer()
    synchronize()
    started = time.perf_counter()
    learner.fit_one_cycle(
        int(config["epochs"]), float(config["learning_rate"]), cbs=[timer]
    )
    synchronize()
    training_seconds = time.perf_counter() - started

    # Prediction helpers can change fastai's Recorder state. Capture the
    # training history now, before calling get_X_preds on any split.
    training_history_frame(
        learner.recorder.values, learner.recorder.metric_names
    ).to_csv(artifact_dir / "history.csv", index=False)

    def predict(values, targets):
        probabilities, _, _ = learner.get_X_preds(
            values,
            targets,
            bs=int(config["prediction_batch_size"]),
            with_decoded=True,
        )
        if hasattr(probabilities, "detach"):
            probabilities = probabilities.detach().cpu().numpy()
        return np.asarray(probabilities).argmax(axis=1).astype(np.int64)

    predictions = {
        "train": predict(x_train, y_train),
        "validation": predict(x_validation, y_validation),
        "test": predict(x_test, y_test),
    }
    targets = {"train": y_train, "validation": y_validation, "test": y_test}
    num_classes = int(config["num_classes"])
    metrics, report, matrix = evaluate_predictions(predictions, targets, num_classes)
    print("Final metrics (macro averages):")
    print(metrics.to_string(float_format=lambda value: f"{value:.4f}"))

    learner.export("model.pkl")
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
        "model_parameters": int(sum(parameter.numel() for parameter in learner.model.parameters())),
        "batch_size": int(config["batch_size"]),
        "replicas": 1,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "pytorch_version": torch.__version__,
        "fastai_version": fastai.__version__,
        "tsai_version": tsai.__version__,
        "python_version": platform.python_version(),
        "precision_policy": str(next(learner.model.parameters()).dtype),
        "model_input_shape": list(x_train.shape[1:]),
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
