"""Shared metrics and artifact helpers for every experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)


def calculate_split_metrics(targets, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def evaluate_predictions(predictions_by_split, targets_by_split, num_classes):
    rows = []
    for split_name in ("train", "validation", "test"):
        row = {"split": split_name}
        row.update(calculate_split_metrics(
            np.asarray(targets_by_split[split_name]),
            np.asarray(predictions_by_split[split_name]),
        ))
        rows.append(row)
    metrics = pd.DataFrame(rows).set_index("split")
    test_targets = np.asarray(targets_by_split["test"])
    test_predictions = np.asarray(predictions_by_split["test"])
    report = classification_report(
        test_targets, test_predictions, labels=np.arange(num_classes),
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(
        test_targets, test_predictions, labels=np.arange(num_classes)
    )
    return metrics, report, matrix


def save_evaluation(artifact_dir, metrics, report, matrix, num_classes):
    artifact_dir = Path(artifact_dir)
    metrics.to_csv(artifact_dir / "split_metrics.csv")
    pd.DataFrame(report).T.to_csv(artifact_dir / "test_classification_report.csv")
    np.savetxt(artifact_dir / "test_confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    fig, ax = plt.subplots(figsize=(12, 12))
    ConfusionMatrixDisplay(matrix, display_labels=np.arange(num_classes)).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Test confusion matrix")
    fig.tight_layout()
    fig.savefig(artifact_dir / "test_confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_training_charts(artifact_dir):
    """Save notebook-ready charts before the run is packed into a ZIP."""
    artifact_dir = Path(artifact_dir)
    history = pd.read_csv(artifact_dir / "history.csv")
    if not history.empty:
        epochs = history["epoch"] if "epoch" in history else range(1, len(history) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for axis, title, columns in (
            (axes[0], "Loss by epoch", ("loss", "val_loss", "train_loss", "valid_loss")),
            (axes[1], "Accuracy by epoch", ("accuracy", "val_accuracy")),
        ):
            available = [column for column in columns if column in history]
            for column in available:
                axis.plot(epochs, history[column], label=column)
            if available:
                axis.set_title(title)
                axis.set_xlabel("Epoch")
                axis.grid(True, alpha=0.3)
                axis.legend()
            else:
                axis.set_visible(False)
        fig.tight_layout()
        fig.savefig(artifact_dir / "learning_curves.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    metrics = pd.read_csv(artifact_dir / "split_metrics.csv", index_col="split")
    axis = metrics[["accuracy", "f1_macro"]].plot.bar(figsize=(8, 4), rot=0)
    axis.set_title("Performance by split")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3)
    axis.figure.tight_layout()
    axis.figure.savefig(artifact_dir / "split_performance.png", dpi=160, bbox_inches="tight")
    plt.close(axis.figure)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")
