"""Compare completed training runs without loading models or the Z24 arrays."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_IDS = ("dcnn_001", "dcnn_002", "tsai_001")
SPLITS = ("train", "validation", "test")
METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


@dataclass
class CompletedRun:
    experiment: str
    source: Path
    metadata: dict
    dataset: dict
    benchmark: dict
    metrics: pd.DataFrame


def _read_text(source: Path, filename: str) -> str:
    if source.is_dir():
        return (source / filename).read_text(encoding="utf-8")
    with zipfile.ZipFile(source) as archive:
        return archive.read(filename).decode("utf-8")


def _candidate_sources(roots):
    seen = set()
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root] if root.suffix.lower() == ".zip" else []
        else:
            candidates = [path.parent for path in root.rglob("experiment.json")]
            candidates.extend(
                path for path in root.rglob("*.zip")
                if path.name.startswith(tuple(f"{name}_" for name in EXPERIMENT_IDS))
                and not path.with_suffix("").is_dir()
            )
        for source in candidates:
            resolved = source.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def _load_completed_run(source: Path) -> CompletedRun:
    metadata = json.loads(_read_text(source, "experiment.json"))
    if metadata.get("status") != "completed" or metadata.get("action") != "train":
        raise ValueError("not a completed full-training run")
    experiment = metadata["config"]["experiment"]
    if experiment not in EXPERIMENT_IDS:
        raise ValueError(f"unknown experiment {experiment!r}")
    dataset = json.loads(_read_text(source, "dataset.json"))
    benchmark = json.loads(_read_text(source, "benchmark_summary.json"))
    if benchmark.get("pipeline") != experiment:
        raise ValueError("benchmark experiment does not match config")
    metrics = pd.read_csv(io.StringIO(_read_text(source, "split_metrics.csv")))
    required = {"split", *METRICS}
    if not required.issubset(metrics.columns) or set(metrics["split"]) != set(SPLITS):
        raise ValueError("split_metrics.csv has missing splits or metric columns")
    metrics = metrics.set_index("split").loc[list(SPLITS), list(METRICS)].astype(float)
    if not np.isfinite(metrics.to_numpy()).all():
        raise ValueError("split_metrics.csv contains non-finite scores")
    return CompletedRun(experiment, source, metadata, dataset, benchmark, metrics)


def _run_sort_key(source: Path):
    """Prefer the run timestamp over upload/copy modification time."""
    stem = source.stem if source.is_file() else source.name
    for experiment in EXPERIMENT_IDS:
        prefix = f"{experiment}_"
        if stem.startswith(prefix):
            try:
                stamp = datetime.strptime(stem[len(prefix):], "%d-%m-%y_%H-%M-%S")
                return (1, stamp, source.stat().st_mtime)
            except ValueError:
                break
    return (0, datetime.fromtimestamp(source.stat().st_mtime), source.stat().st_mtime)


def select_latest_runs(roots):
    """Choose one completed train run per experiment; ignore failed/smoke runs."""
    selected = {}
    skipped = []
    candidates = sorted(_candidate_sources(roots), key=_run_sort_key, reverse=True)
    for source in candidates:
        try:
            run = _load_completed_run(source)
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
            skipped.append(f"{source}: {error}")
            continue
        selected.setdefault(run.experiment, run)
    return [selected[name] for name in EXPERIMENT_IDS if name in selected], skipped


def _check_same_evaluation(runs):
    first = runs[0]
    reference = (
        first.metadata["config"]["seed"],
        first.metadata["config"]["num_classes"],
        first.dataset["setup_split"],
        first.dataset["sample_counts"],
        first.benchmark.get("normalization"),
    )
    for run in runs[1:]:
        candidate = (
            run.metadata["config"]["seed"],
            run.metadata["config"]["num_classes"],
            run.dataset["setup_split"],
            run.dataset["sample_counts"],
            run.benchmark.get("normalization"),
        )
        if candidate != reference:
            raise ValueError(
                f"Cannot compare {first.experiment} with {run.experiment}: "
                "seed, classes, setup split, sample counts, or normalization differ"
            )


def _save_bar_charts(
    runs, directory: Path, *, score_split, score_metrics, split_metric, speed_metric
):
    names = [run.experiment for run in runs]
    positions = np.arange(len(runs))
    width = min(0.8 / len(score_metrics), 0.25)
    chart_paths = {}

    fig, axis = plt.subplots(figsize=(max(8, 2.5 * len(runs)), 5))
    for index, metric in enumerate(score_metrics):
        scores = [run.metrics.loc[score_split, metric] for run in runs]
        axis.bar(
            positions + (index - (len(score_metrics) - 1) / 2) * width,
            scores, width, label=metric,
        )
    axis.set_xticks(positions, names)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title(f"{score_split.title()}-set classification metrics (macro averages)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    chart_paths["Selected metrics"] = directory / "selected_metrics.png"
    fig.savefig(chart_paths["Selected metrics"], dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(max(8, 2.5 * len(runs)), 5))
    for index, split in enumerate(SPLITS):
        scores = [run.metrics.loc[split, split_metric] for run in runs]
        axis.bar(positions + (index - 1) * 0.25, scores, 0.25, label=split)
    axis.set_xticks(positions, names)
    axis.set_ylim(0, 1)
    axis.set_ylabel(split_metric)
    axis.set_title(f"Train / validation / test {split_metric}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    chart_paths["Metric by split"] = directory / "metric_by_split.png"
    fig.savefig(chart_paths["Metric by split"], dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for run in runs:
        seconds = float(run.benchmark["mean_epoch_seconds_excluding_first"])
        score = float(run.metrics.loc[score_split, speed_metric])
        if seconds <= 0:
            raise ValueError(f"Invalid mean epoch time in {run.source}")
        axis.scatter(seconds, score, s=100, label=run.experiment)
    axis.set_xscale("log")
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean epoch time after first epoch (seconds, log scale)")
    axis.set_ylabel(f"{score_split} {speed_metric}")
    axis.set_title("Training speed vs. test quality")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    chart_paths["Speed vs quality"] = directory / "speed_vs_quality.png"
    fig.savefig(chart_paths["Speed vs quality"], dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_paths


def build_comparison_report(
    roots, output_root, *, experiment_ids=None, score_split="test",
    score_metrics=None, split_metric="accuracy", speed_metric="f1_macro",
):
    """Write comparison tables/charts and return their data for notebook display."""
    experiment_ids = tuple(EXPERIMENT_IDS if experiment_ids is None else experiment_ids)
    score_metrics = tuple(METRICS if score_metrics is None else score_metrics)
    if not experiment_ids or len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("experiment_ids must contain unique experiment IDs")
    if not score_metrics or len(set(score_metrics)) != len(score_metrics):
        raise ValueError("score_metrics must contain unique metrics")
    for experiment in experiment_ids:
        if experiment not in EXPERIMENT_IDS:
            raise ValueError(f"Unknown experiment {experiment!r}; choose from {EXPERIMENT_IDS}")
    if score_split not in SPLITS:
        raise ValueError(f"Unknown score_split {score_split!r}; choose from {SPLITS}")
    for metric in (*score_metrics, split_metric, speed_metric):
        if metric not in METRICS:
            raise ValueError(f"Unknown metric {metric!r}; choose from {METRICS}")

    available_runs, skipped = select_latest_runs(roots)
    by_experiment = {run.experiment: run for run in available_runs}
    runs = [by_experiment[name] for name in experiment_ids if name in by_experiment]
    missing_experiments = [name for name in experiment_ids if name not in by_experiment]
    if not runs:
        return None, skipped
    _check_same_evaluation(runs)

    metric_rows = []
    benchmark_rows = []
    for run in runs:
        for split in SPLITS:
            metric_rows.append({
                "experiment": run.experiment,
                "split": split,
                **run.metrics.loc[split].to_dict(),
            })
        gpu_names = run.benchmark.get("gpu_names") or []
        if isinstance(gpu_names, str):
            gpu_names = [gpu_names]
        benchmark_rows.append({
            "experiment": run.experiment,
            "training_minutes": float(run.benchmark["training_seconds_total"]) / 60,
            "epochs_completed": int(run.benchmark["epochs_completed"]),
            "mean_epoch_seconds": float(run.benchmark["mean_epoch_seconds_excluding_first"]),
            "train_samples_per_second": float(run.benchmark["effective_train_samples_per_second"]),
            "model_parameters": int(run.benchmark["model_parameters"]),
            "train_test_accuracy_gap": float(
                run.metrics.loc["train", "accuracy"] - run.metrics.loc["test", "accuracy"]
            ),
            "gpu": " | ".join(gpu_names) if gpu_names else "CPU",
            "run_source": str(run.source),
        })
    quality = pd.DataFrame(metric_rows).set_index(["experiment", "split"])
    efficiency = pd.DataFrame(benchmark_rows).set_index("experiment")
    selected_quality = quality.xs(score_split, level="split").loc[:, list(score_metrics)]

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / f"comparison_{datetime.now():%d-%m-%y_%H-%M-%S}"
    directory = base
    suffix = 2
    while directory.exists():
        directory = Path(f"{base}_{suffix}")
        suffix += 1
    directory.mkdir()
    quality.to_csv(directory / "quality_by_split.csv")
    selected_quality.to_csv(directory / "selected_quality.csv")
    efficiency.to_csv(directory / "training_benchmark.csv")
    (directory / "selected_runs.json").write_text(
        json.dumps({run.experiment: str(run.source) for run in runs}, indent=2),
        encoding="utf-8",
    )
    (directory / "comparison_config.json").write_text(
        json.dumps({
            "experiment_ids": experiment_ids,
            "score_split": score_split,
            "score_metrics": score_metrics,
            "split_metric": split_metric,
            "speed_metric": speed_metric,
        }, indent=2),
        encoding="utf-8",
    )
    charts = _save_bar_charts(
        runs, directory, score_split=score_split, score_metrics=score_metrics,
        split_metric=split_metric, speed_metric=speed_metric,
    )
    return {
        "quality": quality,
        "selected_quality": selected_quality,
        "efficiency": efficiency,
        "charts": charts,
        "directory": directory,
        "experiments": [run.experiment for run in runs],
        "missing_experiments": missing_experiments,
        "same_gpu": efficiency["gpu"].nunique() == 1,
    }, skipped


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Compare completed Z24 training runs.")
    parser.add_argument("--results", default="artifacts", help="result/output directory")
    parser.add_argument(
        "--input-root", action="append", default=[],
        help="optional directory containing previous run folders or ZIPs",
    )
    args = parser.parse_args(argv)
    report, skipped = build_comparison_report(
        [args.results, *args.input_root], args.results
    )
    if report is None:
        print("No completed full-training runs found.")
        return
    print("Compared:", ", ".join(report["experiments"]))
    print("CSV and PNG files:", report["directory"])
    if skipped:
        print(f"Skipped {len(skipped)} failed, smoke, or unreadable run(s).")


if __name__ == "__main__":
    main()
