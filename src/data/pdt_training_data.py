"""Load cleaned PDT CSV files without leaking recordings across splits."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def latest_complete_run(project_root, measurement="avt"):
    """Return the newest timestamped run whose report status is complete."""
    parent = Path(project_root) / "processed" / measurement
    candidates = []
    for path in parent.glob("*"):
        report_path = path / "report.json"
        manifest_path = path / "manifest.json"
        if not report_path.exists() or not manifest_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No complete {measurement.upper()} run found under {parent}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_manifest(run_dir):
    return json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))


def common_channels(records):
    """Return channel names present in every recording, preserving first-record order."""
    if not records:
        raise ValueError("Manifest is empty")
    common = set(records[0]["channel_names"])
    for row in records[1:]:
        common.intersection_update(row["channel_names"])
    ordered = tuple(name for name in records[0]["channel_names"] if name in common)
    if not ordered:
        raise ValueError("No sensor channels are shared by all recordings")
    return ordered


def split_by_recording(records, seed=42, validation_fraction=0.15, test_fraction=0.20):
    """Create stratified splits while keeping each source recording isolated."""
    sources_by_label = defaultdict(set)
    for row in records:
        sources_by_label[int(row["label"])].add(row["source"])

    rng = np.random.default_rng(seed)
    source_split = {}
    for label, sources in sorted(sources_by_label.items()):
        sources = np.array(sorted(sources), dtype=object)
        rng.shuffle(sources)
        count = len(sources)
        validation_count = max(1, int(round(count * validation_fraction)))
        test_count = max(1, int(round(count * test_fraction)))
        train_count = count - validation_count - test_count
        if train_count < 1:
            raise ValueError(f"Label {label} has too few recordings for three splits")
        for source in sources[:train_count]:
            source_split[source] = "train"
        for source in sources[train_count : train_count + validation_count]:
            source_split[source] = "validation"
        for source in sources[train_count + validation_count :]:
            source_split[source] = "test"

    splits = {"train": [], "validation": [], "test": []}
    for row in records:
        splits[source_split[row["source"]]].append(row)

    groups = {name: {row["source"] for row in rows} for name, rows in splits.items()}
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    return splits


def _portable_relative_path(value):
    return Path(*str(value).replace("\\", "/").split("/"))


def count_complete_windows(records, window_samples=8000):
    """Count complete windows after joining CSV segments by source recording."""
    samples_by_source = defaultdict(int)
    for row in records:
        samples_by_source[row["source"]] += int(row["samples"])
    return sum(samples // window_samples for samples in samples_by_source.values())


def load_windows(run_dir, records, channel_names, window_samples=8000):
    """Join each recording's CSV segments, then return fixed-length windows."""
    run_dir = Path(run_dir)
    windows, labels = [], []
    rows_by_source = defaultdict(list)
    for row in records:
        rows_by_source[row["source"]].append(row)

    for source, source_rows in sorted(rows_by_source.items()):
        source_rows.sort(key=lambda row: (int(row["start_sample"]), int(row["segment"])))
        source_labels = {int(row["label"]) for row in source_rows}
        if len(source_labels) != 1:
            raise ValueError(f"Recording has inconsistent labels: {source}")

        recording_parts = []
        for row in source_rows:
            missing = [name for name in channel_names if name not in row["channel_names"]]
            if missing:
                raise ValueError(f"Recording {source} is missing channels: {missing}")
            indexes = [row["channel_names"].index(name) + 1 for name in channel_names]
            path = run_dir / _portable_relative_path(row["file"])
            values = np.loadtxt(
                path,
                delimiter=",",
                skiprows=1,
                usecols=indexes,
                dtype=np.float32,
            )
            if values.ndim == 1:
                values = values[:, None]
            if len(values) != int(row["samples"]):
                raise ValueError(
                    f"Manifest reports {row['samples']} samples but {path} has {len(values)}"
                )
            recording_parts.append(values)

        values = np.concatenate(recording_parts, axis=0)
        window_count = len(values) // window_samples
        if window_count == 0:
            continue
        values = values[: window_count * window_samples]
        windows.append(values.reshape(window_count, window_samples, len(channel_names)))
        labels.append(np.full(window_count, source_labels.pop(), dtype=np.int64))

    if not windows:
        raise RuntimeError("No complete training windows were loaded")
    return np.concatenate(windows), np.concatenate(labels)


def to_tsai_format(x):
    """Convert [samples, timesteps, channels] to tsai's [samples, variables, timesteps]."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected a 3-D time-series array, got shape {x.shape}")
    return np.ascontiguousarray(x.transpose(0, 2, 1))


def normalize_from_train(x_train, *other_arrays):
    """Normalize all arrays using per-channel statistics from training only."""
    mean = x_train.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = []
    for array in (x_train, *other_arrays):
        array -= mean
        array /= std
        normalized.append(array)
    return (*normalized, mean.reshape(-1), std.reshape(-1))
