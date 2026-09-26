"""Load the processed Z24 arrays consistently on local machines and Kaggle."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np


INPUT_FILENAME = "inputs.npy"
LABEL_FILENAME = "labels.npy"
EXPECTED_INPUT_SHAPE = (1530, 27, 6000)
EXPECTED_LABEL_SHAPE = (1530,)
EXPECTED_LABELS = np.arange(17, dtype=np.int64)
SETUPS_PER_CONDITION = 9
SEGMENTS_PER_RECORDING = 10


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_root() -> Path:
    kaggle_working = Path("/kaggle/working")
    return kaggle_working / "results" if kaggle_working.is_dir() else project_root() / "artifacts"


def resolve_data_source(
    inputs_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    archive_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve a direct NPY pair or extract the local ZIP into a reusable cache."""
    if bool(inputs_path) != bool(labels_path):
        raise ValueError("inputs_path and labels_path must be supplied together")
    if inputs_path and archive_path:
        raise ValueError("Use either direct NPY paths or archive_path, not both")

    if inputs_path:
        inputs = Path(inputs_path).expanduser().resolve()
        labels = Path(labels_path).expanduser().resolve()
        _require_file(inputs)
        _require_file(labels)
        return inputs, labels

    kaggle_inputs = Path("/kaggle/input/dataset/inputs.npy")
    kaggle_labels = Path("/kaggle/input/dataset/labels.npy")
    if archive_path is None and kaggle_inputs.is_file() and kaggle_labels.is_file():
        return kaggle_inputs, kaggle_labels

    archive = Path(archive_path) if archive_path else project_root() / "raw_data" / "dataset.zip"
    archive = archive.expanduser().resolve()
    _require_file(archive)
    target_dir = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir
        else project_root() / "processed" / "z24_dataset"
    )
    return extract_dataset_archive(archive, target_dir)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")


def extract_dataset_archive(archive: Path, cache_dir: Path) -> tuple[Path, Path]:
    """Extract only the expected NPY members and reject unsafe/ambiguous ZIPs."""
    with zipfile.ZipFile(archive) as handle:
        file_members = [info for info in handle.infolist() if not info.is_dir()]
        by_basename: dict[str, list[zipfile.ZipInfo]] = {}
        for info in file_members:
            by_basename.setdefault(Path(info.filename).name, []).append(info)

        selected: dict[str, zipfile.ZipInfo] = {}
        for filename in (INPUT_FILENAME, LABEL_FILENAME):
            matches = by_basename.get(filename, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one {filename} in {archive}, found {len(matches)}"
                )
            selected[filename] = matches[0]

        cache_dir.mkdir(parents=True, exist_ok=True)
        for filename, member in selected.items():
            target = cache_dir / filename
            if target.exists() and target.stat().st_size == member.file_size:
                continue
            if target.exists():
                raise ValueError(
                    f"Cached file has an unexpected size: {target}. "
                    "Remove that cache file explicitly and retry."
                )
            partial = target.with_suffix(target.suffix + ".partial")
            if partial.exists():
                raise FileExistsError(f"Incomplete extraction already exists: {partial}")
            with handle.open(member) as source, partial.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
            partial.replace(target)

    return cache_dir / INPUT_FILENAME, cache_dir / LABEL_FILENAME


def load_dataset_arrays(inputs_path: str | Path, labels_path: str | Path):
    """Memory-map inputs, load labels, and enforce the dataset contract."""
    inputs_path = Path(inputs_path)
    labels_path = Path(labels_path)
    inputs = np.load(inputs_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    validate_dataset(inputs, labels)
    return inputs, labels


def validate_dataset(inputs, labels) -> None:
    if inputs.shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(f"Expected inputs {EXPECTED_INPUT_SHAPE}, got {inputs.shape}")
    if inputs.dtype != np.float32:
        raise ValueError(f"Expected float32 inputs, got {inputs.dtype}")
    if labels.shape != EXPECTED_LABEL_SHAPE:
        raise ValueError(f"Expected labels {EXPECTED_LABEL_SHAPE}, got {labels.shape}")
    if labels.dtype != np.int64:
        raise ValueError(f"Expected int64 labels, got {labels.dtype}")

    values, counts = np.unique(labels, return_counts=True)
    expected_count = SETUPS_PER_CONDITION * SEGMENTS_PER_RECORDING
    if not np.array_equal(values, EXPECTED_LABELS):
        raise ValueError(f"Expected labels 0--16, got {values.tolist()}")
    if not np.all(counts == expected_count):
        raise ValueError(f"Expected {expected_count} samples per label, got {counts.tolist()}")
    if not np.array_equal(labels, np.repeat(EXPECTED_LABELS, expected_count)):
        raise ValueError(
            "Labels are not condition-major; setup groups cannot be inferred safely"
        )


def sample_metadata(labels) -> dict[str, np.ndarray]:
    labels = np.asarray(labels)
    sample_index = np.arange(len(labels), dtype=np.int64)
    within_condition = sample_index % (SETUPS_PER_CONDITION * SEGMENTS_PER_RECORDING)
    setup = within_condition // SEGMENTS_PER_RECORDING
    segment = within_condition % SEGMENTS_PER_RECORDING
    recording = labels.astype(np.int64) * SETUPS_PER_CONDITION + setup
    return {
        "condition": labels.astype(np.int64, copy=False),
        "setup": setup,
        "segment": segment,
        "recording": recording,
    }


def split_by_setup(labels, seed=42, train_setups=6, validation_setups=1):
    """Create balanced, recording-disjoint train/validation/test indexes."""
    if train_setups < 1 or validation_setups < 1:
        raise ValueError("train_setups and validation_setups must be positive")
    if train_setups + validation_setups >= SETUPS_PER_CONDITION:
        raise ValueError("At least one of the nine setups must remain for test")

    metadata = sample_metadata(labels)
    setup_order = np.arange(SETUPS_PER_CONDITION, dtype=np.int64)
    np.random.default_rng(seed).shuffle(setup_order)
    setup_split = {
        "train": setup_order[:train_setups],
        "validation": setup_order[train_setups : train_setups + validation_setups],
        "test": setup_order[train_setups + validation_setups :],
    }
    indexes = {
        name: np.flatnonzero(np.isin(metadata["setup"], setup_ids))
        for name, setup_ids in setup_split.items()
    }

    recording_sets = {
        name: set(metadata["recording"][part].tolist()) for name, part in indexes.items()
    }
    names = tuple(recording_sets)
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            if not recording_sets[left].isdisjoint(recording_sets[right]):
                raise RuntimeError(f"Recording leakage between {left} and {right}")
    return indexes, setup_split, metadata


def materialize_split(inputs, labels, indexes, input_layout, chunk_size=64):
    """Copy a split from the memory map into the requested model layout."""
    indexes = np.asarray(indexes, dtype=np.int64)
    if input_layout not in {"time_sensors", "sensors_time"}:
        raise ValueError(f"Unsupported input_layout: {input_layout}")
    shape = (
        (len(indexes), inputs.shape[2], inputs.shape[1])
        if input_layout == "time_sensors"
        else (len(indexes), inputs.shape[1], inputs.shape[2])
    )
    result = np.empty(shape, dtype=np.float32)
    for start in range(0, len(indexes), chunk_size):
        selected = indexes[start : start + chunk_size]
        values = np.asarray(inputs[selected], dtype=np.float32)
        if input_layout == "time_sensors":
            values = values.transpose(0, 2, 1)
        result[start : start + len(selected)] = values
    return result, np.asarray(labels[indexes], dtype=np.int64)


def normalize_from_train(x_train, *other_arrays, input_layout):
    """Apply per-sensor Z-score using statistics calculated from train only."""
    sensor_axis = 2 if input_layout == "time_sensors" else 1
    reduce_axes = tuple(axis for axis in range(3) if axis != sensor_axis)
    mean = x_train.mean(axis=reduce_axes, keepdims=True, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=reduce_axes, keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = []
    for array in (x_train, *other_arrays):
        array -= mean
        array /= std
        normalized.append(array)
    return (*normalized, mean.reshape(-1), std.reshape(-1))


def prepare_splits(config, inputs_path=None, labels_path=None, archive_path=None, cache_dir=None):
    """Resolve, validate, split, materialize, and normalize the Z24 data."""
    resolved_inputs, resolved_labels = resolve_data_source(
        inputs_path, labels_path, archive_path, cache_dir
    )
    inputs, labels = load_dataset_arrays(resolved_inputs, resolved_labels)
    indexes, setup_split, metadata = split_by_setup(
        labels,
        seed=int(config["seed"]),
        train_setups=int(config.get("train_setups", 6)),
        validation_setups=int(config.get("validation_setups", 1)),
    )
    layout = config["input_layout"]
    arrays = {
        name: materialize_split(inputs, labels, part, layout)
        for name, part in indexes.items()
    }
    x_train, y_train = arrays["train"]
    x_validation, y_validation = arrays["validation"]
    x_test, y_test = arrays["test"]
    x_train, x_validation, x_test, sensor_mean, sensor_std = normalize_from_train(
        x_train, x_validation, x_test, input_layout=layout
    )
    return {
        "train": (x_train, y_train),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test),
        "indexes": indexes,
        "setup_split": setup_split,
        "metadata": metadata,
        "sensor_mean": sensor_mean,
        "sensor_std": sensor_std,
        "inputs_path": resolved_inputs,
        "labels_path": resolved_labels,
    }


def dataset_summary(prepared) -> dict:
    return {
        "inputs_path": str(prepared["inputs_path"]),
        "labels_path": str(prepared["labels_path"]),
        "setup_split": {
            name: values.tolist() for name, values in prepared["setup_split"].items()
        },
        "sample_counts": {
            name: int(len(prepared[name][1]))
            for name in ("train", "validation", "test")
        },
        "shapes": {
            name: list(prepared[name][0].shape)
            for name in ("train", "validation", "test")
        },
    }


def write_dataset_summary(prepared, destination: Path) -> None:
    destination.write_text(json.dumps(dataset_summary(prepared), indent=2), encoding="utf-8")
