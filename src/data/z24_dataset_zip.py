"""Load and split the processed Z24 ``dataset.zip`` archive safely.

The archive contains two NumPy files:

* ``inputs.npy`` with layout ``[sample, sensor, time]``;
* ``labels.npy`` with one condition label per sample.

The published dataset contains ten consecutive segments for each
condition/setup recording.  Splits are therefore made by setup group, never by
individual segment, so neighboring segments cannot leak across train,
validation, and test sets.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np


INPUT_MEMBER = "inputs.npy"
LABEL_MEMBER = "labels.npy"
EXPECTED_INPUT_SHAPE = (1530, 27, 6000)
EXPECTED_LABEL_SHAPE = (1530,)
EXPECTED_LABELS = np.arange(17, dtype=np.int64)
SETUPS_PER_CONDITION = 9
SEGMENTS_PER_RECORDING = 10


def _validate_archive(archive: Path) -> dict[str, zipfile.ZipInfo]:
    if not archive.is_file():
        raise FileNotFoundError(
            f"Missing {archive}. Place dataset.zip directly in raw_data/."
        )
    with zipfile.ZipFile(archive) as handle:
        members = {info.filename: info for info in handle.infolist() if not info.is_dir()}
    required = {INPUT_MEMBER, LABEL_MEMBER}
    missing = required.difference(members)
    if missing:
        raise ValueError(f"dataset.zip is missing required members: {sorted(missing)}")
    return members


def prepare_dataset_zip(project_root, archive_path=None):
    """Extract the two NPY members once and return their cache directory.

    Existing cache files are reused only when their byte sizes match the ZIP
    entries.  A mismatched file is not overwritten automatically.
    """
    project_root = Path(project_root)
    archive = Path(archive_path) if archive_path else project_root / "raw_data" / "dataset.zip"
    archive = archive.resolve()
    members = _validate_archive(archive)
    cache_dir = project_root / "processed" / "z24_dataset_zip"
    cache_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as handle:
        for name in (INPUT_MEMBER, LABEL_MEMBER):
            target = cache_dir / name
            expected_size = members[name].file_size
            if target.exists():
                if target.stat().st_size != expected_size:
                    raise ValueError(
                        f"Cached file has the wrong size: {target}. "
                        "Remove that file explicitly before rebuilding the cache."
                    )
                continue

            temporary = target.with_suffix(target.suffix + ".partial")
            if temporary.exists():
                raise FileExistsError(
                    f"Incomplete extraction exists: {temporary}. "
                    "Remove it explicitly before retrying."
                )
            with handle.open(name) as source, temporary.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
            temporary.replace(target)

    inputs, labels = load_dataset_arrays(cache_dir)
    metadata = {
        "archive": str(archive),
        "inputs_shape": list(inputs.shape),
        "inputs_dtype": str(inputs.dtype),
        "labels_shape": list(labels.shape),
        "labels_dtype": str(labels.dtype),
        "classes": np.unique(labels).tolist(),
        "setups_per_condition": SETUPS_PER_CONDITION,
        "segments_per_recording": SEGMENTS_PER_RECORDING,
        "stored_layout": "samples x sensors x time",
    }
    (cache_dir / "dataset.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return cache_dir


def load_dataset_arrays(cache_dir):
    """Memory-map the extracted inputs and load the small label array."""
    cache_dir = Path(cache_dir)
    inputs = np.load(cache_dir / INPUT_MEMBER, mmap_mode="r", allow_pickle=False)
    labels = np.load(cache_dir / LABEL_MEMBER, allow_pickle=False)

    if inputs.shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"Expected inputs shape {EXPECTED_INPUT_SHAPE}, got {inputs.shape}"
        )
    if inputs.dtype != np.float32:
        raise ValueError(f"Expected float32 inputs, got {inputs.dtype}")
    if labels.shape != EXPECTED_LABEL_SHAPE:
        raise ValueError(
            f"Expected labels shape {EXPECTED_LABEL_SHAPE}, got {labels.shape}"
        )
    if labels.dtype != np.int64:
        raise ValueError(f"Expected int64 labels, got {labels.dtype}")

    values, counts = np.unique(labels, return_counts=True)
    if not np.array_equal(values, EXPECTED_LABELS):
        raise ValueError(f"Expected labels 0--16, got {values.tolist()}")
    expected_per_label = SETUPS_PER_CONDITION * SEGMENTS_PER_RECORDING
    if not np.all(counts == expected_per_label):
        raise ValueError(
            f"Expected {expected_per_label} samples per label, got {counts.tolist()}"
        )

    # The public dataset is ordered by condition, setup, then segment.  Require
    # that ordering before deriving recording groups from array positions.
    expected_order = np.repeat(EXPECTED_LABELS, expected_per_label)
    if not np.array_equal(labels, expected_order):
        raise ValueError(
            "Labels are not in condition-major order; recording groups cannot "
            "be inferred safely without a manifest."
        )
    return inputs, labels


def sample_metadata(labels):
    """Derive condition, setup, segment, and recording IDs for every sample."""
    labels = np.asarray(labels)
    sample_index = np.arange(len(labels), dtype=np.int64)
    within_condition = sample_index % (
        SETUPS_PER_CONDITION * SEGMENTS_PER_RECORDING
    )
    setup_id = within_condition // SEGMENTS_PER_RECORDING
    segment_id = within_condition % SEGMENTS_PER_RECORDING
    recording_id = labels.astype(np.int64) * SETUPS_PER_CONDITION + setup_id
    return {
        "condition": labels.astype(np.int64, copy=False),
        "setup": setup_id,
        "segment": segment_id,
        "recording": recording_id,
    }


def split_by_setup(labels, seed=42, train_setups=6, validation_setups=1):
    """Split all conditions using disjoint setup IDs.

    With the default 6/1/2 allocation, every condition contributes 60 segments
    to train, 10 to validation, and 20 to test.  Using the same setup allocation
    for every condition also prevents setup identity from crossing splits.
    """
    if train_setups < 1 or validation_setups < 1:
        raise ValueError("train_setups and validation_setups must be positive")
    test_setups = SETUPS_PER_CONDITION - train_setups - validation_setups
    if test_setups < 1:
        raise ValueError("At least one setup must remain for the test split")

    metadata = sample_metadata(labels)
    setup_order = np.arange(SETUPS_PER_CONDITION, dtype=np.int64)
    np.random.default_rng(seed).shuffle(setup_order)
    setup_split = {
        "train": setup_order[:train_setups],
        "validation": setup_order[
            train_setups : train_setups + validation_setups
        ],
        "test": setup_order[train_setups + validation_setups :],
    }
    splits = {
        name: np.flatnonzero(np.isin(metadata["setup"], setup_ids))
        for name, setup_ids in setup_split.items()
    }

    recording_sets = {
        name: set(metadata["recording"][indexes].tolist())
        for name, indexes in splits.items()
    }
    assert recording_sets["train"].isdisjoint(recording_sets["validation"])
    assert recording_sets["train"].isdisjoint(recording_sets["test"])
    assert recording_sets["validation"].isdisjoint(recording_sets["test"])
    return splits, setup_split, metadata


def materialize_split(inputs, labels, indexes, channels_last=True, chunk_size=64):
    """Copy one split from the memory map into model-ready float32 arrays."""
    indexes = np.asarray(indexes, dtype=np.int64)
    if channels_last:
        result = np.empty(
            (len(indexes), inputs.shape[2], inputs.shape[1]), dtype=np.float32
        )
        for start in range(0, len(indexes), chunk_size):
            selected = indexes[start : start + chunk_size]
            result[start : start + len(selected)] = np.asarray(
                inputs[selected], dtype=np.float32
            ).transpose(0, 2, 1)
    else:
        result = np.empty((len(indexes), *inputs.shape[1:]), dtype=np.float32)
        for start in range(0, len(indexes), chunk_size):
            selected = indexes[start : start + chunk_size]
            result[start : start + len(selected)] = np.asarray(
                inputs[selected], dtype=np.float32
            )
    return result, np.asarray(labels[indexes], dtype=np.int64)


def normalize_from_train(x_train, *other_arrays, channels_last=True):
    """Normalize every split using per-sensor statistics from train only."""
    sensor_axis = 2 if channels_last else 1
    reduce_axes = tuple(axis for axis in range(3) if axis != sensor_axis)
    mean = x_train.mean(axis=reduce_axes, keepdims=True, dtype=np.float64).astype(
        np.float32
    )
    std = x_train.std(axis=reduce_axes, keepdims=True, dtype=np.float64).astype(
        np.float32
    )
    std = np.where(std < 1e-8, 1.0, std)
    normalized = []
    for array in (x_train, *other_arrays):
        array -= mean
        array /= std
        normalized.append(array)
    return (*normalized, mean.reshape(-1), std.reshape(-1))
