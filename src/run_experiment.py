"""Run the same Z24 experiment locally and on Kaggle."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.data.z24_dataset import (
    dataset_summary,
    default_output_root,
    load_dataset_arrays,
    prepare_splits,
    project_root,
    resolve_data_source,
    split_by_setup,
)
from src.evaluation import save_json, save_training_charts


EXPERIMENT_MODULES = {
    "dcnn_001": "src.experiments.dcnn_001",
    "dcnn_002": "src.experiments.dcnn_002",
    "tsai_001": "src.experiments.tsai_001",
}
REQUIRED_KEYS = {
    "experiment",
    "seed",
    "epochs",
    "batch_size",
    "train_setups",
    "validation_setups",
    "num_classes",
    "input_layout",
}
SMOKE_BATCH_SIZE = 8


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Z24: check data, smoke test locally, or train on Kaggle."
    )
    parser.add_argument(
        "action",
        choices=("check", "smoke", "train"),
        help="check=data/split only; smoke=1 epoch; train=full config",
    )
    parser.add_argument(
        "--experiment",
        default="dcnn_002",
        choices=tuple(EXPERIMENT_MODULES),
        help="experiment ID to run (default: dcnn_002)",
    )
    return parser.parse_args(argv)


def load_config(experiment_name):
    if experiment_name not in EXPERIMENT_MODULES:
        raise ValueError(f"Unknown experiment {experiment_name!r}")
    path = project_root() / "configs" / f"{experiment_name}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    if config["experiment"] != experiment_name:
        raise ValueError(f"{path} declares experiment={config['experiment']!r}, expected {experiment_name!r}")
    return config


def validate_config(config):
    missing = REQUIRED_KEYS.difference(config)
    if missing:
        raise ValueError(f"Config is missing required keys: {sorted(missing)}")
    experiment = config["experiment"]
    if experiment not in EXPERIMENT_MODULES:
        raise ValueError(f"Unknown experiment {experiment!r}")
    if int(config["num_classes"]) != 17:
        raise ValueError("The Z24 dataset requires num_classes=17")
    if int(config["epochs"]) < 1 or int(config["batch_size"]) < 1:
        raise ValueError("epochs and batch_size must be positive")
    if int(config["train_setups"]) < 1 or int(config["validation_setups"]) < 1:
        raise ValueError("train_setups and validation_setups must be positive")
    expected_layout = "time_sensors" if experiment == "dcnn_002" else "sensors_time"
    if config["input_layout"] != expected_layout:
        raise ValueError(f"{experiment} requires input_layout={expected_layout!r}")
    if experiment == "tsai_001" and config["scheduler"] != "one_cycle":
        raise ValueError("tsai_001 supports scheduler='one_cycle' only")


def check_dataset(config):
    inputs_path, labels_path = resolve_data_source()
    inputs, labels = load_dataset_arrays(inputs_path, labels_path)
    indexes, setup_split, metadata = split_by_setup(
        labels,
        seed=int(config["seed"]),
        train_setups=int(config["train_setups"]),
        validation_setups=int(config["validation_setups"]),
    )
    print(f"Inputs: {inputs_path} {inputs.shape} {inputs.dtype}")
    print(f"Labels: {labels_path} {labels.shape} {labels.dtype}")
    for name in ("train", "validation", "test"):
        recordings = set(metadata["recording"][indexes[name]].tolist())
        print(
            f"{name:10s}: {len(indexes[name]):4d} samples, "
            f"{len(recordings):3d} recordings, setups={setup_split[name].tolist()}"
        )


def smoke_config(config):
    """Return a short-run copy of any experiment configuration."""
    short_config = config.copy()
    short_config["epochs"] = 1
    short_config["batch_size"] = min(int(config["batch_size"]), SMOKE_BATCH_SIZE)
    return short_config


def run(action, experiment_name):
    config = load_config(experiment_name)
    print(f"Action: {action} | Experiment: {experiment_name}", flush=True)
    if action == "check":
        check_dataset(config)
        return
    if action == "smoke":
        config = smoke_config(config)
        print(f"Smoke test: 1 epoch, batch size {config['batch_size']}; config file unchanged")

    print("Preparing and normalizing dataset...", flush=True)
    prepared = prepare_splits(config)
    print("Prepared dataset:")
    print(json.dumps(dataset_summary(prepared), indent=2))
    for name in ("train", "validation", "test"):
        if not np.isfinite(prepared[name][0]).all():
            raise ValueError(f"Non-finite normalized values in {name}")

    output_root = default_output_root().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = output_root / experiment_name
    model_dir.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    archive = model_dir / f"{experiment_name}_{run_id}.zip"
    if archive.exists():
        raise FileExistsError(f"Run archive already exists: {archive}")

    with TemporaryDirectory(prefix=f".{experiment_name}_", dir=output_root) as temporary_dir:
        artifact_dir = Path(temporary_dir) / "artifacts"
        artifact_dir.mkdir()
        print(f"Temporary training artifacts: {artifact_dir}", flush=True)
        save_json(artifact_dir / "config.json", config)
        save_json(artifact_dir / "dataset.json", dataset_summary(prepared))

        def pack(destination):
            if destination.exists():
                raise FileExistsError(f"Run archive already exists: {destination}")
            staged_archive = shutil.make_archive(
                str(Path(temporary_dir) / destination.stem), "zip", root_dir=artifact_dir
            )
            shutil.move(staged_archive, destination)
            print(f"Download archive: {destination}", flush=True)
            return destination

        print(f"Loading experiment module: {EXPERIMENT_MODULES[experiment_name]}", flush=True)
        experiment = importlib.import_module(EXPERIMENT_MODULES[experiment_name])
        print("Starting model training...", flush=True)
        try:
            benchmark = experiment.run(config, prepared, artifact_dir)
            save_training_charts(artifact_dir)
        except Exception:
            save_json(artifact_dir / "experiment.json", {
                "status": "failed",
                "action": action,
                "config": config,
                "dataset": dataset_summary(prepared),
            })
            failed_archive = model_dir / f"{experiment_name}_{run_id}_failed.zip"
            pack(failed_archive)
            raise
        save_json(artifact_dir / "experiment.json", {
            "status": "completed",
            "action": action,
            "config": config,
            "dataset": dataset_summary(prepared),
            "benchmark": benchmark,
        })
        return pack(archive)


def main(argv=None):
    args = parse_args(argv)
    run(args.action, args.experiment)


if __name__ == "__main__":
    main()
