import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from src.run_experiment import load_config, parse_args, run, smoke_config


class ConfigTests(unittest.TestCase):
    def test_every_experiment_config_is_valid(self):
        for name in ("dcnn_001", "dcnn_002", "tsai_001"):
            config = load_config(name)
            self.assertEqual(config["experiment"], name)

    def test_experiment_parameter_selects_config(self):
        args = parse_args(["check"])
        self.assertEqual((args.action, args.experiment), ("check", "dcnn_002"))
        args = parse_args(["train", "--experiment", "tsai_001"])
        self.assertEqual((args.action, args.experiment), ("train", "tsai_001"))

    def test_smoke_settings_work_for_every_experiment_without_changing_config(self):
        for name in ("dcnn_001", "dcnn_002", "tsai_001"):
            full_config = load_config(name)
            smoke = smoke_config(full_config)
            self.assertEqual(smoke["epochs"], 1)
            self.assertEqual(smoke["batch_size"], min(full_config["batch_size"], 8))
            self.assertNotEqual(full_config["epochs"], 1)

    def test_train_saves_only_a_complete_zip_under_experiment_folder(self):
        with TemporaryDirectory() as directory:
            output_root = Path(directory) / "results"
            config = {"experiment": "tsai_001"}
            arrays = {name: (np.zeros((1, 1), dtype=np.float32), np.array([0]))
                      for name in ("train", "validation", "test")}

            def fake_train(_config, _prepared, artifact_dir):
                (artifact_dir / "history.csv").write_text(
                    "epoch,train_loss,valid_loss,accuracy\n1,0.8,0.7,0.5\n",
                    encoding="utf-8",
                )
                (artifact_dir / "split_metrics.csv").write_text(
                    "split,accuracy,precision_macro,recall_macro,f1_macro\n"
                    "train,0.7,0.7,0.7,0.7\n"
                    "validation,0.6,0.6,0.6,0.6\n"
                    "test,0.5,0.5,0.5,0.5\n",
                    encoding="utf-8",
                )
                return {"pipeline": "tsai_001"}

            experiment = ModuleType("src.experiments.tsai_001")
            experiment.run = fake_train
            with patch("src.run_experiment.load_config", return_value=config), \
                 patch("src.run_experiment.prepare_splits", return_value=arrays), \
                 patch("src.run_experiment.dataset_summary", return_value={}), \
                 patch("src.run_experiment.default_output_root", return_value=output_root), \
                 patch.dict("sys.modules", {"src.experiments.tsai_001": experiment}):
                archive_path = run("train", "tsai_001")

            self.assertEqual(archive_path.parent, output_root / "tsai_001")
            self.assertEqual(list(archive_path.parent.iterdir()), [archive_path])
            with ZipFile(archive_path) as archive:
                self.assertTrue({
                    "experiment.json", "history.csv", "split_metrics.csv",
                    "learning_curves.png", "split_performance.png",
                }.issubset(archive.namelist()))
                self.assertFalse(any(name.endswith(".zip") for name in archive.namelist()))
            self.assertEqual(
                sorted(path.name for path in output_root.iterdir()), ["tsai_001"]
            )

    def test_failed_train_is_packaged_without_loose_files(self):
        with TemporaryDirectory() as directory:
            output_root = Path(directory) / "results"
            config = {"experiment": "tsai_001"}
            arrays = {name: (np.zeros((1, 1), dtype=np.float32), np.array([0]))
                      for name in ("train", "validation", "test")}

            def fail_train(_config, _prepared, artifact_dir):
                (artifact_dir / "partial.log").write_text("partial", encoding="utf-8")
                raise RuntimeError("training failed")

            experiment = ModuleType("src.experiments.tsai_001")
            experiment.run = fail_train
            with patch("src.run_experiment.load_config", return_value=config), \
                 patch("src.run_experiment.prepare_splits", return_value=arrays), \
                 patch("src.run_experiment.dataset_summary", return_value={}), \
                 patch("src.run_experiment.default_output_root", return_value=output_root), \
                 patch.dict("sys.modules", {"src.experiments.tsai_001": experiment}):
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    run("train", "tsai_001")

            archives = list((output_root / "tsai_001").iterdir())
            self.assertEqual(len(archives), 1)
            self.assertTrue(archives[0].name.endswith("_failed.zip"))
            with ZipFile(archives[0]) as archive:
                metadata = json.loads(archive.read("experiment.json"))
                self.assertEqual(metadata["status"], "failed")
                self.assertEqual(archive.read("partial.log"), b"partial")


if __name__ == "__main__":
    unittest.main()
