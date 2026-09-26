import json
import os
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from src.reporting import build_comparison_report


class ReportingTests(unittest.TestCase):
    @staticmethod
    def make_run(root, experiment, *, action="train", status="completed",
                 accuracy=0.6, archive=False, test_setups=(5, 8),
                 stamp="26-09-26_12-00-00"):
        name = f"{experiment}_{stamp}"
        metadata = {
            "status": status,
            "action": action,
            "config": {"experiment": experiment, "seed": 42, "num_classes": 17},
        }
        dataset = {
            "setup_split": {
                "train": [3, 0, 7, 2, 4, 6],
                "validation": [1],
                "test": list(test_setups),
            },
            "sample_counts": {"train": 1020, "validation": 170, "test": 340},
        }
        benchmark = {
            "pipeline": experiment,
            "training_seconds_total": 180,
            "epochs_completed": 20,
            "mean_epoch_seconds_excluding_first": 9,
            "effective_train_samples_per_second": 100,
            "model_parameters": 500_000,
            "gpu_names": ["Tesla T4"],
            "normalization": "train_sensor_zscore",
        }
        csv = (
            "split,accuracy,precision_macro,recall_macro,f1_macro\n"
            "train,0.9,0.9,0.9,0.9\n"
            "validation,0.7,0.7,0.7,0.7\n"
            f"test,{accuracy},{accuracy},{accuracy},{accuracy}\n"
        )
        files = {
            "experiment.json": json.dumps(metadata),
            "dataset.json": json.dumps(dataset),
            "benchmark_summary.json": json.dumps(benchmark),
            "split_metrics.csv": csv,
        }
        root.mkdir(parents=True, exist_ok=True)
        if archive:
            path = root / f"{name}.zip"
            with zipfile.ZipFile(path, "w") as output:
                for filename, contents in files.items():
                    output.writestr(filename, contents)
        else:
            path = root / name
            path.mkdir()
            for filename, contents in files.items():
                (path / filename).write_text(contents, encoding="utf-8")
        return path

    def test_compares_completed_runs_from_directory_and_zip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            attached = root / "input"
            run_dir = self.make_run(output, "dcnn_001", accuracy=0.65)
            run_dir.rename(output / "dcnn_001")
            self.make_run(attached, "dcnn_002", accuracy=0.7, archive=True)
            self.make_run(attached, "tsai_001", status="failed")
            self.make_run(output / "smoke", "tsai_001", action="smoke")

            report, skipped = build_comparison_report([output, attached], output)

            self.assertEqual(report["experiments"], ["dcnn_001", "dcnn_002"])
            self.assertAlmostEqual(report["quality"].loc[("dcnn_002", "test"), "accuracy"], 0.7)
            self.assertAlmostEqual(report["efficiency"].loc["dcnn_001", "train_test_accuracy_gap"], 0.25)
            self.assertEqual(len(skipped), 2)
            self.assertTrue((report["directory"] / "quality_by_split.csv").is_file())
            self.assertTrue((report["directory"] / "training_benchmark.csv").is_file())
            self.assertEqual(len(report["charts"]), 3)
            self.assertTrue(all(path.is_file() for path in report["charts"].values()))

    def test_rejects_incompatible_setup_splits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root, "dcnn_001")
            self.make_run(root, "tsai_001", test_setups=(6, 8))
            with self.assertRaisesRegex(ValueError, "Cannot compare"):
                build_comparison_report([root], root)

    def test_single_completed_run_still_has_tables_and_charts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root, "tsai_001")
            report, skipped = build_comparison_report([root], root)
            self.assertEqual(report["experiments"], ["tsai_001"])
            self.assertEqual(skipped, [])
            self.assertTrue(all(path.is_file() for path in report["charts"].values()))

    def test_comparison_reads_zip_only_model_folder(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            archive = self.make_run(output / "tsai_001", "tsai_001", archive=True)
            report, skipped = build_comparison_report([output], output)
            self.assertEqual(report["experiments"], ["tsai_001"])
            self.assertEqual(report["efficiency"].loc["tsai_001", "run_source"], str(archive))
            self.assertEqual(skipped, [])

    def test_custom_model_split_and_metrics(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            self.make_run(output / "dcnn_002", "dcnn_002", archive=True)
            self.make_run(output / "tsai_001", "tsai_001", archive=True)
            report, skipped = build_comparison_report(
                [output], output,
                experiment_ids=("tsai_001", "dcnn_002"),
                score_split="validation",
                score_metrics=("f1_macro", "accuracy"),
                split_metric="recall_macro",
                speed_metric="precision_macro",
            )
            self.assertEqual(report["experiments"], ["tsai_001", "dcnn_002"])
            self.assertEqual(report["missing_experiments"], [])
            self.assertEqual(list(report["selected_quality"].columns), ["f1_macro", "accuracy"])
            self.assertAlmostEqual(report["selected_quality"].loc["tsai_001", "f1_macro"], 0.7)
            options = json.loads((report["directory"] / "comparison_config.json").read_text())
            self.assertEqual(options["score_split"], "validation")
            self.assertEqual(options["speed_metric"], "precision_macro")
            self.assertEqual(skipped, [])

    def test_invalid_comparison_options_fail_before_writing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Unknown score_split"):
                build_comparison_report([root], root, score_split="dev")
            with self.assertRaisesRegex(ValueError, "Unknown metric"):
                build_comparison_report([root], root, score_metrics=("accuracy", "auc"))
            with self.assertRaisesRegex(ValueError, "Unknown experiment"):
                build_comparison_report([root], root, experiment_ids=("new_model",))

    def test_run_timestamp_wins_over_archive_upload_time(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            older = self.make_run(
                root / "input", "tsai_001", accuracy=0.4,
                archive=True, stamp="25-09-26_12-00-00",
            )
            self.make_run(root / "results", "tsai_001", accuracy=0.8)
            os.utime(older, (older.stat().st_atime + 86400 * 10,
                             older.stat().st_mtime + 86400 * 10))
            report, _ = build_comparison_report(
                [root / "input", root / "results"], root / "results"
            )
            self.assertAlmostEqual(
                report["quality"].loc[("tsai_001", "test"), "accuracy"], 0.8
            )


if __name__ == "__main__":
    unittest.main()
