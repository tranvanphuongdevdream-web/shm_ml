import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.data.z24_dataset import (
    SEGMENTS_PER_RECORDING,
    SETUPS_PER_CONDITION,
    find_npy_pair,
    normalize_from_train,
    sample_metadata,
    split_by_setup,
)


class DatasetDiscoveryTests(unittest.TestCase):
    def test_finds_nested_kaggle_dataset_pair(self):
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "datasets" / "owner" / "dataset"
            dataset.mkdir(parents=True)
            inputs = dataset / "inputs.npy"
            labels = dataset / "labels.npy"
            inputs.touch()
            labels.touch()
            self.assertEqual(find_npy_pair(Path(directory)), (inputs, labels))

    def test_rejects_missing_or_ambiguous_pairs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                find_npy_pair(root)
            for name in ("first", "second"):
                dataset = root / name
                dataset.mkdir()
                (dataset / "inputs.npy").touch()
                (dataset / "labels.npy").touch()
            with self.assertRaises(ValueError):
                find_npy_pair(root)


class Z24GroupingTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat(
            np.arange(17, dtype=np.int64),
            SETUPS_PER_CONDITION * SEGMENTS_PER_RECORDING,
        )

    def test_metadata_groups_ten_segments_per_recording(self):
        metadata = sample_metadata(self.labels)
        recording_ids, counts = np.unique(metadata["recording"], return_counts=True)
        self.assertEqual(len(recording_ids), 17 * SETUPS_PER_CONDITION)
        np.testing.assert_array_equal(counts, SEGMENTS_PER_RECORDING)

    def test_default_split_is_disjoint_balanced_and_stable(self):
        splits, setup_split, metadata = split_by_setup(self.labels, seed=42)
        self.assertEqual(
            {name: len(indexes) for name, indexes in splits.items()},
            {"train": 1020, "validation": 170, "test": 340},
        )
        self.assertEqual(setup_split["train"].tolist(), [3, 0, 7, 2, 4, 6])
        self.assertEqual(setup_split["validation"].tolist(), [1])
        self.assertEqual(setup_split["test"].tolist(), [5, 8])
        recordings = {
            name: set(metadata["recording"][indexes].tolist())
            for name, indexes in splits.items()
        }
        self.assertTrue(recordings["train"].isdisjoint(recordings["validation"]))
        self.assertTrue(recordings["train"].isdisjoint(recordings["test"]))
        self.assertTrue(recordings["validation"].isdisjoint(recordings["test"]))


class NormalizationTests(unittest.TestCase):
    def test_layouts_produce_the_same_sensor_statistics(self):
        sensors_time = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        validation = sensors_time.copy()
        st_train, st_validation, st_mean, st_std = normalize_from_train(
            sensors_time.copy(), validation.copy(), input_layout="sensors_time"
        )
        time_sensors = sensors_time.transpose(0, 2, 1)
        ts_train, ts_validation, ts_mean, ts_std = normalize_from_train(
            time_sensors.copy(), time_sensors.copy(), input_layout="time_sensors"
        )
        np.testing.assert_allclose(st_mean, ts_mean)
        np.testing.assert_allclose(st_std, ts_std)
        np.testing.assert_allclose(st_train.transpose(0, 2, 1), ts_train)
        np.testing.assert_allclose(st_validation.transpose(0, 2, 1), ts_validation)


if __name__ == "__main__":
    unittest.main()
