import unittest

import numpy as np

from src.data.z24_dataset_zip import (
    SEGMENTS_PER_RECORDING,
    SETUPS_PER_CONDITION,
    sample_metadata,
    split_by_setup,
)


class DatasetZipGroupingTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat(
            np.arange(17, dtype=np.int64),
            SETUPS_PER_CONDITION * SEGMENTS_PER_RECORDING,
        )

    def test_metadata_groups_ten_segments_per_recording(self):
        metadata = sample_metadata(self.labels)
        recording_ids, counts = np.unique(
            metadata["recording"], return_counts=True
        )
        self.assertEqual(len(recording_ids), 17 * SETUPS_PER_CONDITION)
        np.testing.assert_array_equal(counts, SEGMENTS_PER_RECORDING)

    def test_default_split_is_disjoint_and_balanced(self):
        splits, setup_split, metadata = split_by_setup(self.labels, seed=42)
        self.assertEqual(
            {name: len(indexes) for name, indexes in splits.items()},
            {"train": 1020, "validation": 170, "test": 340},
        )
        self.assertEqual(
            {name: len(values) for name, values in setup_split.items()},
            {"train": 6, "validation": 1, "test": 2},
        )
        for indexes in splits.values():
            _, counts = np.unique(self.labels[indexes], return_counts=True)
            self.assertEqual(len(counts), 17)
            self.assertTrue(np.all(counts == counts[0]))

        recordings = {
            name: set(metadata["recording"][indexes].tolist())
            for name, indexes in splits.items()
        }
        self.assertTrue(recordings["train"].isdisjoint(recordings["validation"]))
        self.assertTrue(recordings["train"].isdisjoint(recordings["test"]))
        self.assertTrue(recordings["validation"].isdisjoint(recordings["test"]))


if __name__ == "__main__":
    unittest.main()
