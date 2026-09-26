import unittest

from src.experiments.tsai_001 import training_history_frame


class TsaiHistoryTests(unittest.TestCase):
    def test_preserves_named_training_metrics(self):
        names = ["epoch", "train_loss", "valid_loss", "accuracy", "time"]
        history = training_history_frame([[1.2, 1.4, 0.5], [0.8, 1.1, 0.6]], names)
        self.assertEqual(
            list(history.columns),
            ["epoch", "train_loss", "valid_loss", "accuracy"],
        )
        self.assertEqual(history["epoch"].tolist(), [1, 2])

    def test_mismatched_names_do_not_fail_after_training(self):
        names = ["epoch", "train_loss", "valid_loss", "accuracy", "time"]
        history = training_history_frame([[0.8, 1.1], [0.6, 1.0]], names)
        self.assertEqual(list(history.columns), ["epoch", "metric_1", "metric_2"])
        self.assertEqual(history.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
