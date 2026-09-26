import unittest

from src.run_experiment import load_config, parse_args, smoke_config


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


if __name__ == "__main__":
    unittest.main()
