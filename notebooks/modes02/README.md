# Mode 02: processed dataset.zip

This folder is the primary workflow for:

```text
raw_data/dataset.zip
```

Both notebooks use the shared loader in `src/data/z24_dataset_zip.py`.

Common data contract:

```text
stored:     (1530, 27, 6000) [samples, sensors, time]
labels:     (1530,)          [0--16]
grouping:   17 conditions x 9 setups x 10 segments
```

The default seed-42 split uses six setup IDs for train, one for validation, and
two for test. The same setup allocation is used for all conditions, and ten
segments belonging to one inferred recording never cross split boundaries.

- `DCNN_LSTM_ResNet_training.ipynb` converts inputs to Keras layout
  `(samples, 6000, 27)`.
- `Z24_tsai_classification.ipynb` keeps tsai layout
  `(samples, 27, 6000)`.

Both notebooks calculate normalization statistics from train only and keep test
outside model fitting.
