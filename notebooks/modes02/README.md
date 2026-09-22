# Mode 02: processed dataset.zip

This folder is the primary workflow for:

```text
raw_data/dataset.zip
```

The self-contained Kaggle workflows are in `kaggle/`. The notebooks in
`local/` use the shared loader in `src/data/z24_dataset_zip.py`.

Common data contract:

```text
stored:     (1530, 27, 6000) [samples, sensors, time]
labels:     (1530,)          [0--16]
grouping:   17 conditions x 9 setups x 10 segments
```

The default seed-42 split uses six setup IDs for train, one for validation, and
two for test. The same setup allocation is used for all conditions, and ten
segments belonging to one inferred recording never cross split boundaries.

- `kaggle/DCNN_LSTM_ResNet_training_kaggle.ipynb` is the self-contained fast
  Kaggle DCNN notebook. It uses a cuDNN-compatible LSTM, mixed precision,
  batch 32, `tf.data` prefetching, and two-GPU `MirroredStrategy`.
- `kaggle/Z24_tsai_classification_kaggle.ipynb` is the self-contained Kaggle
  tsai notebook;
  it keeps the tsai layout `(samples, 27, 6000)` and installs pinned
  Kaggle-compatible dependencies (`tsai==0.4.1`, `scikit-learn==1.7.2`, and
  `imbalanced-learn==0.14.0`) without replacing Kaggle's preinstalled
  PyTorch/CUDA stack.
- `local/DCNN_LSTM_ResNet_training_local.ipynb` and
  `local/Z24_tsai_classification_local.ipynb` are the modular local workflows.

Both notebooks calculate normalization statistics from train only and keep test
outside model fitting.
