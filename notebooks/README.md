# Notebook modes

The notebooks are separated by data source so experiments from incompatible
preprocessing pipelines are not mixed.

## `modes01/`: original raw-PDT pipeline

Uses `raw_data/sources-Z24-004.zip`, exports CSV files, selects the five common
named channels, and creates 8,000-sample model windows.

Run notebooks in this order:

1. `modes01/PDT_cleaning.ipynb`
2. `modes01/DCNN_LSTM_ResNet_training.ipynb` or
   `modes01/Z24_tsai_classification.ipynb`

## `modes02/`: processed dataset.zip pipeline

Uses `raw_data/dataset.zip` directly through the shared
`src/data/z24_dataset_zip.py` loader. It trains on 27-sensor, 6,000-sample
segments and uses setup-grouped train/validation/test splits.

Run either:

- `modes02/DCNN_LSTM_ResNet_training.ipynb`
- `modes02/Z24_tsai_classification.ipynb`

Both Mode 02 notebooks reuse the same cache, split seed, setup allocation, and
training-only normalization rules.
