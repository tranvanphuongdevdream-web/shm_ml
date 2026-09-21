# Mode 01: original raw-PDT data

This folder preserves the original project workflow based on:

```text
raw_data/sources-Z24-004.zip
```

Pipeline:

```text
PDT_cleaning.ipynb
    -> processed AVT/FVT CSV files and manifest
    -> DCNN_LSTM_ResNet_training.ipynb
       or Z24_tsai_classification.ipynb
```

Mode 01 characteristics:

- AVT and FVT are separate domains;
- five common named sensors are used;
- source recordings are rebuilt from 6,000-row storage CSV files;
- models receive 8,000-sample windows;
- split isolation is based on the original MAT source recording.

These notebooks are retained for comparison and reproducibility. They are not
used by the Mode 02 `dataset.zip` experiments.
