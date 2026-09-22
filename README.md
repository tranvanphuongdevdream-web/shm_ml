# Z24 processed-dataset training

This project trains structural-health-monitoring classifiers from the processed
Z24 archive in `raw_data/dataset.zip`. The archive is the same dataset published
as `thanglexuan/Z24-dataset-processed` on Hugging Face.

The previous MATLAB/PDT pipeline for `sources-Z24-004.zip` remains in `src/data`
for reference, but it is no longer required by the primary training workflow.

## Dataset

Place the archive at:

```text
raw_data/dataset.zip
```

It must contain:

```text
inputs.npy   # (1530, 27, 6000), float32
labels.npy   # (1530,), int64
```

The stored input layout is:

```text
samples x sensors x time samples
```

The 1,530 samples represent 17 conditions, 9 setups per condition, and 10
consecutive 6,000-sample segments per condition/setup recording. Labels are
0--16, with 90 samples per condition.

## Install

```bash
python -m pip install -r requirements.txt
```

Run commands from the project root.

## Validate and prepare the archive

```bash
python -m src.train_z24_dataset_zip --cache-only
```

This extracts the two NPY files once to:

```text
processed/z24_dataset_zip/
```

Subsequent runs reuse those files through NumPy memory mapping. The cache avoids
decompressing the 991 MB input array into an additional in-memory byte buffer.

The command also validates:

- input shape `(1530, 27, 6000)` and `float32` dtype;
- label shape `(1530,)` and `int64` dtype;
- labels 0--16 with exactly 90 samples per label;
- condition-major ordering required to infer recording groups.

## Leakage-safe split

Individual 6,000-sample segments are not independent. Ten neighboring segments
belong to the same condition/setup recording, so they must never be randomly
distributed across train, validation, and test.

The loader derives condition, setup, segment, and recording IDs from the
documented array order. With seed 42 it assigns the same disjoint setup IDs
across all conditions:

```text
train:       6 setups/condition = 1,020 segments = 102 recordings
validation:  1 setup/condition  =   170 segments =  17 recordings
test:        2 setups/condition =   340 segments =  34 recordings
```

Normalization mean and standard deviation are calculated from train only.

## Train the TensorFlow model

For an interactive run, open:

```text
notebooks/modes02/kaggle/DCNN_LSTM_ResNet_training_kaggle.ipynb
```

That notebook contains one self-contained Kaggle code cell. For local,
repository-based development, use:

```text
notebooks/modes02/local/DCNN_LSTM_ResNet_training_local.ipynb
```

The matching tsai baseline is:

```text
notebooks/modes02/kaggle/Z24_tsai_classification_kaggle.ipynb
```

Or use the command-line entry point:

```bash
python -m src.train_z24_dataset_zip
```

Optional settings:

```bash
python -m src.train_z24_dataset_zip --epochs 50 --batch-size 8 --seed 42
```

The NPY file stores each sample as `(27 sensors, 6000 time samples)`. Before
TensorFlow training, the loader converts it to the standard Keras Conv1D layout:

```text
(samples, 6000 time samples, 27 sensors)
```

This intentionally differs from the teacher script's `(5, 8000)` input, which
makes Keras interpret five sensor rows as the temporal dimension.

Training outputs are written to a timestamped directory under:

```text
artifacts/dataset_zip_dcnn_lstm_resnet/
```

Each completed experiment contains the best and final models, training history,
classification report, confusion matrix, exact split indexes, normalization
values, and experiment metadata.

## Important limitations

`dataset.zip` contains only numeric arrays and labels. It does not contain the
original sensor names, AVT/FVT identifiers, or an explicit setup manifest. The
setup/recording grouping is therefore derived from the documented reshape order
`17 x 9 x 10` and is accepted only after the expected label ordering is
validated.

The dataset has 1,530 segments but only 153 inferred independent recordings.
High training accuracy can still coexist with poor unseen-setup accuracy. Use
the grouped test result, not a random segment split, when reporting performance.

## Source layout

```text
notebooks/
|-- modes01/                     # Original sources-Z24-004.zip experiments
|   |-- PDT_cleaning.ipynb
|   |-- DCNN_LSTM_ResNet_training.ipynb
|   `-- Z24_tsai_classification.ipynb
`-- modes02/                     # Processed dataset.zip experiments
    |-- DCNN_LSTM_ResNet_training.ipynb         # One-cell Kaggle version
    |-- DCNN_LSTM_ResNet_training_local.ipynb   # Modular local version
    `-- Z24_tsai_classification.ipynb

src/
|-- data/
|   |-- z24_dataset_zip.py       # Primary dataset.zip loader and grouped split
|   |-- pdt_cleaning.py          # Legacy MATLAB ZIP cleaner
|   `-- pdt_training_data.py     # Legacy CSV training loader
|-- models/
|   `-- dcnn_lstm_resnet.py
`-- train_z24_dataset_zip.py     # Primary training entry point
```
