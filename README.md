# Z24 PDT data preparation

This project processes the Z24 Progressive Damage Test (PDT) MATLAB recordings. The old EMS (`.aaa`) cleaning pipeline, notebook, and generated files have been removed. The raw ZIP is not stored in GitHub because it is several gigabytes in size.

## Download the raw data

Download the archive from the shared Google Drive folder:

[Z24 raw data on Google Drive](https://drive.google.com/drive/folders/12M86AaH_Wg6Ub8jzQ6lsWKw95gh8HMlu?usp=sharing)

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Create the directory `raw_data/` if it does not exist.
4. Download `sources-Z24.zip` from Google Drive.
5. Place the ZIP directly inside `raw_data/`.

Expected layout:

```text
shm/
|-- src/
|   `-- data/
|       `-- pdt_cleaning.py
|-- notebooks/
|   `-- PDT_cleaning.ipynb
|-- raw_data/
|   `-- sources-Z24.zip
`-- requirements.txt
```

The notebook also accepts the original filename `sources-Z24-004.zip`. Do not place the ZIP inside another subfolder and do not extract the nested PDT packages manually.

## Install dependencies

From the project folder, run:

```bash
python -m pip install -r requirements.txt
```

Open `notebooks/PDT_cleaning.ipynb` in Jupyter or VS Code and select the Python environment containing these dependencies. The setup cell automatically searches `raw_data/` for a file matching `sources-Z24*.zip`.

## Input structure

The reader opens the two PDT members `pdt_01-08.zip` and `pdt_09_17.zip`. Each member contains condition folders 01--17, measurement folders (`avt` and `fvt`), setup folders, and `.mat` recordings. Every MATLAB file must contain `data` and `labelshulp`; the documented sampling rate is 100 Hz.

## Cleaning and labels

`clean_pdt()` removes rows containing NaN or infinity, keeps all remaining finite samples, and writes consecutive 6,000-sample CSV segments. The final segment is preserved even when it is shorter than 6,000 samples. The condition directory is the label source: `label = condition_id - 1`. No label is guessed from signal values. Channel names remain in `manifest.json`.

AVT and FVT are processed independently. Run only the measurement required for the experiment. Each recording prints progress, and `status.json` shows whether the run is still running or complete.

## Run the cleaner

Run the setup cell first. Then run one of the measurement cells:

```python
avt_report = run_cleaning("avt")
```

```python
fvt_report = run_cleaning("fvt")
```

Each run creates a timestamped directory:

```text
processed/avt/avt_dd-mm-yyyy_HH-mm-ss/
processed/fvt/fvt_dd-mm-yyyy_HH-mm-ss/
```

## Output files

- `segments/*.csv`: one row per time sample, with `sample_index` and one column per sensor channel.
- `manifest.json`: source file, condition, label, setup, measurement, segment position, channel count, and channel names for every CSV.
- `labels.csv`: the documented condition-to-label table.
- `report.json`: processing scope, counts, and skipped or invalid recordings.
- `status.json`: live progress; `running` means incomplete and `complete` means the run finished successfully.

CSV is human-readable and can be reviewed in VS Code or loaded with NumPy/Pandas. For model training, read the sensor columns and convert them to tensors.

## Visual checks

The final notebook cell plots:

- segment counts by condition;
- AVT/FVT segment totals;
- several sensor channels over time;
- a sensor-value histogram.

These charts are for data inspection. They do not train a model.

## Before training

Conditions 1, 2, and 8 are reference or transition scenarios. Keep all 17 labels for multiclass classification, or define an explicit rule before combining them into healthy/damaged classes. Confirm sensor units, sensor locations, and acquisition boundaries from `doc/Knowledge_based.pdf` before creating model windows.

Split train, validation, and test data by recording or setup so segments from one recording cannot leak across splits. Compute normalization statistics from the training subset only. AVT and FVT should initially be evaluated as separate measurement domains.

## Train the 1DCNN-LSTM-ResNet model

Open `notebooks/DCNN_LSTM_ResNet_training.ipynb` after completing the cleaning notebook. The training notebook reads the newest complete AVT or FVT run directly from `processed/`; it does not require the pickle files used by the original implementation.

The training notebook reproduces the teacher's 1DCNN-LSTM-ResNet layer
configuration and training settings. Only the data interface differs:

- 17 output classes, with labels 0--16 corresponding to conditions 1--17;
- the five channels shared by every setup: `R1V`, `R2L`, `R2T`, `R2V`, and `R3V`;
- 8,000 samples per window, equal to 80 seconds at 100 Hz;
- the teacher source input layout `(channels, time_samples)`, giving an input shape of `(5, 8000)`;
- a condition-stratified split by source recording: six recordings for training, one for validation, and two for testing in every condition;
- normalization statistics calculated from the training split only.

The loader first joins the 6,000-row CSV storage segments belonging to the same source recording, then creates non-overlapping 8,000-sample model windows. A remainder shorter than 8,000 samples is discarded without crossing into another recording. All windows from a source `.mat` recording remain in one split. This prevents neighboring windows from appearing in both training and evaluation data. The test set is evaluated only after model training.

Set `MEASUREMENT = "avt"` or `MEASUREMENT = "fvt"` near the beginning of the notebook. Train the two measurement types in separate runs so their results can be compared fairly. Generated models, metrics, and normalization values are written to `artifacts/dcnn_lstm_resnet/` and are excluded from Git.

The model module is stored as `src/models/dcnn_lstm_resnet.py`. Python module filenames cannot contain hyphens when imported normally, so the original filename `DCNN-LSTM-ResNet.py` was changed to an importable name. The architecture follows the teacher's source and the associated publication:

> Le-Xuan Thang, Bui-Tien Thanh, and Tran-Ngoc Hoa, "A novel approach model design for signal data using 1DCNN combing with LSTM and ResNet for damaged detection problem," *Structures*, 59, 105784, 2024. DOI: 10.1016/j.istruc.2023.105784.

The original reported accuracy is not treated as a result of this project. Run the notebook on the cleaned data and report the resulting test metrics.

## Standard time-series format and tsai baseline

The official [tsai repository](https://github.com/timeseriesAI/tsai) defines time-series input as a three-dimensional array:

```text
samples x variables x sequence length
```

Open `notebooks/Z24_tsai_classification.ipynb` to convert the cleaned Z24 windows to this contract and train a PyTorch/fastai baseline. For the current experiment:

```text
X shape = (number of windows, 5 sensors, 8000 time samples)
y shape = (number of windows,)
```

The notebook uses `ResNet` as the first tsai baseline. It preserves the recording-level split, applies training-only normalization, keeps the test set outside the learner during training, and saves the trained learner and test metrics under `artifacts/tsai/`.

## Project structure

```text
shm/
|-- src/
|   |-- data/
|   |   |-- pdt_cleaning.py          # Raw ZIP and MAT to clean CSV
|   |   `-- pdt_training_data.py     # Clean CSV to model-ready arrays
|   `-- models/
|       `-- dcnn_lstm_resnet.py      # Model architecture
|-- notebooks/
|   |-- PDT_cleaning.ipynb         # Raw PDT to clean CSV
|   |-- DCNN_LSTM_ResNet_training.ipynb
|   `-- Z24_tsai_classification.ipynb
|-- processed/                     # Generated clean data; ignored by Git
|-- artifacts/                     # Generated models and metrics; ignored by Git
|-- raw_data/                      # Downloaded source ZIP; ignored by Git
|-- requirements.txt
`-- README.md
```

## Common problems

- `FileNotFoundError`: confirm that the ZIP is directly under `raw_data/` and that its name starts with `sources-Z24`.
- The notebook appears busy but no files change: interrupt the cell, restart the kernel, and run only one measurement cell. Progress output shows the recording currently being read.
- A run folder has no `report.json`: that run was interrupted and is incomplete. Use a new timestamped run folder.
