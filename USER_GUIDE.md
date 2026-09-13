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
├── prepare_pdt.py
├── notebooks/
│   └── PDT_cleaning.ipynb
├── raw_data/
│   └── sources-Z24.zip
└── requirements.txt
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

## Common problems

- `FileNotFoundError`: confirm that the ZIP is directly under `raw_data/` and that its name starts with `sources-Z24`.
- The notebook appears busy but no files change: interrupt the cell, restart the kernel, and run only one measurement cell. Progress output shows the recording currently being read.
- A run folder has no `report.json`: that run was interrupted and is incomplete. Use a new timestamped run folder.
