# Z24 PDT data preparation

This project keeps the Z24 Progressive Damage Test (PDT) MATLAB recordings. The old EMS (`.aaa`) cleaning pipeline, notebook, and generated files have been removed. The source ZIP is retained because it also contains the PDT packages and is treated as read-only.

## Install and run

From the project folder, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Open `notebooks/PDT_cleaning.ipynb` in Jupyter or VS Code and select that Python environment. The notebook imports `prepare_pdt.py` and is configured to export all available PDT conditions and setups. The smoke test remains as an optional commented example.

## Input structure

The reader opens `raw_data/sources-Z24-004.zip`, then the two PDT members `pdt_01-08.zip` and `pdt_09_17.zip`. Each member contains condition folders 01--17, measurement folders (`avt` and `fvt`), setup folders, and `.mat` recordings. Every MATLAB file must contain `data` and `labelshulp`; the documented sampling rate is 100 Hz.

The outer ZIP currently ends before the complete `Z24ems3.zip` member. This does not affect the PDT members that appear before it, but the archive should be re-downloaded before a final completeness check.

## Cleaning and labels

`clean_pdt()` reads each selected MATLAB recording, removes rows containing NaN or infinity, keeps the first 60,000 finite samples, and writes ten consecutive 6,000-sample CSV segments. The condition directory is the label source: `label = condition_id - 1`. No label is guessed from signal values. Channel names are stored in `manifest.json` and remain associated with their setup.

The full notebook run uses both AVT and FVT. Pass `measurement="avt"` or `measurement="fvt"` when only one measurement type is required. A smoke test with one condition and one setup produces ten CSV files; a full export of both measurement types can produce up to `17 x 9 x 2 x 10 = 3,060` segments.

## Output files

- `processed/avt/<run-name>/segments/*.csv`: AVT segments.
- `processed/fvt/<run-name>/segments/*.csv`: FVT segments.
- Each `segments/*.csv` has one row per time sample, with `sample_index` and one column per sensor channel.
- `manifest.json`: source file, condition, label, setup, measurement, segment position, channel count, and channel names for every CSV.
- `labels.csv`: the documented condition-to-label table.
- `report.json`: processing scope, counts, and skipped or invalid recordings.

CSV is human-readable and can be reviewed in VS Code or loaded with NumPy/Pandas. For model training, read the sensor columns and convert them to tensors in the training code.

## Before training

Conditions 1, 2, and 8 are reference or transition scenarios. Keep all 17 labels for multiclass classification, or define an explicit rule before combining them into healthy/damaged classes. Confirm sensor units and acquisition boundaries from `doc/Knowledge_based.pdf` before creating windows that cross a gap.
