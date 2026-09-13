# Z24: prepare data with Jupyter

On a new machine, open a terminal in the `AI` folder and run `python -m pip install -r requirements.txt`. Open `notebooks/Z24_prepare_training.ipynb` in Jupyter or VS Code and select the Python kernel that has these dependencies. The data reader is in `z24_prepare.py`; this `.aaa` pipeline does not require SciPy.

## Scope

The notebook processes numeric-suffix `.aaa` channels in the EMS group. Each example is one channel over 10 seconds. PDT, `car.aaa`, and multi-sensor stacking are not processed. `.env` files and `.aaa` footers are retained in the manifest.

The outer ZIP is truncated: `Z24ems3.zip` is skipped and only complete preceding packages are read. ZIP CRCs are checked when members are read. Raw data is never modified.

## Configuration

`MAX_SESSIONS = 20` is a pipeline smoke test, not the final dataset. Set it to `None` to process every session in complete EMS packages. Complete or re-download the ZIP before producing the final dataset.

`WINDOW_SECONDS = 10` creates 1,000 samples per example at 100 Hz. Every run creates a new folder under `processed`; if the timestamp already exists, a numeric suffix is added.

## Validation and normalization

The parser reads sample count and time step from the header and only reads signal values before `Timehistories end here`. Recordings with a wrong count, time step, NaN, or infinity are rejected. Constant windows and incomplete tails shorter than 10 seconds are dropped.

No spike removal, frequency filtering, per-window centering, or label inference is performed. Per-channel mean and standard deviation are computed from train only, then applied to validation and test. Sessions are assigned approximately 70/15/15 by a deterministic hash; every channel and window from one session remains in the same split.

## Output

- `train/`, `val/`, `test/`: one CSV per session/channel. Each row is one window; `start_sample` is the original offset and `sample_0` through `sample_999` are the normalized signal. `read_windows_csv` returns sample columns as `X` with shape `(n_windows, 1000, 1)` and dtype `float32`.
- `manifest.json`: source, session, channel, sampling rate, headers, footers, ENV text, window counts, and rejected data.
- `normalization.json`: train-only count, mean, and standard deviation per channel.
- `report.json`: run scope, counts, and issues.

The notebook generator reads shards and yields batches without loading the complete dataset into RAM. The example selects one channel; channels are not assumed to be interchangeable.

## Before final training

There is no `y` label yet. A verified label table is required for damage classification. Do not assume all EMS recordings are healthy. Autoencoder or self-supervised experiments are possible, but reconstruction loss alone does not prove damage detection quality.

Sample footers report units in g and multiple `Segment #...` acquisitions. The notebook currently windows consecutive sample indices without verifying segment boundaries or gaps. Review the footer and avoid windows crossing acquisition gaps before final analysis.

## Smoke-test result

With 20 sessions: 160 CSV shards and 10,400 windows (train 7,800; validation 1,040; test 1,560). Shape, dtype, finite-value, and session-isolation checks passed. This validates the pipeline on a sample, not the complete dataset.

## Reviewing CSV

CSV files can be opened directly in VS Code, although each row has 1,001 columns. The notebook preview shows sample values and the chart cells plot one window, three windows, and a histogram. CSV values are normalized and no longer in the original g unit. JSON files retain metadata and reports.
