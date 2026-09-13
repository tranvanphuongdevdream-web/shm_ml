# Download the raw Z24 data

The raw archive is not stored in GitHub because it is several gigabytes in size. Download it from the shared Google Drive folder:

[Z24 raw data on Google Drive](https://drive.google.com/drive/folders/12M86AaH_Wg6Ub8jzQ6lsWKw95gh8HMlu?usp=sharing)

## Setup steps

1. Clone or download this GitHub repository.
2. Open a terminal in the project folder.
3. Create the missing raw-data directory:

   ```text
   raw_data/
   ```

4. Download the archive named `sources-Z24.zip` from Google Drive.
5. Place the ZIP directly inside `raw_data/`.

The expected layout is:

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

Then open `notebooks/PDT_cleaning.ipynb` in Jupyter or VS Code. The setup cell automatically searches `raw_data/` for a file matching `sources-Z24*.zip`.

## Run the cleaner

Run the setup cell first. Then run the AVT cell or the FVT cell separately:

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

The run is complete only when `status.json` contains `"status": "complete"` and `report.json` is present. Generated data are ignored by Git through `.gitignore` and should not be committed to the repository.

## Common problems

- `FileNotFoundError`: confirm that the ZIP is directly under `raw_data/` and its name starts with `sources-Z24`.
- The notebook appears busy but no files change: interrupt the cell, restart the kernel, and run only one measurement cell. The progress output shows the recording currently being read.
- A run folder has no `report.json`: that run was interrupted and is incomplete. Use a new timestamped run folder.
