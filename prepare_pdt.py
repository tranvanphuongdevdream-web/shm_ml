"""Clean Z24 Progressive Damage Test (PDT) MATLAB recordings.

The pipeline keeps condition labels from the PDT directory (01--17), preserves
measurement type/setup/channel metadata, and exports one CSV per segment.
It deliberately does not guess a global sensor mapping across setups.
"""
from __future__ import annotations

import csv
import io
import json
import re
import struct
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

class ArchiveSlice(io.RawIOBase):
    def __init__(self, path, start, size):
        self.file = open(path, 'rb')
        self.start, self.size, self.pos = start, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if pos < 0:
            raise ValueError('Negative seek')
        self.pos = pos
        return pos

    def read(self, size=-1):
        size = max(0, self.size - self.pos) if size < 0 else max(0, min(size, self.size - self.pos))
        self.file.seek(self.start + self.pos)
        result = self.file.read(size)
        self.pos += len(result)
        return result

    def close(self):
        self.file.close()
        super().close()


def inventory(path):
    """Read local ZIP headers; do not guess unsupported layouts."""
    rows = []
    total = Path(path).stat().st_size
    with open(path, 'rb') as f:
        while f.tell() + 30 <= total:
            offset = f.tell()
            header = f.read(30)
            if header[:4] != b'PK\x03\x04':
                break
            _, version, flags, method, _, _, crc, packed, size, nl, el = struct.unpack('<4s5H3I2H', header)
            name = f.read(nl).decode('utf-8' if flags & 2048 else 'cp437')
            f.read(el)
            if flags & 9 or packed == 0xffffffff:
                raise ValueError(f'Unsupported encrypted/descriptor/ZIP64 entry: {name}')
            start = f.tell()
            complete = start + packed <= total
            rows.append(dict(name=name, offset=offset, start=start, packed=packed,
                             size=size, method=method, complete=complete))
            if not complete:
                break
            f.seek(start + packed)
    return rows



SCENARIO_LABELS = {
    1: "undamaged_reference_1",
    2: "pier_settlement_system_installation",
    3: "pier_settlement_20mm",
    4: "pier_settlement_40mm",
    5: "pier_settlement_80mm",
    6: "pier_settlement_95mm",
    7: "foundation_tilt",
    8: "undamaged_reference_3",
    9: "concrete_spalling_12m2",
    10: "concrete_spalling_24m2",
    11: "abutment_landslide_1m",
    12: "concrete_hinge_failure",
    13: "failure_2_anchor_heads",
    14: "failure_4_anchor_heads",
    15: "tendon_rupture_2_of_16",
    16: "tendon_rupture_4_of_16",
    17: "tendon_rupture_6_of_16",
}


def _outer_member(path: Path, name: str):
    for member in inventory(path):
        if member["name"] == name:
            return member
    raise FileNotFoundError(f"Outer archive member not found: {name}")


def _read_mat(payload: bytes):
    mat = loadmat(io.BytesIO(payload), squeeze_me=True, struct_as_record=False,
                  verify_compressed_data_integrity=True)
    if "data" not in mat or "labelshulp" not in mat:
        raise ValueError("MAT file must contain data and labelshulp")
    data = np.asarray(mat["data"])
    names = np.atleast_1d(mat["labelshulp"]).astype(str).tolist()
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D data matrix, got {data.shape}")
    if data.shape[1] != len(names) and data.shape[0] == len(names):
        data = data.T
    if data.shape[1] != len(names):
        raise ValueError(f"Data shape {data.shape} does not match {len(names)} channel names")
    data = data.astype(np.float64, copy=False)
    return data, [name.strip() for name in names]


def _condition_setup(path: str):
    parts = Path(path).parts
    condition = next((int(part) for part in parts if part.isdigit() and 1 <= int(part) <= 17), None)
    match = re.search(r"setup(\d+)", Path(path).stem, flags=re.IGNORECASE)
    if condition is None or match is None:
        raise ValueError(f"Cannot parse condition/setup from {path}")
    measurement = "fvt" if any(part.lower() == "fvt" for part in parts) else "avt"
    return condition, int(match.group(1)), measurement


def _write_segment(path: Path, values: np.ndarray):
    header = ["sample_index"] + [f"sensor_{i}" for i in range(values.shape[1])]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(np.column_stack((np.arange(len(values)), values)).tolist())


def clean_pdt(archive, output, measurement="avt", max_conditions=None,
              max_setups=None, target_samples=60000, segment_samples=6000):
    """Clean PDT MAT files and export labeled CSV segments.

    One output row is one time sample and one output CSV is one segment. The
    condition directory is the label source; no label is inferred from signal
    values. By default only AVT is used, yielding up to 17 * 9 * 10 segments.
    """
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if measurement not in {"avt", "fvt", "both"}:
        raise ValueError("measurement must be 'avt', 'fvt', or 'both'")
    if target_samples <= 0 or segment_samples <= 0 or target_samples % segment_samples:
        raise ValueError("target_samples must be a positive multiple of segment_samples")
    if max_conditions is not None and max_conditions <= 0:
        raise ValueError("max_conditions must be positive or None")
    if max_setups is not None and max_setups <= 0:
        raise ValueError("max_setups must be positive or None")

    output.mkdir(parents=True)
    for folder in ("segments",):
        (output / folder).mkdir()
    records, issues = [], []
    condition_limit = set(range(1, max_conditions + 1)) if max_conditions else set(range(1, 18))
    package_names = ("pdt_01-08.zip", "pdt_09_17.zip")
    for package_name in package_names:
        try:
            member = _outer_member(archive, package_name)
            if not member["complete"] or member["method"] != 0:
                issues.append({"source": package_name, "reason": "Incomplete or unsupported outer member"})
                continue
            with ArchiveSlice(archive, member["start"], member["packed"]) as stream, zipfile.ZipFile(stream) as package:
                entries = [item for item in package.infolist()
                           if item.filename.lower().endswith(".mat")]
                for item in entries:
                    try:
                        condition, setup, kind = _condition_setup(item.filename)
                        if condition not in condition_limit or (measurement != "both" and kind != measurement):
                            continue
                        if max_setups is not None and setup > max_setups:
                            continue
                        data, channel_names = _read_mat(package.read(item))
                        finite = np.isfinite(data).all(axis=1)
                        clean = data[finite]
                        if len(clean) < target_samples:
                            raise ValueError(f"Only {len(clean)} finite rows; need {target_samples}")
                        clean = clean[:target_samples]
                        for segment_index, start in enumerate(range(0, target_samples, segment_samples)):
                            values = clean[start:start + segment_samples]
                            name = f"condition{condition:02d}_setup{setup:02d}_{kind}_segment{segment_index:02d}.csv"
                            relative = Path("segments") / name
                            _write_segment(output / relative, values)
                            records.append({
                                "file": str(relative),
                                "condition_id": condition,
                                "condition_name": SCENARIO_LABELS[condition],
                                "label": condition - 1,
                                "setup_id": setup,
                                "measurement": kind,
                                "segment": segment_index,
                                "start_sample": start,
                                "samples": segment_samples,
                                "channels": len(channel_names),
                                "channel_names": channel_names,
                                "source": f"{package_name}::{item.filename}",
                                "removed_rows": int((~finite).sum()),
                                "original_rows": int(data.shape[0]),
                            })
                    except (ValueError, KeyError, zipfile.BadZipFile, OSError) as error:
                        issues.append({"source": f"{package_name}::{item.filename}", "reason": str(error)})
        except (FileNotFoundError, ValueError, zipfile.BadZipFile, OSError) as error:
            issues.append({"source": package_name, "reason": str(error)})

    if not records:
        raise RuntimeError("No valid PDT recordings were found; inspect report.json")
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    report = {
        "archive": str(archive.resolve()),
        "output": str(output.resolve()),
        "measurement": measurement,
        "target_samples": target_samples,
        "segment_samples": segment_samples,
        "conditions": sorted({r["condition_id"] for r in records}),
        "recordings": len({r["source"] for r in records}),
        "segments": len(records),
        "csv_files": len(records),
        "issues": issues,
        "labeled": True,
        "label_definition": "PDT condition directory number minus one; verify scenario semantics in documentation",
        "sensor_mapping": "Preserved per setup; not forced into a global sensor order",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition_id", "label", "condition_name"])
        for condition, name in SCENARIO_LABELS.items():
            writer.writerow([condition, condition - 1, name])
    return report
