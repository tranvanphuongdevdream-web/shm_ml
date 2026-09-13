"""Read intact EMS ZIP members without extracting the 10 GB outer archive."""
from pathlib import Path
import collections
import csv
import hashlib
import io
import json
import struct
import zipfile

import numpy as np


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


def read_aaa(payload):
    lines = payload.decode('ascii').splitlines()
    if len(lines) < 4:
        raise ValueError('Missing AAA header/data')
    header, count, dt = lines[0], int(lines[1]), float(lines[2])
    if count <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid sample count or interval')
    body = lines[3:]
    marker = next((i for i, line in enumerate(body) if line.strip() == 'Timehistories end here'), len(body))
    values = np.array([float(line) for line in body[:marker] if line.strip()], dtype=np.float64)
    if len(values) != count:
        raise ValueError(f'Sample count: expected {count}, found {len(values)}')
    if not np.isfinite(values).all():
        raise ValueError('NaN/Inf in recording; reject without interpolation')
    return header, dt, values, '\n'.join(body[marker:])


def group_split(session, seed=42):
    # All channels/windows from one session have the same split.
    score = int(hashlib.sha256(f'{seed}:{session}'.encode()).hexdigest()[:8], 16) / 2**32
    return 'train' if score < .7 else 'val' if score < .85 else 'test'


def write_windows_csv(path, windows, starts):
    """Write one row per window; sample columns retain time order."""
    windows = np.asarray(windows, dtype=np.float32)
    header = 'start_sample,' + ','.join(f'sample_{i}' for i in range(windows.shape[1]))
    np.savetxt(path, np.column_stack((starts, windows)), delimiter=',',
               header=header, comments='', fmt=['%d'] + ['%.9g'] * windows.shape[1],
               encoding='utf-8')


def read_windows_csv(path):
    """Return float32 X (windows, samples, 1) and integer start indices."""
    with open(path, encoding='utf-8', newline='') as f:
        columns = next(csv.reader(f))
    if len(columns) < 3 or columns != ['start_sample'] + [f'sample_{i}' for i in range(len(columns) - 1)]:
        raise ValueError(f'Unexpected CSV columns: {path}')
    values = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2, encoding='utf-8')
    if values.shape[1] != len(columns) or not np.isfinite(values).all():
        raise ValueError(f'Invalid CSV data: {path}')
    starts = values[:, 0]
    if np.any(starts < 0) or np.any(starts != np.floor(starts)):
        raise ValueError(f'Invalid start_sample: {path}')
    return values[:, 1:].astype(np.float32)[..., None], starts.astype(np.int64)


def prepare(archive, output, max_sessions=20, window_seconds=10, seed=42):
    """Process EMS only, one channel per example; never infer labels."""
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {output}')
    if max_sessions is not None and max_sessions <= 0:
        raise ValueError('max_sessions must be positive or None')
    if window_seconds <= 0:
        raise ValueError('window_seconds must be positive')
    members = inventory(archive)
    output.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        (output / split).mkdir()
    records, issues, stats = [], [], {}
    seen_sessions = set()
    for member in members:
        if not member['name'].lower().startswith('z24ems'):
            continue
        if not member['complete'] or member['method'] != 0:
            issues.append(dict(source=member['name'], reason='Incomplete or unsupported outer member; skipped'))
            continue
        with ArchiveSlice(archive, member['start'], member['packed']) as stream, zipfile.ZipFile(stream) as outer:
            for item in outer.infolist():
                if not item.filename.lower().endswith('.zip'):
                    continue
                if max_sessions is not None and len(seen_sessions) >= max_sessions:
                    break
                session = Path(item.filename).stem
                if session in seen_sessions:
                    issues.append(dict(source=item.filename, reason='Duplicate session ID; skipped'))
                    continue
                seen_sessions.add(session)
                split = group_split(session, seed)
                try:
                    session_bytes = outer.read(item)  # Checks this member's CRC.
                    with zipfile.ZipFile(io.BytesIO(session_bytes)) as inner:
                        environment = {i.filename: inner.read(i).decode('ascii', errors='replace')
                                       for i in inner.infolist() if i.filename.lower().endswith('.env')}
                        for entry in inner.infolist():
                            stem = Path(entry.filename).stem
                            # Numeric sensor suffix only; exclude car and other signal types.
                            if not entry.filename.lower().endswith('.aaa') or not stem.startswith(session) or not stem[len(session):].isdigit():
                                continue
                            channel = stem[len(session):]
                            try:
                                header, dt, values, footer = read_aaa(inner.read(entry))
                                if not np.isclose(dt, .01):
                                    raise ValueError(f'Unexpected dt={dt}; resampling requires review')
                                length = round(window_seconds / dt)
                                if length < 2:
                                    raise ValueError('Window too short')
                                n = len(values) // length
                                windows = values[:n * length].reshape(n, length)
                                good = np.ptp(windows, axis=1) > 0
                                starts = np.flatnonzero(good) * length
                                windows = windows[good]
                                if not len(windows):
                                    raise ValueError('No nonconstant complete windows')
                                if split == 'train':
                                    batch = (windows.size, float(windows.mean()), float(np.sum((windows - windows.mean()) ** 2)))
                                    count, mean, m2 = stats.get(channel, (0, 0., 0.))
                                    bc, bm, b2 = batch
                                    delta = bm - mean
                                    stats[channel] = (count + bc, mean + delta * bc / (count + bc), m2 + b2 + delta**2 * count * bc / (count + bc))
                                shard = f'{split}/{session}_{channel}.csv'
                                write_windows_csv(output / shard, windows, starts)
                                records.append(dict(shard=shard, session=session, channel=channel, split=split,
                                                    source=f"{member['name']}::{item.filename}::{entry.filename}",
                                                    header=header, footer=footer, header_name_mismatch=header.split('|')[0].lower() != Path(entry.filename).name.lower(),
                                                    fs=1 / dt, samples=len(values), windows=len(windows),
                                                    dropped_tail_samples=len(values) % length,
                                                    rejected_constant_windows=int((~good).sum()), environment=environment))
                            except (ValueError, UnicodeError, zipfile.BadZipFile) as e:
                                issues.append(dict(source=entry.filename, reason=str(e)))
                except (ValueError, zipfile.BadZipFile, RuntimeError) as e:
                    issues.append(dict(source=item.filename, reason=str(e)))
    normalization = {key: dict(count=int(n), mean=mean, std=float(np.sqrt(m2 / n)))
                     for key, (n, mean, m2) in stats.items()}
    for row in records:
        params = normalization.get(row['channel'])
        if params is None or params['std'] <= 0:
            row['normalized'] = False
            issues.append(dict(source=row['source'], reason='No valid train statistics for channel'))
            continue
        path = output / row['shard']
        data, starts = read_windows_csv(path)
        x = ((data.astype(np.float64) - params['mean']) / params['std']).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f'Nonfinite normalized output: {path}')
        write_windows_csv(path, x[..., 0], starts)
        row['normalized'] = True
    report = dict(archive=str(archive.resolve()), archive_bytes=archive.stat().st_size,
                  inventory=members, max_sessions=max_sessions, window_seconds=window_seconds, seed=seed,
                  sessions=len(seen_sessions), shards=len(records), windows=sum(r['windows'] for r in records),
                  split_windows=dict(collections.Counter({s: sum(r['windows'] for r in records if r['split'] == s) for s in ('train', 'val', 'test')})),
                  format='csv', csv_layout='start_sample,sample_0,...,sample_N',
                  labeled=False, issues=issues)
    (output / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    (output / 'normalization.json').write_text(json.dumps(normalization, indent=2), encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
