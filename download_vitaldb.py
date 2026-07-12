"""Download VitalDB PPG windows paired with intraoperative lab glucose readings.

VitalDB stores PPG as SNUADC/PLETH (500 Hz) and glucose as sporadic lab results
(name=gluc). Lab timestamps ``dt`` are in seconds from case start.
Each valid glucose draw is exported as one PPG window plus a label row, using
the same Labels/ + RawData/ layout as MUST so existing training scripts can
consume the dataset with ``--dataset vitaldb``.
"""

from __future__ import annotations

import argparse
import json
import signal
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

VITALDB_API = 'https://api.vitaldb.net'
PPG_TRACK = 'SNUADC/PLETH'
SOURCE_FS = 500.0
_INTERRUPTED = False


def _patch_ssl() -> None:
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
    urllib.request.install_opener(opener)


def _load_cases() -> pd.DataFrame:
    return pd.read_csv(f'{VITALDB_API}/cases')


def _load_tracks() -> pd.DataFrame:
    return pd.read_csv(f'{VITALDB_API}/trks')


def _load_labs() -> pd.DataFrame:
    return pd.read_csv(f'{VITALDB_API}/labs')


def _pleth_tid(tracks: pd.DataFrame, caseid: int) -> str | None:
    rows = tracks[(tracks['caseid'] == caseid) & (tracks['tname'] == PPG_TRACK)]
    if rows.empty:
        return None
    return str(rows.iloc[0]['tid'])


def _load_pleth_waveform(tid: str) -> np.ndarray:
    from vitaldb.dataset import load_trk

    waveform = load_trk(tid, interval=1 / SOURCE_FS)
    return np.asarray(waveform, dtype=np.float32).reshape(-1)


def _seconds_to_sample_index(dt_seconds: float) -> int:
    return int(round(float(dt_seconds) * SOURCE_FS))


# def _downsample(signal: np.ndarray, source_fs: float, target_fs: float) -> np.ndarray:
#     if target_fs <= 0 or source_fs <= 0:
#         raise ValueError('sampling rates must be positive')
#     if abs(source_fs - target_fs) < 1e-6:
#         return signal.astype(np.float32, copy=False)

#     target_length = max(1, int(round(signal.size * target_fs / source_fs)))
#     try:
#         from scipy.signal import resample

#         return resample(signal, target_length).astype(np.float32)
#     except Exception:
#         factor = max(1, int(round(source_fs / target_fs)))
#         return signal[::factor].astype(np.float32)


def _extract_window(
    pleth: np.ndarray,
    center_index: int,
    window_seconds: float,
) -> np.ndarray | None:
    half = int(round(window_seconds * SOURCE_FS / 2.0))
    start = center_index - half
    end = center_index + half
    if start < 0 or end > pleth.size:
        return None

    window = pleth[start:end]
    if window.size == 0 or np.all(np.isnan(window)):
        return None

    window = window.copy()
    if np.isnan(window).any():
        valid = np.flatnonzero(~np.isnan(window))
        if valid.size == 0:
            return None
        window[np.isnan(window)] = np.interp(
            np.flatnonzero(np.isnan(window)),
            valid,
            window[valid],
        )
    return window


def _sex_label(value) -> str:
    if pd.isna(value):
        return ''
    text = str(value).strip().upper()
    if text in {'M', 'MALE', '1', '1.0'}:
        return 'Male'
    if text in {'F', 'FEMALE', '0', '0.0'}:
        return 'Female'
    return str(value)


def _write_sample(
    output_dir: Path,
    caseid: int,
    sample_idx: int,
    glucose_mg_dl: float,
    ppg_window: np.ndarray,
    clinical_row: pd.Series,
) -> None:
    labels_dir = output_dir / 'Labels'
    raw_dir = output_dir / 'RawData'
    labels_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    session_key = f'{sample_idx:04d}'
    label_path = labels_dir / f'label_{caseid:04d}_{session_key}.csv'
    signal_path = raw_dir / f'signal_{caseid:04d}_{session_key}.csv'

    label_df = pd.DataFrame(
        [{
            'ID': int(caseid),
            'Gender': _sex_label(clinical_row.get('sex')),
            'Age': clinical_row.get('age'),
            'Glucose': float(glucose_mg_dl),
            'Height': clinical_row.get('height'),
            'Weight': clinical_row.get('weight'),
        }]
    )
    label_df.to_csv(label_path, index=False)
    pd.DataFrame({'ppg': ppg_window}).to_csv(signal_path, index=False, header=False)


def _sample_paths(output_dir: Path, caseid: int, sample_idx: int) -> tuple[Path, Path]:
    session_key = f'{sample_idx:04d}'
    label_path = output_dir / 'Labels' / f'label_{caseid:04d}_{session_key}.csv'
    signal_path = output_dir / 'RawData' / f'signal_{caseid:04d}_{session_key}.csv'
    return label_path, signal_path


def _sample_exists(output_dir: Path, caseid: int, sample_idx: int) -> bool:
    label_path, signal_path = _sample_paths(output_dir, caseid, sample_idx)
    return label_path.exists() and signal_path.exists()


def _latest_existing_sample(output_dir: Path) -> tuple[int, int] | None:
    latest: tuple[int, int] | None = None
    for label_path in (output_dir / 'Labels').glob('label_*.csv'):
        parts = label_path.stem.removeprefix('label_').split('_', 1)
        if len(parts) != 2:
            continue
        try:
            caseid = int(parts[0])
            sample_idx = int(parts[1])
        except ValueError:
            continue
        if not _sample_exists(output_dir, caseid, sample_idx):
            continue
        key = (caseid, sample_idx)
        if latest is None or key > latest:
            latest = key
    return latest


def _filtered_sample_indices(
    case_labs: pd.DataFrame,
    min_glucose_mg_dl: float | None,
    max_glucose_mg_dl: float | None,
) -> list[int]:
    indices: list[int] = []
    for sample_idx, (_, lab_row) in enumerate(case_labs.iterrows(), start=1):
        glucose = float(lab_row['result'])
        if not np.isfinite(glucose):
            continue
        if min_glucose_mg_dl is not None and glucose < min_glucose_mg_dl:
            continue
        if max_glucose_mg_dl is not None and glucose > max_glucose_mg_dl:
            continue
        indices.append(sample_idx)
    return indices


def download_vitaldb(
    output_dir: Path,
    window_seconds: float = 60.0,
    target_fs: float = 500.0,
    min_glucose_mg_dl: float | None = None,
    max_glucose_mg_dl: float | None = None,
    max_cases: int | None = None,
    max_samples: int | None = None,
    overwrite: bool = False,
) -> dict:
    global _INTERRUPTED
    _INTERRUPTED = False

    old_sigint_handler = signal.getsignal(signal.SIGINT)

    def _mark_interrupted(_signum, _frame):
        global _INTERRUPTED
        _INTERRUPTED = True
        print('\nInterrupted by user. Stopping download...')

    signal.signal(signal.SIGINT, _mark_interrupted)

    _patch_ssl()
    import vitaldb  # noqa: F401 - ensures package is available after SSL patch

    output_dir = output_dir.resolve()
    labels_dir = output_dir / 'Labels'
    raw_dir = output_dir / 'RawData'
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for path in labels_dir.glob('label_*.csv'):
            path.unlink()
        for path in raw_dir.glob('signal_*.csv'):
            path.unlink()
        print('Overwrite mode: cleared existing label/signal files before download')
        resume_cursor = None
    else:
        resume_cursor = _latest_existing_sample(output_dir)
        print('Resume mode: keeping existing files and skipping already downloaded samples')
        if resume_cursor is not None:
            print(
                f'Resume cursor: starting from case {resume_cursor[0]} sample {resume_cursor[1] + 1}'
            )

    cases = _load_cases()
    tracks = _load_tracks()
    labs = _load_labs()
    pleth_case_ids = set(vitaldb.find_cases(PPG_TRACK))

    gluc_labs = labs[labs['name'] == 'gluc'].copy()
    gluc_labs = gluc_labs[gluc_labs['caseid'].isin(pleth_case_ids)]
    gluc_labs = gluc_labs.sort_values(['caseid', 'dt'])

    grouped = gluc_labs.groupby('caseid', sort=True)
    case_ids = sorted(grouped.groups.keys())
    if max_cases is not None:
        case_ids = case_ids[:max_cases]

    samples_written = 0
    cases_written = 0
    skipped_cases = 0
    samples_skipped_existing = 0
    cases_skipped_existing = 0
    interrupted = False

    try:
        for caseid in case_ids:
            if _INTERRUPTED:
                interrupted = True
                break

            if max_samples is not None and samples_written >= max_samples:
                break

            if resume_cursor is not None and caseid < resume_cursor[0]:
                continue

            case_labs = grouped.get_group(caseid)
            filtered_indices = _filtered_sample_indices(case_labs, min_glucose_mg_dl, max_glucose_mg_dl)

            if not overwrite and filtered_indices:
                if all(_sample_exists(output_dir, caseid, sample_idx) for sample_idx in filtered_indices):
                    samples_skipped_existing += len(filtered_indices)
                    cases_skipped_existing += 1
                    print(
                        f'Skipping case {caseid}: all {len(filtered_indices)} filtered sample(s) already exist'
                    )
                    continue

            tid = _pleth_tid(tracks, caseid)
            if tid is None:
                skipped_cases += 1
                print(f'Skipping case {caseid}: no PPG track')
                continue

            clinical_rows = cases[cases['caseid'] == caseid]
            if clinical_rows.empty:
                skipped_cases += 1
                print(f'Skipping case {caseid}: no clinical row')
                continue
            clinical_row = clinical_rows.iloc[0]

            try:
                pleth = _load_pleth_waveform(tid)
            except KeyboardInterrupt:
                print('Interrupted by user. Stopping download.')
                interrupted = True
                break
            except Exception as exc:
                print(f'Skipping case {caseid}: failed to load PPG ({exc})')
                skipped_cases += 1
                continue

            if _INTERRUPTED:
                interrupted = True
                break

            case_samples = 0
            case_existing_skipped = 0
            for sample_idx, (_, lab_row) in enumerate(case_labs.iterrows(), start=1):
                if _INTERRUPTED:
                    interrupted = True
                    break

                if max_samples is not None and samples_written >= max_samples:
                    break

                if (
                    resume_cursor is not None
                    and caseid == resume_cursor[0]
                    and sample_idx <= resume_cursor[1]
                ):
                    continue

                glucose = float(lab_row['result'])
                if not np.isfinite(glucose):
                    continue
                if min_glucose_mg_dl is not None and glucose < min_glucose_mg_dl:
                    continue
                if max_glucose_mg_dl is not None and glucose > max_glucose_mg_dl:
                    continue

                if not overwrite and _sample_exists(output_dir, caseid, sample_idx):
                    samples_skipped_existing += 1
                    case_existing_skipped += 1
                    continue

                center = _seconds_to_sample_index(lab_row['dt'])
                window = _extract_window(pleth, center, window_seconds)
                if window is None:
                    continue

                # Keep VitalDB raw here; downsampling is handled later in the shared preprocessing pipeline.
                # window = _downsample(window, SOURCE_FS, target_fs)
                _write_sample(
                    output_dir,
                    caseid=caseid,
                    sample_idx=sample_idx,
                    glucose_mg_dl=glucose,
                    ppg_window=window,
                    clinical_row=clinical_row,
                )
                samples_written += 1
                case_samples += 1

            if interrupted:
                break

            if case_samples:
                cases_written += 1
                print(f'Case {caseid}: wrote {case_samples} sample(s)')
            elif case_existing_skipped:
                print(f'Case {caseid}: all filtered samples already existed ({case_existing_skipped})')
            else:
                print(f'Skipping case {caseid}: no usable glucose windows')
    finally:
        signal.signal(signal.SIGINT, old_sigint_handler)

    manifest = {
        'dataset': 'vitaldb',
        'ppg_track': PPG_TRACK,
        'source_fs_hz': SOURCE_FS,
        'target_fs_hz': target_fs,
        'window_seconds': window_seconds,
        'cases_written': cases_written,
        'samples_written': samples_written,
        'skipped_cases': skipped_cases,
        'cases_skipped_existing': cases_skipped_existing,
        'samples_skipped_existing': samples_skipped_existing,
        'overwrite': overwrite,
        'interrupted': interrupted,
        'glucose_unit': 'mg/dL',
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    if samples_written == 0 and samples_skipped_existing == 0 and not interrupted:
        raise RuntimeError('No VitalDB samples were written. Check network access and filters.')

    if interrupted:
        raise KeyboardInterrupt

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Download VitalDB PPG + glucose samples.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('dataset/vitaldb'),
        help='Directory for Labels/, RawData/, and manifest.json',
    )
    parser.add_argument('--window-seconds', type=float, default=60.0)
    parser.add_argument('--target-fs', type=float, default=500.0)
    parser.add_argument('--min-glucose', type=float, default=None)
    parser.add_argument('--max-glucose', type=float, default=None)
    parser.add_argument('--max-cases', type=int, default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Delete existing label/signal files before downloading (default is resume mode).',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    old_sigint_handler = signal.getsignal(signal.SIGINT)

    def _handle_sigint(_signum, _frame):
        print('\nInterrupted by user. Stopping download.')
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        manifest = download_vitaldb(
            output_dir=args.output_dir,
            window_seconds=args.window_seconds,
            target_fs=args.target_fs,
            min_glucose_mg_dl=args.min_glucose,
            max_glucose_mg_dl=args.max_glucose,
            max_cases=args.max_cases,
            max_samples=args.max_samples,
            overwrite=args.overwrite,
        )
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGINT, old_sigint_handler)

    print(json.dumps(manifest, indent=2))
    return 130 if manifest.get('interrupted') else 0


if __name__ == '__main__':
    sys.exit(main())
