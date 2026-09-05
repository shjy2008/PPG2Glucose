"""Download VitalDB PPG windows paired with intraoperative lab glucose readings.

VitalDB stores PPG as SNUADC/PLETH (500 Hz) and glucose as sporadic lab results
(name=gluc). Lab timestamps ``dt`` are in seconds from case start.
Each valid glucose draw is exported as one PPG window plus a label row, using
the same Labels/ + RawData/ layout as MUST so existing training scripts can
consume the dataset with ``--dataset vitaldb``.
"""

# python download_vitaldb.py --output-dir dataset/vitaldb --window-seconds 60

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
COMPLETED_CASES_FILE = 'completed_cases.json'
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


def _parse_sample_stem(stem: str, prefix: str) -> tuple[int, int] | None:
    parts = stem.removeprefix(prefix).split('_', 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _index_existing_samples(output_dir: Path) -> set[tuple[int, int]]:
    """One filesystem scan; membership checks are O(1) afterwards."""
    labels: set[tuple[int, int]] = set()
    for path in (output_dir / 'Labels').glob('label_*.csv'):
        key = _parse_sample_stem(path.stem, 'label_')
        if key is not None:
            labels.add(key)

    existing: set[tuple[int, int]] = set()
    for path in (output_dir / 'RawData').glob('signal_*.csv'):
        key = _parse_sample_stem(path.stem, 'signal_')
        if key is not None and key in labels:
            existing.add(key)
    return existing


def _completed_cases_path(output_dir: Path) -> Path:
    return output_dir / COMPLETED_CASES_FILE


def _load_completed_cases(output_dir: Path, window_seconds: float) -> set[int]:
    path = _completed_cases_path(output_dir)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    if abs(float(data.get('window_seconds', -1)) - window_seconds) > 1e-6:
        return set()
    return {int(x) for x in data.get('case_ids', [])}


def _save_completed_cases(
    output_dir: Path,
    completed_case_ids: set[int],
    window_seconds: float,
) -> None:
    payload = {
        'window_seconds': window_seconds,
        'case_ids': sorted(completed_case_ids),
    }
    _completed_cases_path(output_dir).write_text(json.dumps(payload) + '\n')


def _case_duration_seconds(clinical_row: pd.Series) -> float | None:
    caseend = clinical_row.get('caseend')
    if caseend is None or pd.isna(caseend):
        return None
    duration = float(caseend)
    return duration if duration > 0 else None


def _window_fits_case(dt_seconds: float, window_seconds: float, case_duration: float | None) -> bool:
    half = window_seconds / 2.0
    if float(dt_seconds) < half:
        return False
    if case_duration is not None and float(dt_seconds) + half > case_duration:
        return False
    return True


def _filtered_labs(
    case_labs: pd.DataFrame,
    min_glucose_mg_dl: float | None,
    max_glucose_mg_dl: float | None,
) -> list[tuple[int, pd.Series]]:
    rows: list[tuple[int, pd.Series]] = []
    for sample_idx, (_, lab_row) in enumerate(case_labs.iterrows(), start=1):
        glucose = float(lab_row['result'])
        if not np.isfinite(glucose):
            continue
        if min_glucose_mg_dl is not None and glucose < min_glucose_mg_dl:
            continue
        if max_glucose_mg_dl is not None and glucose > max_glucose_mg_dl:
            continue
        rows.append((sample_idx, lab_row))
    return rows


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
        completed_path = _completed_cases_path(output_dir)
        if completed_path.exists():
            completed_path.unlink()
        existing_samples: set[tuple[int, int]] = set()
        completed_case_ids: set[int] = set()
        print('Overwrite mode: cleared existing label/signal files before download')
    else:
        print('Indexing existing samples for fast resume...')
        existing_samples = _index_existing_samples(output_dir)
        completed_case_ids = _load_completed_cases(output_dir, window_seconds)
        print(
            f'Resume mode: {len(existing_samples)} sample(s) on disk, '
            f'{len(completed_case_ids)} completed case(s); '
            f'starting from case 0 and skipping as needed'
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

    cases_by_id = cases.set_index('caseid', drop=False)

    samples_written = 0
    cases_written = 0
    skipped_cases = 0
    samples_skipped_existing = 0
    cases_skipped_existing = 0
    cases_skipped_complete = 0
    interrupted = False
    completed_dirty = 0

    def _mark_case_complete(caseid: int, *, flush: bool = False) -> None:
        nonlocal completed_dirty
        if caseid in completed_case_ids:
            return
        completed_case_ids.add(caseid)
        completed_dirty += 1
        if flush or completed_dirty >= 50:
            _save_completed_cases(output_dir, completed_case_ids, window_seconds)
            completed_dirty = 0

    try:
        for caseid in case_ids:
            if _INTERRUPTED:
                interrupted = True
                break

            if max_samples is not None and samples_written >= max_samples:
                break

            case_labs = grouped.get_group(caseid)
            filtered = _filtered_labs(case_labs, min_glucose_mg_dl, max_glucose_mg_dl)
            if not filtered:
                _mark_case_complete(caseid)
                continue

            existing_for_case = [
                sample_idx for sample_idx, _ in filtered
                if (caseid, sample_idx) in existing_samples
            ]
            missing = [
                (sample_idx, lab_row) for sample_idx, lab_row in filtered
                if (caseid, sample_idx) not in existing_samples
            ]

            if caseid not in cases_by_id.index:
                skipped_cases += 1
                print(f'Skipping case {caseid}: no clinical row')
                _mark_case_complete(caseid, flush=True)
                continue
            clinical_row = cases_by_id.loc[caseid]
            if isinstance(clinical_row, pd.DataFrame):
                clinical_row = clinical_row.iloc[0]

            if not missing:
                samples_skipped_existing += len(existing_for_case)
                cases_skipped_existing += 1
                _mark_case_complete(caseid)
                continue

            if caseid in completed_case_ids:
                samples_skipped_existing += len(existing_for_case)
                cases_skipped_complete += 1
                continue

            case_duration = _case_duration_seconds(clinical_row)
            candidates = [
                (sample_idx, lab_row) for sample_idx, lab_row in missing
                if _window_fits_case(float(lab_row['dt']), window_seconds, case_duration)
            ]
            if not candidates:
                # Remaining labs cannot fit this window; no PPG download needed.
                samples_skipped_existing += len(existing_for_case)
                _mark_case_complete(caseid)
                if not existing_for_case:
                    skipped_cases += 1
                continue

            tid = _pleth_tid(tracks, caseid)
            if tid is None:
                skipped_cases += 1
                print(f'Skipping case {caseid}: no PPG track')
                _mark_case_complete(caseid, flush=True)
                continue

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
            for sample_idx, lab_row in candidates:
                if _INTERRUPTED:
                    interrupted = True
                    break

                if max_samples is not None and samples_written >= max_samples:
                    break

                glucose = float(lab_row['result'])
                center = _seconds_to_sample_index(lab_row['dt'])
                window = _extract_window(pleth, center, window_seconds)
                if window is None:
                    continue

                _write_sample(
                    output_dir,
                    caseid=caseid,
                    sample_idx=sample_idx,
                    glucose_mg_dl=glucose,
                    ppg_window=window,
                    clinical_row=clinical_row,
                )
                existing_samples.add((caseid, sample_idx))
                samples_written += 1
                case_samples += 1

            if interrupted:
                break

            # Case fully attempted (wrote 0 or more); do not reload PPG next resume.
            _mark_case_complete(caseid, flush=True)
            samples_skipped_existing += len(existing_for_case)

            if case_samples:
                cases_written += 1
                print(f'Case {caseid}: wrote {case_samples} sample(s)')
            elif not existing_for_case:
                skipped_cases += 1
                print(f'Skipping case {caseid}: no usable glucose windows')
    finally:
        signal.signal(signal.SIGINT, old_sigint_handler)
        if not overwrite:
            _save_completed_cases(output_dir, completed_case_ids, window_seconds)

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
        'cases_skipped_complete': cases_skipped_complete,
        'samples_skipped_existing': samples_skipped_existing,
        'completed_cases': len(completed_case_ids),
        'existing_samples_on_disk': len(existing_samples),
        'overwrite': overwrite,
        'interrupted': interrupted,
        'glucose_unit': 'mg/dL',
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    if samples_written == 0 and samples_skipped_existing == 0 and cases_skipped_complete == 0 and not interrupted:
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
