from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils import (
    exclude_target_outlier_arrays,
    mg_dl_to_mmol_l,
    safe_float,
)
from config import (
    DEFAULT_WINDOW_SECONDS,
    PPG_BEST_SEGMENT_HOP_SECONDS,
    PPG_SQA_WINDOW_SECONDS,
    basic_feature_keys,
    ppg_best_segment_seconds,
    ppg_feature_groups,
    ppg_filter,
    ppg_sampling_rate,
    ppg_sqa_threshold,
    ppg_target_fs,
    ppg_trim_fractions,
    resolve_ppg_center_window_seconds,
)
from data_basic_info import extract_basic_info
from data_ppg import (
    extract_raw_ppg,
    extract_raw_ppg_windows,
    read_signal_waveform,
    extract_ppg_bpm,
)
# from data_ppg_basic import extract_raw_ppg, read_signal_waveform, extract_ppg_bpm  # performance worse than data_ppg
from sklearn.model_selection import KFold

FEATURE_CACHE_DIR = Path(__file__).resolve().parent / 'feature_cache'
WAVEFORM_CACHE_DIR = Path(__file__).resolve().parent / 'waveform_cache'

_DEMOGRAPHIC_COLUMNS = ('sex', 'age', 'height', 'weight', 'bmi')


def _feature_cache_path(dataset_name: str, fingerprint: dict) -> Path:
    key = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return FEATURE_CACHE_DIR / f'{dataset_name}_{key}.pkl'


def _waveform_cache_path(dataset_name: str, fingerprint: dict) -> Path:
    key = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    WAVEFORM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return WAVEFORM_CACHE_DIR / f'{dataset_name}_{key}.npz'


def _waveform_cache_fingerprint(
    dataset_name: str,
    root: Path,
    *,
    signal_length: int,
    include_basic: bool,
    resample: bool,
    max_samples: int | None,
    ppg_seconds: float | None,
) -> dict:
    center = resolve_ppg_center_window_seconds(ppg_seconds)
    return {
        'kind': 'waveform',
        'dataset': dataset_name,
        'root': str(root.resolve()),
        'signal_length': int(signal_length),
        'include_basic': bool(include_basic),
        'basic_feature_keys': list(basic_feature_keys(dataset_name)) if include_basic else [],
        'resample': bool(resample),
        'max_samples': max_samples,
        'ppg_seconds': ppg_seconds,
        'ppg_center_window_seconds': center,
        'ppg_sampling_rate': ppg_sampling_rate(dataset_name),
        'ppg_target_fs': ppg_target_fs(dataset_name),
        'ppg_trim_fractions': list(ppg_trim_fractions(dataset_name)),
        'ppg_filter': ppg_filter(dataset_name),
    }


def _try_load_waveform_cache(
    cache_path: Path,
    refresh_cache: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if refresh_cache or not cache_path.is_file():
        return None
    data = np.load(cache_path, allow_pickle=False)
    if 'sessions' not in data.files:
        return None
    signals = np.asarray(data['signals'], dtype=np.float32)
    demographics = np.asarray(data['demographics'], dtype=np.float32)
    targets = np.asarray(data['targets'], dtype=np.float32)
    groups = np.asarray(data['groups'])
    sessions = np.asarray(data['sessions']).astype(str)
    print(f'Loaded waveforms from {cache_path.name} ({len(targets)} rows)')
    return signals, demographics, targets, groups, sessions


def _save_waveform_cache(
    cache_path: Path,
    signals: np.ndarray,
    demographics: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    sessions: np.ndarray,
    fingerprint: dict,
) -> None:
    np.savez_compressed(
        cache_path,
        signals=np.asarray(signals, dtype=np.float32),
        demographics=np.asarray(demographics, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        groups=np.asarray(groups),
        sessions=np.asarray(sessions, dtype='U32'),
    )
    cache_path.with_suffix('.meta.json').write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True, default=str)
    )
    print(f'Saved waveform cache to {cache_path.name} ({len(targets)} rows)')


def _ppg_cache_fingerprint(
    dataset_name: str,
    window_seconds: float,
    max_samples: int | None,
    identity: dict,
    center_window_seconds: float | None = None,
) -> dict:
    # Demographics are not part of the PPG cache key; they are joined at load time.
    return {
        'dataset': dataset_name,
        'window_seconds': float(window_seconds or 0.0),
        'max_samples': max_samples,
        'ppg_sampling_rate': ppg_sampling_rate(dataset_name),
        'ppg_target_fs': ppg_target_fs(dataset_name),
        'ppg_feature_groups': list(ppg_feature_groups(dataset_name)),
        'ppg_trim_fractions': list(ppg_trim_fractions(dataset_name)),
        'ppg_filter': ppg_filter(dataset_name),
        'ppg_best_segment_seconds': ppg_best_segment_seconds(dataset_name),
        'ppg_best_segment_hop_seconds': PPG_BEST_SEGMENT_HOP_SECONDS,
        'ppg_sqa_threshold': ppg_sqa_threshold(dataset_name),
        'ppg_sqa_window_seconds': PPG_SQA_WINDOW_SECONDS,
        'ppg_center_window_seconds': center_window_seconds,
        'identity': identity,
    }


def _ppg_only_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in _DEMOGRAPHIC_COLUMNS if c in df.columns]
    return df.drop(columns=drop) if drop else df


def _fingerprint_without_demos(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if k != 'basic_feature_keys'}


def _try_load_feature_cache(
    cache_path: Path,
    refresh_cache: bool,
    fingerprint: dict | None = None,
    dataset_name: str | None = None,
) -> pd.DataFrame | None:
    if refresh_cache:
        return None
    if cache_path.is_file():
        df = pd.read_pickle(cache_path)
        print(f'Loaded PPG features from {cache_path.name} ({len(df)} rows)')
        return _ppg_only_frame(df)

    # Reuse older caches whose meta still listed basic_feature_keys.
    if fingerprint is None or dataset_name is None or not FEATURE_CACHE_DIR.is_dir():
        return None
    want = _fingerprint_without_demos(fingerprint)
    for meta_path in FEATURE_CACHE_DIR.glob(f'{dataset_name}_*.meta.json'):
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if _fingerprint_without_demos(meta) != want:
            continue
        pkl_path = meta_path.with_name(meta_path.name.removesuffix('.meta.json') + '.pkl')
        if not pkl_path.is_file():
            continue
        df = pd.read_pickle(pkl_path)
        print(f'Loaded PPG features from legacy {pkl_path.name} ({len(df)} rows)')
        return _ppg_only_frame(df)
    return None


def _save_feature_cache(cache_path: Path, df: pd.DataFrame, fingerprint: dict) -> None:
    ppg_df = _ppg_only_frame(df)
    ppg_df.to_pickle(cache_path)
    cache_path.with_suffix('.meta.json').write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True, default=str),
        encoding='utf-8',
    )
    print(f'Saved PPG feature cache to {cache_path.name} ({len(ppg_df)} rows)')


def _attach_labels_rawdata_basic(
    df: pd.DataFrame,
    labels_dir: Path,
    dataset_name: str,
) -> pd.DataFrame:
    out = _ppg_only_frame(df)
    keys = basic_feature_keys(dataset_name)
    if not keys:
        return out.reset_index(drop=True)

    demo_records = []
    for label_path in labels_dir.glob('label_*.csv'):
        try:
            _, session_key = _parse_label_stem(label_path)
        except ValueError:
            continue
        label_df = pd.read_csv(label_path)
        if label_df.empty:
            continue
        row = label_df.iloc[0]
        subject_id = safe_float(row['ID'])
        if np.isnan(subject_id):
            continue
        info = extract_basic_info(row, keys)
        if info:
            demo_records.append(
                {'subject_ID': int(subject_id), 'session_ID': str(session_key), **info}
            )
    if not demo_records:
        raise RuntimeError(f'No usable demographics {list(keys)} under {labels_dir}')
    merged = out.assign(session_ID=out['session_ID'].astype(str)).merge(
        pd.DataFrame(demo_records), on=['subject_ID', 'session_ID'], how='inner'
    )
    dropped = len(out) - len(merged)
    if dropped:
        print(f'Dropped {dropped} row(s) missing demographics {list(keys)}')
    if merged.empty:
        raise RuntimeError(f'All rows dropped attaching demographics {list(keys)}')
    # Keep historical column order: IDs/target, then demos, then PPG features.
    id_cols = [
        c for c in ('subject_ID', 'session_ID', 'window_ID', 'target') if c in merged.columns
    ]
    ppg_cols = [c for c in out.columns if c not in id_cols]
    return merged.loc[:, id_cols + list(keys) + ppg_cols].reset_index(drop=True)


def _attach_IPS_basic(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    out = _ppg_only_frame(df)
    keys = basic_feature_keys('IPS')
    if not keys:
        return out.reset_index(drop=True)

    metadata = pd.read_excel(root / 'bg_re.xlsx', sheet_name=0, header=1)
    demo_records = []
    for _, row in metadata[metadata['subject_ID'].notna()].iterrows():
        subject_id = safe_float(row['subject_ID'])
        if np.isnan(subject_id):
            continue
        info = extract_basic_info(row, keys)
        if info:
            demo_records.append({'subject_ID': int(subject_id), **info})
    if not demo_records:
        raise RuntimeError(f'No usable demographics {list(keys)} in {root / "bg_re.xlsx"}')
    merged = out.merge(
        pd.DataFrame(demo_records).drop_duplicates('subject_ID'),
        on='subject_ID',
        how='inner',
    )
    dropped = len(out) - len(merged)
    if dropped:
        print(f'Dropped {dropped} row(s) missing demographics {list(keys)}')
    if merged.empty:
        raise RuntimeError(f'All rows dropped attaching demographics {list(keys)}')
    id_cols = [c for c in ('subject_ID', 'window_ID', 'target') if c in merged.columns]
    ppg_cols = [c for c in out.columns if c not in id_cols]
    return merged.loc[:, id_cols + list(keys) + ppg_cols].reset_index(drop=True)


def _ppg_feature_dicts(
    signal_path: Path,
    fs: float,
    window_seconds: float,
    dataset_name: str,
    center_window_seconds: float | None = None,
) -> list[dict]:
    """Full-signal features when window_seconds<=0; else one dict per window."""
    trim_fractions = ppg_trim_fractions(dataset_name)
    filter_name = ppg_filter(dataset_name)
    target_fs = ppg_target_fs(dataset_name)
    best_segment_seconds = ppg_best_segment_seconds(dataset_name)
    feature_groups = ppg_feature_groups(dataset_name)
    sqa_threshold = ppg_sqa_threshold(dataset_name)
    if center_window_seconds is None:
        center_window_seconds = resolve_ppg_center_window_seconds()
    if window_seconds is None or window_seconds <= 0:
        features = extract_raw_ppg(
            signal_path,
            fs=fs,
            trim_fractions=trim_fractions,
            filter_name=filter_name,
            target_fs=target_fs,
            best_segment_seconds=best_segment_seconds,
            feature_groups=feature_groups,
            sqa_threshold=sqa_threshold,
            sqa_window_seconds=PPG_SQA_WINDOW_SECONDS,
            center_window_seconds=center_window_seconds,
        )
        return [] if features is None else [features]
    return extract_raw_ppg_windows(
        signal_path,
        fs=fs,
        window_seconds=window_seconds,
        trim_fractions=trim_fractions,
        filter_name=filter_name,
        target_fs=target_fs,
        best_segment_seconds=best_segment_seconds,
        feature_groups=feature_groups,
        sqa_threshold=sqa_threshold,
        center_window_seconds=center_window_seconds,
    )


def _config_dataset_name(dataset: str | Path | None, layout: str) -> str:
    """Prefer CLI name (vitaldb) over layout for config lookups."""
    from config import DATASET_CONFIG_PROFILE, DATASET_ROOTS

    if dataset is None:
        return layout
    key = dataset.name if isinstance(dataset, Path) else str(dataset)
    if key in DATASET_ROOTS or key in DATASET_CONFIG_PROFILE:
        return key
    name = Path(key).name
    if name in DATASET_ROOTS or name in DATASET_CONFIG_PROFILE:
        return name
    return layout


def resolve_dataset_root(dataset: str | Path | None = None) -> Path:
    from config import DATASET_ROOTS

    if dataset is None:
        dataset = 'IPS'
    if isinstance(dataset, Path):
        root = dataset
    elif str(dataset) in DATASET_ROOTS:
        root = DATASET_ROOTS[str(dataset)]
    else:
        root = Path(dataset)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    return root.resolve()


def dataset_layout(root: Path) -> str:
    if (root / 'manifest.json').is_file() and (root / 'Labels').is_dir() and (root / 'RawData').is_dir():
        return 'vitaldb'
    if (root / 'Labels').is_dir() and (root / 'RawData').is_dir():
        return 'MUST'
    if (root / 'bg_re.xlsx').is_file() and (root / 'ppg').is_dir():
        return 'IPS'
    raise RuntimeError(
        f'Unrecognized dataset layout under {root}. '
        'Expected vitaldb/MUST (Labels/ + RawData/) or IPS (bg_re.xlsx + ppg/).'
    )


def _parse_label_stem(label_path: Path) -> tuple[str, str]:
    """Return (subject_id, session_id) from label_01_0001.csv."""
    parts = label_path.stem.removeprefix('label_').split('_', 1)
    if len(parts) != 2:
        raise ValueError(f'Unexpected label filename: {label_path.name}')
    return parts[0], parts[1]


def _signal_path(raw_data_dir: Path, subject_id: str, session_id: str) -> Path:
    return raw_data_dir / f'signal_{subject_id}_{session_id}.csv'


def load_dataset(
    dataset: str | Path | None = None,
    labels_dir: Path | None = None,
    raw_data_dir: Path | None = None,
    max_samples: int | None = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    refresh_cache: bool = False,
    ppg_seconds: float | None = None,
) -> pd.DataFrame:
    if labels_dir is not None or raw_data_dir is not None:
        if labels_dir is None or raw_data_dir is None:
            raise ValueError('labels_dir and raw_data_dir must both be provided')
        layout = 'MUST'
        return _load_labels_rawdata_dataset(
            labels_dir,
            raw_data_dir,
            layout,
            max_samples=max_samples,
            window_seconds=window_seconds,
            refresh_cache=refresh_cache,
            ppg_seconds=ppg_seconds,
        )

    root = resolve_dataset_root(dataset)
    layout = dataset_layout(root)
    config_name = _config_dataset_name(dataset, layout)
    if layout in {'MUST', 'vitaldb'}:
        return _load_labels_rawdata_dataset(
            root / 'Labels',
            root / 'RawData',
            config_name,
            max_samples=max_samples,
            window_seconds=window_seconds,
            refresh_cache=refresh_cache,
            ppg_seconds=ppg_seconds,
        )
    return _load_IPS_dataset(
        root,
        window_seconds=window_seconds,
        max_samples=max_samples,
        refresh_cache=refresh_cache,
        ppg_seconds=ppg_seconds,
    )


def _load_labels_rawdata_dataset(
    labels_dir: Path,
    raw_data_dir: Path,
    dataset_name: str,
    max_samples: int | None = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    refresh_cache: bool = False,
    ppg_seconds: float | None = None,
) -> pd.DataFrame:
    label_files = sorted(labels_dir.glob('label_*.csv'))
    if not label_files:
        raise RuntimeError(f'No label files found in {labels_dir}')

    center_seconds = resolve_ppg_center_window_seconds(ppg_seconds)
    identity = {
        'n_label_files': len(label_files),
        'max_label_mtime': round(max(p.stat().st_mtime for p in label_files), 3),
        'raw_data_mtime': round(raw_data_dir.stat().st_mtime, 3),
    }
    fingerprint = _ppg_cache_fingerprint(
        dataset_name,
        window_seconds,
        max_samples,
        identity,
        center_window_seconds=center_seconds,
    )
    cache_path = _feature_cache_path(dataset_name, fingerprint)
    cached = _try_load_feature_cache(
        cache_path, refresh_cache, fingerprint=fingerprint, dataset_name=dataset_name
    )
    if cached is not None:
        return _attach_labels_rawdata_basic(cached, labels_dir, dataset_name)

    if max_samples is not None:
        label_files = label_files[:max_samples]
        print(f'Limited to {len(label_files)} sample(s)')

    n_files = len(label_files)
    progress_every = max(1, n_files // 20)  # ~5% steps
    print(f'Loading {n_files} sample(s) from {labels_dir} (PPG feature extraction)...')
    if center_seconds:
        print(f'Center-cropping PPG to {center_seconds:g} s ({dataset_name})')
    else:
        print(f'Using full on-disk PPG signal ({dataset_name})')

    records = []
    for i, label_path in enumerate(label_files, start=1):
        try:
            subject_key, session_key = _parse_label_stem(label_path)
        except ValueError as exc:
            print(exc)
            continue

        label_df = pd.read_csv(label_path)
        if label_df.empty:
            print('Skipping empty label file', label_path.name)
            continue

        row = label_df.iloc[0]
        subject_id = safe_float(row['ID'])
        target = mg_dl_to_mmol_l(safe_float(row['Glucose']))
        if np.isnan(subject_id) or np.isnan(target):
            print('Skipping', label_path.name, 'with invalid subject ID or target', target)
            continue

        subject_id_int = int(subject_id)

        signal_path = _signal_path(raw_data_dir, subject_key, session_key)
        if not signal_path.exists():
            print('Skipping', label_path.name, 'because signal file does not exist:', signal_path.name)
            continue
        feature_dicts = _ppg_feature_dicts(
            signal_path,
            fs=ppg_sampling_rate(dataset_name),
            window_seconds=window_seconds,
            dataset_name=dataset_name,
            center_window_seconds=center_seconds,
        )
        if not feature_dicts:
            print('Skipping', label_path.name, 'because signal is empty or failed SQA')
            continue

        for window_id, raw_ppg_dict in enumerate(feature_dicts):
            record = {
                'subject_ID': subject_id_int,
                'session_ID': session_key,
                'target': target,
            }
            if window_seconds is not None and window_seconds > 0:
                record['window_ID'] = window_id
            record.update(raw_ppg_dict)
            records.append(record)

        if i == 1 or i == n_files or i % progress_every == 0:
            print(f'  loaded {i}/{n_files} files ({len(records)} usable)', flush=True)

    dataset = pd.DataFrame(records)
    if dataset.empty:
        raise RuntimeError(
            'No usable rows were found. Check Labels/*.csv and matching RawData/signal_*.csv files.'
        )

    print(f'Finished PPG extraction: {len(dataset)} usable sample(s)', flush=True)
    _save_feature_cache(cache_path, dataset, fingerprint)
    return _attach_labels_rawdata_basic(dataset, labels_dir, dataset_name)


def _load_IPS_dataset(
    root: Path,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    max_samples: int | None = None,
    refresh_cache: bool = False,
    ppg_seconds: float | None = None,
) -> pd.DataFrame:
    ppg_dir = root / 'ppg'
    ppg_files = list(ppg_dir.glob('*.txt')) if ppg_dir.is_dir() else []
    center_seconds = resolve_ppg_center_window_seconds(ppg_seconds)
    identity = {
        'n_ppg_files': len(ppg_files),
        'max_ppg_mtime': round(max((p.stat().st_mtime for p in ppg_files), default=0.0), 3),
        'bg_re_mtime': round((root / 'bg_re.xlsx').stat().st_mtime, 3),
    }
    fingerprint = _ppg_cache_fingerprint(
        'IPS',
        window_seconds,
        max_samples,
        identity,
        center_window_seconds=center_seconds,
    )
    cache_path = _feature_cache_path('IPS', fingerprint)
    cached = _try_load_feature_cache(
        cache_path, refresh_cache, fingerprint=fingerprint, dataset_name='IPS'
    )
    if cached is not None:
        return _attach_IPS_basic(cached, root)

    metadata = pd.read_excel(root / 'bg_re.xlsx', sheet_name=0, header=1)
    rows = metadata[metadata['subject_ID'].notna()]
    if max_samples is not None:
        rows = rows.iloc[:max_samples]
        print(f'Limited to {len(rows)} sample(s)')

    if center_seconds:
        print(f'Center-cropping PPG to {center_seconds:g} s (IPS)')
    else:
        print('Using full on-disk PPG signal (IPS)')

    records = []
    for _, row in rows.iterrows():
        subject_id = safe_float(row['subject_ID'])
        target = safe_float(row['Diabetes'])
        if np.isnan(subject_id) or np.isnan(target):
            print('Skipping subject', row['subject_ID'], 'with invalid ID or target', target)
            continue

        subject_id_int = int(subject_id)
        signal_path = ppg_dir / f'{subject_id_int}.txt'
        if not signal_path.exists():
            print('Skipping subject', subject_id_int, 'because signal file does not exist:', signal_path.name)
            continue

        feature_dicts = _ppg_feature_dicts(
            signal_path,
            fs=ppg_sampling_rate('IPS'),
            window_seconds=window_seconds,
            dataset_name='IPS',
            center_window_seconds=center_seconds,
        )
        if not feature_dicts:
            print('Skipping subject', subject_id_int, 'because signal is empty or failed SQA')
            continue

        for window_id, raw_ppg_dict in enumerate(feature_dicts):
            record = {
                'subject_ID': subject_id_int,
                'target': target,
            }
            if window_seconds is not None and window_seconds > 0:
                record['window_ID'] = window_id
            record.update(raw_ppg_dict)
            # PPG_BPM is not useful, including it cause worse performance (tested through 210 subjects with ppg_bpm)
            # ppg_bpm = extract_ppg_bpm(row)
            # if not ppg_bpm:
            #     continue
            # record.update(ppg_bpm)
            records.append(record)

    if not records:
        raise RuntimeError(
            f'No usable rows were found. Check bg_re.xlsx and matching {ppg_dir.name}/*.txt files.'
        )
    dataset = pd.DataFrame(records)
    _save_feature_cache(cache_path, dataset, fingerprint)
    return _attach_IPS_basic(dataset, root)


def _resize_waveform(waveform: np.ndarray, target_length: int, resample: bool) -> np.ndarray:
    if waveform.size >= target_length:
        if resample:
            indices = np.linspace(0, waveform.size - 1, target_length).astype(int)
            return waveform[indices]
        return waveform[:target_length]
    return np.pad(waveform, (0, target_length - waveform.size), mode='constant')


def load_waveform_dataset(
    dataset: str | Path | None = None,
    labels_dir: Path | None = None,
    raw_data_dir: Path | None = None,
    signal_length: int = 0,
    include_basic: bool = True,
    resample: bool = False,
    max_samples: int | None = None,
    return_groups: bool = False,
    return_sessions: bool = False,
    ppg_seconds: float | None = None,
    refresh_cache: bool = False,
) -> tuple[np.ndarray, ...]:
    """Load per-session PPG waveforms and glucose targets (mmol/L).

    Returns:
        signals: (n_samples, signal_length)
        demographics: (n_samples, 4) or (n_samples, 0) when include_basic is False
        targets: (n_samples,)
        groups: subject IDs, returned only when return_groups is True
        sessions: recording IDs, returned only when return_sessions is True

    Caches resized waveforms under ``waveform_cache/`` (like feature_cache for CatBoost).
    """
    cache_path = None
    fingerprint = None

    if labels_dir is not None or raw_data_dir is not None:
        if labels_dir is None or raw_data_dir is None:
            raise ValueError('labels_dir and raw_data_dir must both be provided')
        config_name = 'MUST'
        root = Path(labels_dir).resolve().parent
        fingerprint = _waveform_cache_fingerprint(
            config_name,
            root,
            signal_length=signal_length,
            include_basic=include_basic,
            resample=resample,
            max_samples=max_samples,
            ppg_seconds=ppg_seconds,
        )
        cache_path = _waveform_cache_path(config_name, fingerprint)
        cached = _try_load_waveform_cache(cache_path, refresh_cache)
        if cached is not None:
            signals, demographics, targets, groups, sessions = cached
        else:
            signals, demographics, targets, groups, sessions = _load_labels_rawdata_waveform_dataset(
                labels_dir,
                raw_data_dir,
                config_name,
                signal_length=signal_length,
                include_basic=include_basic,
                resample=resample,
                max_samples=max_samples,
                ppg_seconds=ppg_seconds,
            )
            _save_waveform_cache(
                cache_path, signals, demographics, targets, groups, sessions, fingerprint
            )
    else:
        root = resolve_dataset_root(dataset)
        layout = dataset_layout(root)
        config_name = _config_dataset_name(dataset, layout)
        fingerprint = _waveform_cache_fingerprint(
            config_name,
            root,
            signal_length=signal_length,
            include_basic=include_basic,
            resample=resample,
            max_samples=max_samples,
            ppg_seconds=ppg_seconds,
        )
        cache_path = _waveform_cache_path(config_name, fingerprint)
        cached = _try_load_waveform_cache(cache_path, refresh_cache)
        if cached is not None:
            signals, demographics, targets, groups, sessions = cached
        elif layout in {'MUST', 'vitaldb'}:
            signals, demographics, targets, groups, sessions = _load_labels_rawdata_waveform_dataset(
                root / 'Labels',
                root / 'RawData',
                config_name,
                signal_length=signal_length,
                include_basic=include_basic,
                resample=resample,
                max_samples=max_samples,
                ppg_seconds=ppg_seconds,
            )
            _save_waveform_cache(
                cache_path, signals, demographics, targets, groups, sessions, fingerprint
            )
        else:
            signals, demographics, targets, groups, sessions = _load_IPS_waveform_dataset(
                root,
                signal_length=signal_length,
                include_basic=include_basic,
                resample=resample,
                ppg_seconds=ppg_seconds,
            )
            _save_waveform_cache(
                cache_path, signals, demographics, targets, groups, sessions, fingerprint
            )

    targets, signals, demographics, groups, sessions = exclude_target_outlier_arrays(
        targets,
        signals,
        demographics,
        groups,
        sessions,
    )
    if return_groups and return_sessions:
        return signals, demographics, targets, groups, sessions
    if return_groups:
        return signals, demographics, targets, groups
    if return_sessions:
        return signals, demographics, targets, sessions
    return signals, demographics, targets


def _load_labels_rawdata_waveform_dataset(
    labels_dir: Path,
    raw_data_dir: Path,
    dataset_name: str,
    signal_length: int = 0,
    include_basic: bool = True,
    resample: bool = False,
    max_samples: int | None = None,
    ppg_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_files = sorted(labels_dir.glob('label_*.csv'))
    if not label_files:
        raise RuntimeError(f'No label files found in {labels_dir}')
    if max_samples is not None:
        label_files = label_files[:max_samples]
        print(f'Limited to {len(label_files)} sample(s)')

    center_seconds = resolve_ppg_center_window_seconds(ppg_seconds)
    if center_seconds:
        print(
            f'Center-cropping PPG to {center_seconds:g} s '
            f'({dataset_name}, fs={ppg_sampling_rate(dataset_name):g} Hz)'
        )
    else:
        print(f'Using full on-disk PPG signal ({dataset_name})')

    pending = []
    lengths = []
    for label_path in label_files:
        try:
            subject_key, session_key = _parse_label_stem(label_path)
        except ValueError as exc:
            print(exc)
            continue

        label_df = pd.read_csv(label_path)
        if label_df.empty:
            print('Skipping empty label file', label_path.name)
            continue

        row = label_df.iloc[0]
        subject_id = safe_float(row['ID'])
        target = mg_dl_to_mmol_l(safe_float(row['Glucose']))
        if np.isnan(subject_id) or np.isnan(target):
            print('Skipping', label_path.name, 'with invalid subject ID or target', target)
            continue

        if include_basic:
            keys = basic_feature_keys(dataset_name)
            basic_info_dict = extract_basic_info(row, keys)
            if keys and not basic_info_dict:
                continue
            basic_values = np.asarray(
                [basic_info_dict[key] for key in keys],
                dtype=np.float32,
            )
        else:
            basic_values = np.empty(0, dtype=np.float32)

        signal_path = _signal_path(raw_data_dir, subject_key, session_key)
        if not signal_path.exists():
            print('Skipping', label_path.name, 'because signal file does not exist:', signal_path.name)
            continue

        waveform = read_signal_waveform(
            signal_path,
            trim_fractions=ppg_trim_fractions(dataset_name),
            fs=ppg_sampling_rate(dataset_name),
            filter_name=ppg_filter(dataset_name),
            target_fs=ppg_target_fs(dataset_name),
            center_window_seconds=center_seconds,
        ).astype(np.float32, copy=False)
        if waveform.size == 0:
            print('Skipping', label_path.name, 'because signal is empty')
            continue

        pending.append((waveform, basic_values, target, int(subject_id), str(session_key)))
        lengths.append(int(waveform.size))

    if not pending:
        raise RuntimeError(
            'No usable waveform rows were found. Check Labels/*.csv and matching RawData/signal_*.csv files.'
        )

    target_length = signal_length if signal_length > 0 else int(np.median(lengths))
    signals = []
    demographics = []
    targets = []
    groups = []
    sessions = []
    for waveform, basic_values, target, subject_id, session_id in pending:
        signals.append(_resize_waveform(waveform, target_length, resample=resample))
        demographics.append(basic_values)
        targets.append(target)
        groups.append(subject_id)
        sessions.append(session_id)

    return (
        np.stack(signals, axis=0),
        np.stack(demographics, axis=0),
        np.asarray(targets, dtype=np.float32),
        np.asarray(groups),
        np.asarray(sessions).astype(str),
    )


def _load_IPS_waveform_dataset(
    root: Path,
    signal_length: int = 0,
    include_basic: bool = True,
    resample: bool = False,
    ppg_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ppg_dir = root / 'ppg'
    metadata = pd.read_excel(root / 'bg_re.xlsx', sheet_name=0, header=1)
    center_seconds = resolve_ppg_center_window_seconds(ppg_seconds)
    if center_seconds:
        print(f'Center-cropping PPG to {center_seconds:g} s (IPS)')
    else:
        print('Using full on-disk PPG signal (IPS)')
    pending = []
    lengths = []
    for _, row in metadata[metadata['subject_ID'].notna()].iterrows():
        subject_id = safe_float(row['subject_ID'])
        target = safe_float(row['Diabetes'])
        if np.isnan(subject_id) or np.isnan(target):
            print('Skipping subject', row['subject_ID'], 'with invalid ID or target', target)
            continue

        subject_id_int = int(subject_id)
        signal_path = ppg_dir / f'{subject_id_int}.txt'
        if not signal_path.exists():
            print('Skipping subject', subject_id_int, 'because signal file does not exist:', signal_path.name)
            continue

        if include_basic:
            keys = basic_feature_keys('IPS')
            basic_info_dict = extract_basic_info(row, keys)
            if keys and not basic_info_dict:
                continue
            basic_values = np.asarray(
                [basic_info_dict[key] for key in keys],
                dtype=np.float32,
            )
        else:
            basic_values = np.empty(0, dtype=np.float32)

        waveform = read_signal_waveform(
            signal_path,
            trim_fractions=ppg_trim_fractions('IPS'),
            fs=ppg_sampling_rate('IPS'),
            filter_name=ppg_filter('IPS'),
            target_fs=ppg_target_fs('IPS'),
            center_window_seconds=center_seconds,
        ).astype(np.float32, copy=False)
        if waveform.size == 0:
            print('Skipping subject', subject_id_int, 'because signal is empty')
            continue

        pending.append((waveform, basic_values, target, subject_id_int))
        lengths.append(int(waveform.size))

    if not pending:
        raise RuntimeError(
            f'No usable waveform rows were found. Check bg_re.xlsx and matching {ppg_dir.name}/*.txt files.'
        )

    target_length = signal_length if signal_length > 0 else int(np.median(lengths))
    signals = []
    demographics = []
    targets = []
    groups = []
    sessions = []
    for waveform, basic_values, target, subject_id in pending:
        signals.append(_resize_waveform(waveform, target_length, resample=resample))
        demographics.append(basic_values)
        targets.append(target)
        groups.append(subject_id)
        sessions.append('0')

    return (
        np.stack(signals, axis=0),
        np.stack(demographics, axis=0),
        np.asarray(targets, dtype=np.float32),
        np.asarray(groups),
        np.asarray(sessions).astype(str),
    )


def split(dataset: pd.DataFrame, return_groups: bool = False, return_sessions: bool = False):
    feature_columns = [
        column
        for column in dataset.columns
        if column not in {'subject_ID', 'session_ID', 'window_ID', 'target'}
    ]
    X = dataset[feature_columns]
    y = dataset['target']
    groups = dataset['subject_ID'] if 'subject_ID' in dataset.columns else None
    sessions = dataset['session_ID'] if 'session_ID' in dataset.columns else None

    from utils import _log_outlier_exclusion, _target_outlier_mask

    mask, lower, upper = _target_outlier_mask(y)
    _log_outlier_exclusion(int((~mask).sum()), lower, upper)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    feature_columns = list(X.columns)
    groups_out = None if groups is None else groups.loc[mask].reset_index(drop=True).to_numpy()
    sessions_out = (
        None if sessions is None else sessions.loc[mask].reset_index(drop=True).to_numpy()
    )
    if return_groups and return_sessions:
        return X, y, feature_columns, groups_out, sessions_out
    if return_groups:
        return X, y, feature_columns, groups_out
    if return_sessions:
        return X, y, feature_columns, sessions_out
    return X, y, feature_columns


def get_X_y(dataset: pd.DataFrame):
    X, y, _ = split(dataset)
    return X, y

# def impute_scale_dataset(X: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
#     from sklearn.impute import SimpleImputer
#     imputer = SimpleImputer(strategy=strategy)
#     X_imputed = imputer.fit_transform(X)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_imputed)
#     return pd.DataFrame(X_scaled, columns=X.columns)
# def split_and_preprocess(dataset: pd.DataFrame, test_size: float = 0.2, seed: int = 42, impute_strategy: str = 'median', scale: bool = True):
#     """Split dataset into train/test and apply imputation (fit on train) and optional scaling.
#
#     Returns: X_train (DataFrame), X_test (DataFrame), y_train (Series), y_test (Series), feature_names (list)
#     """
#     X, y = get_X_y(dataset)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
#
#     imputer = SimpleImputer(strategy=impute_strategy)
#     imputer.fit(X_train)
#     X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X.columns, index=X_train.index)
#     X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)
#
#     if scale:
#         scaler = StandardScaler()
#         scaler.fit(X_train_imputed)
#         X_train_proc = pd.DataFrame(scaler.transform(X_train_imputed), columns=X.columns, index=X_train.index)
#         X_test_proc = pd.DataFrame(scaler.transform(X_test_imputed), columns=X.columns, index=X_test.index)
#     else:
#         X_train_proc = X_train_imputed
#         X_test_proc = X_test_imputed
#
#     return X_train_proc, X_test_proc, y_train.reset_index(drop=True), y_test.reset_index(drop=True), list(X.columns)


def get_kfold(n_splits: int = 5, seed: int = 42):
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MUST', help='MUST, vitaldb, or IPS')
    cli_args = parser.parse_args()

    ds = load_dataset(cli_args.dataset)
    print(ds.head())
    print('Target summary (mmol/L):')
    print(ds['target'].describe().to_string())
