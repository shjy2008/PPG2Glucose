from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from utils import (
    exclude_target_outlier_arrays,
    mg_dl_to_mmol_l,
    safe_float,
)
from config import (
    DEFAULT_WINDOW_SECONDS,
    PPG_BEST_SEGMENT_HOP_SECONDS,
    PPG_SQA_WINDOW_SECONDS,
    attach_demo_keys,
    basic_feature_keys,
    ppg_best_segment_seconds,
    ppg_feature_groups,
    ppg_filter,
    ppg_normalize,
    ppg_sampling_rate,
    ppg_csv_column,
    ppg_csv_column_fingerprint,
    ppg_sqa_method,
    ppg_sqa_threshold,
    ppg_target_fs,
    ppg_trim_fractions,
    resolve_ppg_center_window_seconds,
)
from data_basic_info import extract_basic_info
from data_ppg import (
    effective_sampling_rate,
    extract_raw_ppg,
    extract_raw_ppg_windows,
    read_signal_waveform,
    extract_ppg_bpm,
    segment_signal,
)
# from data_ppg_basic import extract_raw_ppg, read_signal_waveform, extract_ppg_bpm  # performance worse than data_ppg
from sklearn.model_selection import KFold

FEATURE_CACHE_DIR = Path(__file__).resolve().parent / 'feature_cache'
WAVEFORM_CACHE_DIR = Path(__file__).resolve().parent / 'waveform_cache'

_DEMOGRAPHIC_COLUMNS = ('sex', 'age', 'height', 'weight', 'bmi')
LABEL_UNITS = {
    'glucose': 'mmol/L',
    'target': 'mmol/L',
    'age': 'years',
    'height': 'cm',
    'weight': 'kg',
    'bmi': 'kg/m^2',
}
_LABEL_MISSING_HINT = (
    'MUST/VitalDB label CSVs have no blood pressure. '
    'IPS has a BP string in bg_re.xlsx that is not attached. '
    'To predict a field that exists in the tables but is missing here, '
    'add it to BASIC_FEATURE_KEYS in config.py.'
)


def prediction_label_key(label: str) -> str:
    key = str(label).strip().lower()
    if key in {'glucose', 'target', ''}:
        return 'glucose'
    return key


def label_unit(label: str) -> str:
    key = prediction_label_key(label)
    return LABEL_UNITS.get(key, key)


def needs_demographics(include_basic: bool, label: str) -> bool:
    """Load BASIC_FEATURE_KEYS when they are inputs or the prediction target."""
    return bool(include_basic) or prediction_label_key(label) != 'glucose'


def add_prediction_label_argument(parser, *, keep_flag: str = '--include-basic') -> None:
    parser.add_argument(
        '--label',
        default='glucose',
        help=(
            'Predict glucose (default) or an attached demographic '
            f'(age, height, weight, bmi). Non-glucose is PPG-only unless {keep_flag}. '
            'Look at MAE/R²/r; Clarke is glucose-only.'
        ),
    )


def _label_eval_kwargs(key: str) -> tuple[str, bool]:
    return label_unit(key), key == 'glucose'


def retarget_tabular_frame(
    ds: pd.DataFrame, label: str
) -> tuple[pd.DataFrame, str, str, bool]:
    """Swap y from glucose to another column already attached after load."""
    key = prediction_label_key(label)
    y_unit, plot_clarke = _label_eval_kwargs(key)
    if key == 'glucose':
        return ds, key, y_unit, plot_clarke
    if key not in ds.columns:
        attached = [c for c in _DEMOGRAPHIC_COLUMNS if c in ds.columns]
        raise SystemExit(
            f'--label {label!r} is not a column after load. '
            f'Attached demographics: {attached or "none"}. {_LABEL_MISSING_HINT}'
        )
    out = ds.copy()
    n_before = len(out)
    out['target'] = pd.to_numeric(out[key], errors='coerce')
    out = out.loc[out['target'].notna()].reset_index(drop=True)
    dropped = n_before - len(out)
    if dropped:
        print(f'Dropped {dropped} row(s) with missing {key}')
    if out.empty:
        raise SystemExit(f'No rows left after retargeting to {key}')
    print(f'Predicting {key} instead of glucose ({len(out)} rows)')
    return out, key, y_unit, plot_clarke


def retarget_waveform_arrays(
    signals: np.ndarray,
    demographics: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    sessions: np.ndarray,
    *,
    dataset: str,
    label: str,
    keep_demo: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, bool]:
    """Swap waveform y from glucose to an attached demographic column.

    Load with ``include_basic=True`` (see ``needs_demographics``). Attached
    columns follow ``attach_demo_keys``; glucose model inputs use
    ``basic_feature_keys`` when ``keep_demo`` is True.
    """
    key = prediction_label_key(label)
    y_unit, plot_clarke = _label_eval_kwargs(key)
    demo = np.asarray(demographics)
    keys = list(attach_demo_keys(dataset))
    input_keys = list(basic_feature_keys(dataset))

    def _demo_inputs(matrix: np.ndarray) -> np.ndarray:
        if not keep_demo or not input_keys:
            return np.empty((len(matrix), 0), dtype=np.float32)
        keep_idx = [keys.index(name) for name in input_keys if name in keys and name != key]
        if not keep_idx:
            return np.empty((len(matrix), 0), dtype=np.float32)
        return matrix[:, keep_idx].astype(np.float32, copy=False)

    if key == 'glucose':
        return (
            signals, _demo_inputs(demo), np.asarray(targets), groups, sessions,
            key, y_unit, plot_clarke,
        )

    if demo.ndim != 2 or demo.shape[1] != len(keys) or key not in keys:
        raise SystemExit(
            f'--label {label!r} needs that field attached at load '
            f'(attach_demo_keys={keys}). '
            f'Got demographics.shape={getattr(demo, "shape", None)}. '
            f'{_LABEL_MISSING_HINT}'
        )

    col = keys.index(key)
    y = demo[:, col].astype(np.float64, copy=False)
    mask = np.isfinite(y)
    dropped = int((~mask).sum())
    if dropped:
        print(f'Dropped {dropped} row(s) with missing {key}')
    if not mask.any():
        raise SystemExit(f'No rows left after retargeting to {key}')

    signals = np.asarray(signals)[mask]
    groups = np.asarray(groups)[mask]
    sessions = np.asarray(sessions)[mask]
    y = y[mask].astype(np.float32, copy=False)
    demo = demo[mask]
    demographics = _demo_inputs(demo)
    if keep_demo:
        print(
            f'Predicting {key} instead of glucose ({len(y)} rows); '
            f'demographics={list(basic_feature_keys(dataset))}'
        )
    else:
        print(f'Predicting {key} instead of glucose ({len(y)} rows); PPG only')
    return signals, demographics, y, groups, sessions, key, y_unit, plot_clarke


def _feature_cache_path(dataset_name: str, fingerprint: dict) -> Path:
    key = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return FEATURE_CACHE_DIR / f'{dataset_name}_{key}.pkl'


def _names_sha256(names: list[str]) -> str:
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


def _labels_rawdata_identity(label_files: list[Path]) -> dict:
    """Stable across git clone / re-download: file names, not mtimes."""
    names = sorted(path.name for path in label_files)
    return {
        'n_label_files': len(names),
        'label_names_sha256': _names_sha256(names),
    }


def _published_feature_cache_path(dataset_name: str, window_seconds: float) -> Path:
    window = float(window_seconds or 0.0)
    if window > 0:
        return FEATURE_CACHE_DIR / f'{dataset_name}_window{window:g}.pkl'
    return FEATURE_CACHE_DIR / f'{dataset_name}.pkl'


# Default ResNet/CBHFF --signal-length. Published as waveform_cache/{dataset}.npz.
_PUBLISHED_WAVEFORM_SIGNAL_LENGTH = 2048


def _published_waveform_cache_path(
    dataset_name: str,
    signal_length: int,
    window_seconds: float = 0.0,
) -> Path:
    window = float(window_seconds or 0.0)
    if window > 0:
        return WAVEFORM_CACHE_DIR / f'{dataset_name}_window{window:g}.npz'
    n = int(signal_length)
    if n <= 0:
        return WAVEFORM_CACHE_DIR / f'{dataset_name}_native.npz'
    if n == _PUBLISHED_WAVEFORM_SIGNAL_LENGTH:
        return WAVEFORM_CACHE_DIR / f'{dataset_name}.npz'
    return WAVEFORM_CACHE_DIR / f'{dataset_name}_len{n}.npz'


def _is_published_waveform_fingerprint(fingerprint: dict) -> bool:
    if (
        fingerprint.get('include_basic')
        or not fingerprint.get('resample')
        or fingerprint.get('max_samples') is not None
    ):
        return False
    window = float(fingerprint.get('window_seconds') or 0.0)
    n = int(fingerprint.get('signal_length') or 0)
    if window > 0:
        return n == _PUBLISHED_WAVEFORM_SIGNAL_LENGTH
    return n == _PUBLISHED_WAVEFORM_SIGNAL_LENGTH or n <= 0


def _waveform_cache_path(dataset_name: str, fingerprint: dict) -> Path:
    WAVEFORM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _is_published_waveform_fingerprint(fingerprint):
        return _published_waveform_cache_path(
            dataset_name,
            int(fingerprint['signal_length']),
            float(fingerprint.get('window_seconds') or 0.0),
        )
    key = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    return WAVEFORM_CACHE_DIR / f'{dataset_name}_{key}.npz'


def _waveform_npz_from_meta(meta_path: Path) -> Path:
    return meta_path.with_name(meta_path.name.removesuffix('.meta.json') + '.npz')


def _waveform_source_identity(root: Path, layout: str) -> dict:
    if layout in {'MUST', 'vitaldb'}:
        return _labels_rawdata_identity(sorted((root / 'Labels').glob('label_*.csv')))
    if layout == 'IPS':
        names = sorted(path.name for path in (root / 'ppg').glob('*.txt'))
        return {'n_ppg_files': len(names), 'ppg_names_sha256': _names_sha256(names)}
    if layout == 'Hb_PPG':
        names = sorted(path.name for path in (root / 'data_csv').glob('*.csv'))
        return {'n_ppg_files': len(names), 'ppg_names_sha256': _names_sha256(names)}
    return {}


def _waveform_center_and_identity(
    root: Path,
    layout: str,
    ppg_seconds: float | None,
) -> tuple[float | None, dict]:
    center = resolve_ppg_center_window_seconds(ppg_seconds)
    if layout in {'MUST', 'vitaldb'}:
        center = _effective_center_seconds(center, root / 'Labels')
    return center, _waveform_source_identity(root, layout)


def _waveform_cache_fingerprint(
    dataset_name: str,
    *,
    signal_length: int,
    include_basic: bool,
    resample: bool,
    max_samples: int | None,
    center_window_seconds: float | None,
    identity: dict,
    window_seconds: float = 0.0,
) -> dict:
    del include_basic  # Demographics are joined at load; they are not part of the row set.
    return {
        'kind': 'waveform',
        'dataset': dataset_name,
        'signal_length': int(signal_length),
        'window_seconds': float(window_seconds or 0.0),
        'include_basic': False,
        'basic_feature_keys': [],
        'resample': bool(resample),
        'max_samples': max_samples,
        'ppg_center_window_seconds': center_window_seconds,
        'ppg_sampling_rate': ppg_sampling_rate(dataset_name),
        'ppg_target_fs': ppg_target_fs(dataset_name),
        'ppg_trim_fractions': list(ppg_trim_fractions(dataset_name)),
        'ppg_filter': ppg_filter(dataset_name),
        'ppg_normalize': ppg_normalize(dataset_name),
        'ppg_csv_column': ppg_csv_column_fingerprint(dataset_name),
        'identity': identity,
    }


def _read_waveform_npz(
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not cache_path.is_file():
        return None
    data = np.load(cache_path, allow_pickle=False)
    if 'sessions' not in data.files:
        return None
    return (
        np.asarray(data['signals'], dtype=np.float32),
        np.asarray(data['demographics'], dtype=np.float32),
        np.asarray(data['targets'], dtype=np.float32),
        np.asarray(data['groups']),
        np.asarray(data['sessions']).astype(str),
    )


def _waveform_cache_meta_paths(
    dataset_name: str,
    signal_length: int,
    window_seconds: float = 0.0,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    candidates = [
        _published_waveform_cache_path(
            dataset_name, signal_length, window_seconds
        ).with_suffix('.meta.json')
    ]
    if WAVEFORM_CACHE_DIR.is_dir():
        candidates.extend(sorted(WAVEFORM_CACHE_DIR.glob(f'{dataset_name}_*.meta.json')))
    for path in candidates:
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _try_load_waveform_cache(
    cache_path: Path,
    refresh_cache: bool,
    fingerprint: dict | None = None,
    dataset_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if refresh_cache:
        return None

    def _from_npz(npz_path: Path, meta_path: Path | None) -> tuple | None:
        if fingerprint is not None:
            if meta_path is None or not meta_path.is_file():
                return None
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                return None
            if not _fingerprints_compatible(fingerprint, meta):
                return None
        loaded = _read_waveform_npz(npz_path)
        if loaded is not None:
            print(f'Loaded waveforms from {npz_path.name} ({len(loaded[2])} rows)')
        return loaded

    loaded = _from_npz(cache_path, cache_path.with_suffix('.meta.json'))
    if loaded is not None:
        return loaded

    if fingerprint is None or dataset_name is None or not WAVEFORM_CACHE_DIR.is_dir():
        return None
    signal_length = int(fingerprint.get('signal_length') or 0)
    window_seconds = float(fingerprint.get('window_seconds') or 0.0)
    for meta_path in _waveform_cache_meta_paths(dataset_name, signal_length, window_seconds):
        npz_path = _waveform_npz_from_meta(meta_path)
        loaded = _from_npz(npz_path, meta_path)
        if loaded is not None:
            return loaded
    return None


def _save_waveform_cache(
    cache_path: Path,
    signals: np.ndarray,
    demographics: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    sessions: np.ndarray,
    fingerprint: dict,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        signals=np.asarray(signals, dtype=np.float32),
        demographics=np.asarray(demographics, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        groups=np.asarray(groups),
        sessions=np.asarray(sessions, dtype='U32'),
    )
    cache_path.with_suffix('.meta.json').write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True, default=str),
        encoding='utf-8',
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
        'ppg_normalize': ppg_normalize(dataset_name),
        'ppg_best_segment_seconds': ppg_best_segment_seconds(dataset_name),
        'ppg_best_segment_hop_seconds': PPG_BEST_SEGMENT_HOP_SECONDS,
        'ppg_sqa_method': ppg_sqa_method(dataset_name),
        'ppg_sqa_threshold': ppg_sqa_threshold(dataset_name),
        'ppg_sqa_window_seconds': PPG_SQA_WINDOW_SECONDS,
        'ppg_center_window_seconds': center_window_seconds,
        'ppg_csv_column': ppg_csv_column_fingerprint(dataset_name),
        'identity': identity,
    }


def _ppg_only_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in _DEMOGRAPHIC_COLUMNS if c in df.columns]
    return df.drop(columns=drop) if drop else df


# Keys added after some caches were written. Missing meta values mean these defaults.
_FINGERPRINT_OPTIONAL_DEFAULTS = {
    'ppg_center_window_seconds': None,
    'ppg_csv_column': None,
    'ppg_normalize': 'none',
    'ppg_sqa_method': 'ssqi',
    'window_seconds': 0.0,
}


def _fingerprint_without_demos(meta: dict) -> dict:
    out = {k: v for k, v in meta.items() if k != 'basic_feature_keys'}
    # Machine-local / superseded keys. Old waveform caches stored an absolute root.
    out.pop('root', None)
    out.pop('ppg_seconds', None)
    out.pop('kind', None)
    for key, default in _FINGERPRINT_OPTIONAL_DEFAULTS.items():
        out.setdefault(key, default)
    if out.get('window_seconds') is not None:
        out['window_seconds'] = float(out['window_seconds'])
    return out


def _identities_compatible(want: object, have: object) -> bool:
    if not isinstance(want, dict) or not isinstance(have, dict):
        return want == have
    want_n = want.get('n_label_files', want.get('n_ppg_files'))
    have_n = have.get('n_label_files', have.get('n_ppg_files'))
    if want_n != have_n:
        return False
    want_hash = want.get('label_names_sha256') or want.get('ppg_names_sha256')
    have_hash = have.get('label_names_sha256') or have.get('ppg_names_sha256')
    if want_hash and have_hash:
        return want_hash == have_hash
    return True


def _fingerprints_compatible(want: dict, have: dict) -> bool:
    want_fp = _fingerprint_without_demos(want)
    have_fp = _fingerprint_without_demos(have)
    want_id = want_fp.pop('identity', None)
    have_id = have_fp.pop('identity', None)
    return want_fp == have_fp and _identities_compatible(want_id, have_id)


def _feature_cache_meta_paths(dataset_name: str, window_seconds: float) -> list[Path]:
    paths = []
    published = _published_feature_cache_path(dataset_name, window_seconds).with_suffix('.meta.json')
    if published.is_file():
        paths.append(published)
    paths.extend(sorted(FEATURE_CACHE_DIR.glob(f'{dataset_name}_*.meta.json')))
    return paths


def _manifest_window_seconds(labels_dir: Path) -> float | None:
    manifest = labels_dir.parent / 'manifest.json'
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get('window_seconds')
    if value is None:
        return None
    return float(value)


def _effective_center_seconds(
    center_seconds: float | None,
    labels_dir: Path,
) -> float | None:
    """Treat a crop as a no-op when on-disk windows are already that short."""
    if center_seconds is None:
        return None
    on_disk = _manifest_window_seconds(labels_dir)
    if on_disk is not None and on_disk <= float(center_seconds) + 1e-6:
        return None
    return center_seconds


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

    # Reuse older caches whose meta still listed basic_feature_keys or mtimes.
    if fingerprint is None or dataset_name is None or not FEATURE_CACHE_DIR.is_dir():
        return None
    window_seconds = float((fingerprint or {}).get('window_seconds') or 0.0)
    for meta_path in _feature_cache_meta_paths(dataset_name, window_seconds):
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if not _fingerprints_compatible(fingerprint, meta):
            continue
        pkl_path = meta_path.with_name(meta_path.name.removesuffix('.meta.json') + '.pkl')
        if not pkl_path.is_file():
            continue
        df = pd.read_pickle(pkl_path)
        print(f'Loaded PPG features from {pkl_path.name} ({len(df)} rows)')
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
    keys = attach_demo_keys(dataset_name)
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
    keys = attach_demo_keys('IPS')
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


def _join_waveform_demographics(
    signals: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    sessions: np.ndarray,
    *,
    dataset_name: str,
    root: Path,
    layout: str,
    include_basic: bool,
    labels_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Attach demographics after cache load (same join as tabular PPG features)."""
    n = len(targets)
    empty = np.empty((n, 0), dtype=np.float32)
    if not include_basic:
        return signals, empty, targets, groups, sessions

    keys = list(attach_demo_keys(dataset_name))
    if not keys:
        return signals, empty, targets, groups, sessions

    frame = pd.DataFrame(
        {
            '_row': np.arange(n),
            'subject_ID': np.asarray(groups),
            'session_ID': np.asarray(sessions).astype(str),
            'target': np.asarray(targets),
        }
    )
    frame['subject_ID'] = pd.to_numeric(frame['subject_ID'], errors='coerce')
    if layout == 'Hb_PPG':
        attached = _attach_Hb_PPG_basic(frame, root)
    elif layout == 'IPS':
        attached = _attach_IPS_basic(frame, root)
    else:
        attach_labels = Path(labels_dir) if labels_dir is not None else root / 'Labels'
        attached = _attach_labels_rawdata_basic(frame, attach_labels, dataset_name)

    idx = attached['_row'].to_numpy(dtype=int)
    order = np.argsort(idx)
    idx = idx[order]
    demographics = attached.loc[:, keys].to_numpy(dtype=np.float32)[order]
    return (
        np.asarray(signals)[idx],
        demographics,
        np.asarray(targets)[idx],
        np.asarray(groups)[idx],
        np.asarray(sessions)[idx],
    )


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
    normalize = ppg_normalize(dataset_name)
    target_fs = ppg_target_fs(dataset_name)
    best_segment_seconds = ppg_best_segment_seconds(dataset_name)
    feature_groups = ppg_feature_groups(dataset_name)
    sqa_threshold = ppg_sqa_threshold(dataset_name)
    sqa_method = ppg_sqa_method(dataset_name)
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
            sqa_method=sqa_method,
            center_window_seconds=center_window_seconds,
            csv_column=ppg_csv_column(dataset_name),
            normalize=normalize,
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
        sqa_method=sqa_method,
        center_window_seconds=center_window_seconds,
        csv_column=ppg_csv_column(dataset_name),
        normalize=normalize,
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
    if (root / 'subject information.xlsx').is_file() and (root / 'data_csv').is_dir():
        return 'Hb_PPG'
    raise RuntimeError(
        f'Unrecognized dataset layout under {root}. '
        'Expected vitaldb/MUST (Labels/ + RawData/), IPS (bg_re.xlsx + ppg/), '
        'or Hb_PPG (subject information.xlsx + data_csv/).'
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
    if layout == 'Hb_PPG':
        return _load_Hb_PPG_dataset(
            root,
            window_seconds=window_seconds,
            max_samples=max_samples,
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

    center_seconds = _effective_center_seconds(
        resolve_ppg_center_window_seconds(ppg_seconds),
        labels_dir,
    )
    identity = _labels_rawdata_identity(label_files)
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


HB_PPG_INFO_FILE = 'subject information.xlsx'
HB_PPG_CSV_DIR = 'data_csv'
HB_PPG_GLUCOSE_COL = 'Blood glucose (mmol/L)'


def _hb_ppg_info_path(root: Path) -> Path:
    return root / HB_PPG_INFO_FILE


def _hb_ppg_csv_dir(root: Path) -> Path:
    return root / HB_PPG_CSV_DIR


def _read_hb_ppg_table(root: Path) -> pd.DataFrame:
    path = _hb_ppg_info_path(root)
    if not path.is_file():
        raise RuntimeError(f'Missing {path}')
    return pd.read_excel(path)


def _hb_ppg_signal_path(root: Path, subject_id: int) -> Path:
    return _hb_ppg_csv_dir(root) / f'{subject_id}.csv'


def _iter_hb_ppg_rows(root: Path, max_samples: int | None = None):
    metadata = _read_hb_ppg_table(root)
    n_glucose_missing = 0
    rows = []
    for _, row in metadata.iterrows():
        subject_id = safe_float(row.get('ID'))
        target = safe_float(row.get(HB_PPG_GLUCOSE_COL))
        if np.isnan(subject_id):
            continue
        if np.isnan(target):
            n_glucose_missing += 1
            continue
        rows.append((int(subject_id), float(target), row))
    rows.sort(key=lambda item: item[0])
    if n_glucose_missing:
        print(f'Skipped {n_glucose_missing} Hb_PPG subject(s) with missing glucose')
    if max_samples is not None:
        rows = rows[:max_samples]
        print(f'Limited to {len(rows)} sample(s)')
    return rows


def _attach_Hb_PPG_basic(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    out = _ppg_only_frame(df)
    keys = attach_demo_keys('Hb_PPG')
    if not keys:
        return out.reset_index(drop=True)

    demo_records = []
    for _, row in _read_hb_ppg_table(root).iterrows():
        subject_id = safe_float(row.get('ID'))
        if np.isnan(subject_id):
            continue
        info = extract_basic_info(row, keys)
        if info:
            demo_records.append({'subject_ID': int(subject_id), **info})
    if not demo_records:
        raise RuntimeError(f'No usable demographics {list(keys)} in {_hb_ppg_info_path(root)}')
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


def _load_Hb_PPG_dataset(
    root: Path,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    max_samples: int | None = None,
    refresh_cache: bool = False,
    ppg_seconds: float | None = None,
) -> pd.DataFrame:
    csv_dir = _hb_ppg_csv_dir(root)
    csv_files = list(csv_dir.glob('*.csv')) if csv_dir.is_dir() else []
    center_seconds = resolve_ppg_center_window_seconds(ppg_seconds)
    identity = {
        'n_ppg_files': len(csv_files),
        'max_ppg_mtime': round(max((p.stat().st_mtime for p in csv_files), default=0.0), 3),
        'subject_info_mtime': round(_hb_ppg_info_path(root).stat().st_mtime, 3),
    }
    fingerprint = _ppg_cache_fingerprint(
        'Hb_PPG',
        window_seconds,
        max_samples,
        identity,
        center_window_seconds=center_seconds,
    )
    cache_path = _feature_cache_path('Hb_PPG', fingerprint)
    print(f'Hb_PPG wavelengths: {ppg_csv_column("Hb_PPG")}')
    cached = _try_load_feature_cache(
        cache_path, refresh_cache, fingerprint=fingerprint, dataset_name='Hb_PPG'
    )
    if cached is not None:
        return _attach_Hb_PPG_basic(cached, root)

    rows = _iter_hb_ppg_rows(root, max_samples=max_samples)
    n_files = len(rows)
    progress_every = max(1, n_files // 20)
    print(f'Loading {n_files} sample(s) from {csv_dir} (PPG feature extraction)...')
    if center_seconds:
        print(f'Center-cropping PPG to {center_seconds:g} s (Hb_PPG)')
    else:
        print('Using full on-disk PPG signal (Hb_PPG)')

    records = []
    for i, (subject_id, target, _row) in enumerate(rows, start=1):
        signal_path = _hb_ppg_signal_path(root, subject_id)
        if not signal_path.exists():
            print('Skipping subject', subject_id, 'because signal file does not exist:', signal_path.name)
            continue
        feature_dicts = _ppg_feature_dicts(
            signal_path,
            fs=ppg_sampling_rate('Hb_PPG'),
            window_seconds=window_seconds,
            dataset_name='Hb_PPG',
            center_window_seconds=center_seconds,
        )
        if not feature_dicts:
            print('Skipping subject', subject_id, 'because signal is empty or failed SQA')
            continue
        for window_id, raw_ppg_dict in enumerate(feature_dicts):
            record = {
                'subject_ID': subject_id,
                'target': target,
            }
            if window_seconds is not None and window_seconds > 0:
                record['window_ID'] = window_id
            record.update(raw_ppg_dict)
            records.append(record)
        if i == 1 or i == n_files or i % progress_every == 0:
            print(f'  loaded {i}/{n_files} files ({len(records)} usable)', flush=True)

    if not records:
        raise RuntimeError(
            f'No usable rows were found. Check {HB_PPG_INFO_FILE} and matching {csv_dir.name}/*.csv files.'
        )
    dataset = pd.DataFrame(records)
    print(f'Finished PPG extraction: {len(dataset)} usable sample(s)', flush=True)
    _save_feature_cache(cache_path, dataset, fingerprint)
    return _attach_Hb_PPG_basic(dataset, root)


def _resize_waveform(waveform: np.ndarray, target_length: int, resample: bool) -> np.ndarray:
    values = np.asarray(waveform)
    if values.ndim == 2:
        return np.stack(
            [
                _resize_waveform(channel, target_length, resample=resample)
                for channel in values
            ],
            axis=0,
        )
    if values.size >= target_length:
        if resample:
            indices = np.linspace(0, values.size - 1, target_length).astype(int)
            return values[indices]
        return values[:target_length]
    return np.pad(values, (0, target_length - values.size), mode='constant')


def expand_waveform_windows(
    signals: np.ndarray,
    demographics: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    dataset: str,
    window_seconds: float,
    signal_length: int,
    resample: bool = True,
    sessions: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Chop each waveform into windows that share the same glucose label."""
    fs = effective_sampling_rate(ppg_sampling_rate(dataset), ppg_target_fs(dataset))
    windowed_signals = []
    windowed_demographics = []
    windowed_targets = []
    windowed_groups = []
    windowed_sessions = [] if sessions is not None else None
    for index in range(len(targets)):
        windows = segment_signal(signals[index], fs=fs, window_seconds=window_seconds)
        if not windows:
            continue
        demo = demographics[index]
        target = targets[index]
        group = groups[index]
        session = None if sessions is None else sessions[index]
        for window in windows:
            resized = _resize_waveform(
                np.asarray(window, dtype=np.float32),
                signal_length,
                resample=resample,
            )
            windowed_signals.append(resized)
            windowed_demographics.append(demo)
            windowed_targets.append(target)
            windowed_groups.append(group)
            if windowed_sessions is not None:
                windowed_sessions.append(session)

    if not windowed_signals:
        raise RuntimeError(
            f'No windows produced with window_seconds={window_seconds} at fs={fs}'
        )
    out = (
        np.stack(windowed_signals, axis=0),
        np.stack(windowed_demographics, axis=0),
        np.asarray(windowed_targets, dtype=np.float32),
        np.asarray(windowed_groups),
    )
    if sessions is None:
        return out
    return (*out, np.asarray(windowed_sessions).astype(str))


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
    window_seconds: float = 0.0,
) -> tuple[np.ndarray, ...]:
    """Load per-session PPG waveforms and glucose targets (mmol/L).

    Returns:
        signals: (n_samples, signal_length) or (n_samples, n_channels, signal_length)
            when ``PPG_CSV_COLUMN`` lists multiple wavelengths
        demographics: (n_samples, n_basic) or (n_samples, 0) when include_basic is False.
            Demographics are joined after the waveform cache load, so changing
            BASIC_FEATURE_KEYS does not require rebuilding waveforms.
        targets: (n_samples,)
        groups: subject IDs, returned only when return_groups is True
        sessions: recording IDs, returned only when return_sessions is True

    Caches resized waveforms under ``waveform_cache/`` using stable names like
    ``feature_cache/`` (``vitaldb.npz``, ``vitaldb_window10.npz``).
    """
    window_seconds = float(window_seconds or 0.0)
    if window_seconds > 0:
        out_length = int(signal_length) if int(signal_length) > 0 else 1024
        load_length = 0
    else:
        out_length = int(signal_length)
        load_length = int(signal_length)

    def _cached_or_build(
        config_name: str,
        root: Path,
        layout: str,
        build,
        attach_labels_dir: Path | None = None,
    ):
        center, identity = _waveform_center_and_identity(root, layout, ppg_seconds)
        fingerprint = _waveform_cache_fingerprint(
            config_name,
            signal_length=out_length,
            include_basic=False,
            resample=resample,
            max_samples=max_samples,
            center_window_seconds=center,
            identity=identity,
            window_seconds=window_seconds,
        )
        cache_path = _waveform_cache_path(config_name, fingerprint)
        cached = _try_load_waveform_cache(
            cache_path,
            refresh_cache,
            fingerprint=fingerprint,
            dataset_name=config_name,
        )
        if cached is not None:
            loaded = cached
        else:
            loaded = build(center)
            if window_seconds > 0:
                n_labs = len(loaded[2])
                loaded = expand_waveform_windows(
                    loaded[0],
                    loaded[1],
                    loaded[2],
                    loaded[3],
                    dataset=config_name,
                    window_seconds=window_seconds,
                    signal_length=out_length,
                    resample=resample,
                    sessions=loaded[4],
                )
                print(
                    f'Windowed {n_labs} lab draws → {len(loaded[2])} segments '
                    f'({window_seconds:g} s, shared glucose labels)'
                )
            _save_waveform_cache(cache_path, *loaded, fingerprint)
        signals, _ignored_demo, targets, groups, sessions = loaded
        return _join_waveform_demographics(
            signals,
            targets,
            groups,
            sessions,
            dataset_name=config_name,
            root=root,
            layout=layout,
            include_basic=include_basic,
            labels_dir=attach_labels_dir,
        )

    if labels_dir is not None or raw_data_dir is not None:
        if labels_dir is None or raw_data_dir is None:
            raise ValueError('labels_dir and raw_data_dir must both be provided')
        root = Path(labels_dir).resolve().parent
        layout = dataset_layout(root)
        config_name = layout if layout in {'MUST', 'vitaldb'} else 'MUST'
        signals, demographics, targets, groups, sessions = _cached_or_build(
            config_name,
            root,
            layout,
            lambda center: _load_labels_rawdata_waveform_dataset(
                labels_dir,
                raw_data_dir,
                config_name,
                signal_length=load_length,
                include_basic=False,
                resample=resample,
                max_samples=max_samples,
                center_window_seconds=center,
            ),
            attach_labels_dir=Path(labels_dir),
        )
    else:
        root = resolve_dataset_root(dataset)
        layout = dataset_layout(root)
        config_name = _config_dataset_name(dataset, layout)
        if layout in {'MUST', 'vitaldb'}:
            signals, demographics, targets, groups, sessions = _cached_or_build(
                config_name,
                root,
                layout,
                lambda center: _load_labels_rawdata_waveform_dataset(
                    root / 'Labels',
                    root / 'RawData',
                    config_name,
                    signal_length=load_length,
                    include_basic=False,
                    resample=resample,
                    max_samples=max_samples,
                    center_window_seconds=center,
                ),
                attach_labels_dir=root / 'Labels',
            )
        elif layout == 'Hb_PPG':
            print(f'Hb_PPG wavelengths: {ppg_csv_column("Hb_PPG")}')
            signals, demographics, targets, groups, sessions = _cached_or_build(
                config_name,
                root,
                layout,
                lambda center: _load_Hb_PPG_waveform_dataset(
                    root,
                    signal_length=load_length,
                    include_basic=False,
                    resample=resample,
                    max_samples=max_samples,
                    ppg_seconds=ppg_seconds,
                    center_window_seconds=center,
                ),
            )
        else:
            signals, demographics, targets, groups, sessions = _cached_or_build(
                config_name,
                root,
                layout,
                lambda center: _load_IPS_waveform_dataset(
                    root,
                    signal_length=load_length,
                    include_basic=False,
                    resample=resample,
                    ppg_seconds=ppg_seconds,
                    center_window_seconds=center,
                ),
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
    center_window_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_files = sorted(labels_dir.glob('label_*.csv'))
    if not label_files:
        raise RuntimeError(f'No label files found in {labels_dir}')
    if max_samples is not None:
        label_files = label_files[:max_samples]
        print(f'Limited to {len(label_files)} sample(s)')

    if center_window_seconds is None:
        center_seconds = _effective_center_seconds(
            resolve_ppg_center_window_seconds(ppg_seconds),
            labels_dir,
        )
    else:
        center_seconds = center_window_seconds
    n_files = len(label_files)
    progress_every = max(1, n_files // 20)
    print(f'Loading {n_files} waveform(s) from {labels_dir}...')
    if center_seconds:
        print(
            f'Center-cropping PPG to {center_seconds:g} s '
            f'({dataset_name}, fs={ppg_sampling_rate(dataset_name):g} Hz)'
        )
    else:
        print(f'Using full on-disk PPG signal ({dataset_name})')

    pending = []
    lengths = []
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

        if include_basic:
            keys = attach_demo_keys(dataset_name)
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
            csv_column=ppg_csv_column(dataset_name),
            normalize=ppg_normalize(dataset_name),
        ).astype(np.float32, copy=False)
        if waveform.size == 0:
            print('Skipping', label_path.name, 'because signal is empty')
            continue

        pending.append((waveform, basic_values, target, int(subject_id), str(session_key)))
        lengths.append(int(waveform.shape[-1]))
        if i == 1 or i == n_files or i % progress_every == 0:
            print(f'  loaded {i}/{n_files} files ({len(pending)} usable)', flush=True)

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
    center_window_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ppg_dir = root / 'ppg'
    metadata = pd.read_excel(root / 'bg_re.xlsx', sheet_name=0, header=1)
    center_seconds = (
        center_window_seconds
        if center_window_seconds is not None
        else resolve_ppg_center_window_seconds(ppg_seconds)
    )
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
            keys = attach_demo_keys('IPS')
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
            csv_column=ppg_csv_column('IPS'),
            normalize=ppg_normalize('IPS'),
        ).astype(np.float32, copy=False)
        if waveform.size == 0:
            print('Skipping subject', subject_id_int, 'because signal is empty')
            continue

        pending.append((waveform, basic_values, target, subject_id_int))
        lengths.append(int(waveform.shape[-1]))

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


def _load_Hb_PPG_waveform_dataset(
    root: Path,
    signal_length: int = 0,
    include_basic: bool = True,
    resample: bool = False,
    max_samples: int | None = None,
    ppg_seconds: float | None = None,
    center_window_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center_seconds = (
        center_window_seconds
        if center_window_seconds is not None
        else resolve_ppg_center_window_seconds(ppg_seconds)
    )
    if center_seconds:
        print(f'Center-cropping PPG to {center_seconds:g} s (Hb_PPG)')
    else:
        print('Using full on-disk PPG signal (Hb_PPG)')

    pending = []
    lengths = []
    for subject_id, target, row in _iter_hb_ppg_rows(root, max_samples=max_samples):
        signal_path = _hb_ppg_signal_path(root, subject_id)
        if not signal_path.exists():
            print('Skipping subject', subject_id, 'because signal file does not exist:', signal_path.name)
            continue

        if include_basic:
            keys = attach_demo_keys('Hb_PPG')
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
            trim_fractions=ppg_trim_fractions('Hb_PPG'),
            fs=ppg_sampling_rate('Hb_PPG'),
            filter_name=ppg_filter('Hb_PPG'),
            target_fs=ppg_target_fs('Hb_PPG'),
            center_window_seconds=center_seconds,
            csv_column=ppg_csv_column('Hb_PPG'),
            normalize=ppg_normalize('Hb_PPG'),
        ).astype(np.float32, copy=False)
        if waveform.size == 0:
            print('Skipping subject', subject_id, 'because signal is empty')
            continue

        pending.append((waveform, basic_values, target, subject_id))
        lengths.append(int(waveform.shape[-1]))

    if not pending:
        raise RuntimeError(
            f'No usable waveform rows were found. Check {HB_PPG_INFO_FILE} and matching '
            f'{HB_PPG_CSV_DIR}/*.csv files.'
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
    parser.add_argument('--dataset', default='MUST', help='MUST, vitaldb, IPS, or Hb_PPG')
    cli_args = parser.parse_args()

    ds = load_dataset(cli_args.dataset)
    print(ds.head())
    print('Target summary (mmol/L):')
    print(ds['target'].describe().to_string())
