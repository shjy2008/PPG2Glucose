from __future__ import annotations

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
    basic_feature_keys,
    ppg_sampling_rate,
    ppg_trim_fractions,
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


def _ppg_feature_dicts(
    signal_path: Path,
    fs: float,
    window_seconds: float,
    dataset_name: str,
) -> list[dict]:
    """Full-signal features when window_seconds<=0; else one dict per window."""
    trim_fractions = ppg_trim_fractions(dataset_name)
    if window_seconds is None or window_seconds <= 0:
        return [extract_raw_ppg(signal_path, fs=fs, trim_fractions=trim_fractions)]
    return extract_raw_ppg_windows(
        signal_path,
        fs=fs,
        window_seconds=window_seconds,
        trim_fractions=trim_fractions,
    )

def resolve_dataset_root(dataset: str | Path | None = None) -> Path:
    if dataset is None:
        dataset = 'southwest'
    if isinstance(dataset, Path):
        root = dataset
    elif dataset == 'MUST':
        root = Path('dataset/MUST')
    elif dataset == 'vitaldb':
        root = Path('dataset/vitaldb')
    elif dataset == 'southwest':
        root = Path('dataset/southwest')
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
        return 'southwest'
    raise RuntimeError(
        f'Unrecognized dataset layout under {root}. '
        'Expected vitaldb/MUST (Labels/ + RawData/) or southwest (bg_re.xlsx + ppg/).'
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
        )

    root = resolve_dataset_root(dataset)
    layout = dataset_layout(root)
    if layout in {'MUST', 'vitaldb'}:
        return _load_labels_rawdata_dataset(
            root / 'Labels',
            root / 'RawData',
            layout,
            max_samples=max_samples,
            window_seconds=window_seconds,
        )
    return _load_southwest_dataset(root, window_seconds=window_seconds)


def _load_labels_rawdata_dataset(
    labels_dir: Path,
    raw_data_dir: Path,
    dataset_name: str,
    max_samples: int | None = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> pd.DataFrame:
    label_files = sorted(labels_dir.glob('label_*.csv'))
    if not label_files:
        raise RuntimeError(f'No label files found in {labels_dir}')
    if max_samples is not None:
        label_files = label_files[:max_samples]
        print(f'Limited to {len(label_files)} sample(s)')

    n_files = len(label_files)
    progress_every = max(1, n_files // 20)  # ~5% steps
    print(f'Loading {n_files} sample(s) from {labels_dir} (PPG feature extraction)...')

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

        keys = basic_feature_keys(dataset_name)
        basic_info_dict = extract_basic_info(row, keys)
        if keys and not basic_info_dict:
            continue

        signal_path = _signal_path(raw_data_dir, subject_key, session_key)
        if not signal_path.exists():
            print('Skipping', label_path.name, 'because signal file does not exist:', signal_path.name)
            continue
        feature_dicts = _ppg_feature_dicts(
            signal_path,
            fs=ppg_sampling_rate(dataset_name),
            window_seconds=window_seconds,
            dataset_name=dataset_name,
        )
        if not feature_dicts:
            print('Skipping', label_path.name, 'because signal is empty')
            continue

        for window_id, raw_ppg_dict in enumerate(feature_dicts):
            record = {
                'subject_ID': subject_id_int,
                'session_ID': session_key,
                'target': target,
            }
            if window_seconds is not None and window_seconds > 0:
                record['window_ID'] = window_id
            record.update(basic_info_dict)
            record.update(raw_ppg_dict)
            records.append(record)

        if i == 1 or i == n_files or i % progress_every == 0:
            print(f'  loaded {i}/{n_files} files ({len(records)} usable)', flush=True)

    dataset = pd.DataFrame(records)
    if dataset.empty:
        raise RuntimeError(
            'No usable rows were found. Check Labels/*.csv and matching RawData/signal_*.csv files.'
        )

    print(f'Finished loading: {len(dataset)} usable sample(s)', flush=True)
    return dataset


def _load_southwest_dataset(
    root: Path,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> pd.DataFrame:
    ppg_dir = root / 'ppg'
    metadata = pd.read_excel(root / 'bg_re.xlsx', sheet_name=0, header=1)
    records = []
    for _, row in metadata[metadata['subject_ID'].notna()].iterrows():
        subject_id = safe_float(row['subject_ID'])
        target = safe_float(row['Diabetes'])
        if np.isnan(subject_id) or np.isnan(target):
            print('Skipping subject', row['subject_ID'], 'with invalid ID or target', target)
            continue

        keys = basic_feature_keys('southwest')
        basic_info_dict = extract_basic_info(row, keys)
        if keys and not basic_info_dict:
            continue

        subject_id_int = int(subject_id)
        signal_path = ppg_dir / f'{subject_id_int}.txt'
        if not signal_path.exists():
            print('Skipping subject', subject_id_int, 'because signal file does not exist:', signal_path.name)
            continue

        feature_dicts = _ppg_feature_dicts(
            signal_path,
            fs=ppg_sampling_rate('southwest'),
            window_seconds=window_seconds,
            dataset_name='southwest',
        )
        if not feature_dicts:
            print('Skipping subject', subject_id_int, 'because signal is empty')
            continue

        for window_id, raw_ppg_dict in enumerate(feature_dicts):
            record = {
                'subject_ID': subject_id_int,
                'target': target,
            }
            if window_seconds is not None and window_seconds > 0:
                record['window_ID'] = window_id
            record.update(basic_info_dict)
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
    return pd.DataFrame(records)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-session PPG waveforms and glucose targets (mmol/L).

    Returns:
        signals: (n_samples, signal_length)
        demographics: (n_samples, 4) or (n_samples, 0) when include_basic is False
        targets: (n_samples,)
    """
    if labels_dir is not None or raw_data_dir is not None:
        if labels_dir is None or raw_data_dir is None:
            raise ValueError('labels_dir and raw_data_dir must both be provided')
        signals, demographics, targets = _load_labels_rawdata_waveform_dataset(
            labels_dir,
            raw_data_dir,
            'MUST',
            signal_length=signal_length,
            include_basic=include_basic,
            resample=resample,
            max_samples=max_samples,
        )
    else:
        root = resolve_dataset_root(dataset)
        layout = dataset_layout(root)
        if layout in {'MUST', 'vitaldb'}:
            signals, demographics, targets = _load_labels_rawdata_waveform_dataset(
                root / 'Labels',
                root / 'RawData',
                layout,
                signal_length=signal_length,
                include_basic=include_basic,
                resample=resample,
                max_samples=max_samples,
            )
        else:
            signals, demographics, targets = _load_southwest_waveform_dataset(
                root,
                signal_length=signal_length,
                include_basic=include_basic,
                resample=resample,
            )

    targets, signals, demographics = exclude_target_outlier_arrays(
        targets,
        signals,
        demographics,
    )
    return signals, demographics, targets


def _load_labels_rawdata_waveform_dataset(
    labels_dir: Path,
    raw_data_dir: Path,
    dataset_name: str,
    signal_length: int = 0,
    include_basic: bool = True,
    resample: bool = False,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_files = sorted(labels_dir.glob('label_*.csv'))
    if not label_files:
        raise RuntimeError(f'No label files found in {labels_dir}')
    if max_samples is not None:
        label_files = label_files[:max_samples]
        print(f'Limited to {len(label_files)} sample(s)')

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
            signal_path, ppg_trim_fractions(dataset_name)
        ).astype(np.float32, copy=False)
        if waveform.size == 0:
            print('Skipping', label_path.name, 'because signal is empty')
            continue

        pending.append((waveform, basic_values, target))
        lengths.append(int(waveform.size))

    if not pending:
        raise RuntimeError(
            'No usable waveform rows were found. Check Labels/*.csv and matching RawData/signal_*.csv files.'
        )

    target_length = signal_length if signal_length > 0 else int(np.median(lengths))
    signals = []
    demographics = []
    targets = []
    for waveform, basic_values, target in pending:
        signals.append(_resize_waveform(waveform, target_length, resample=resample))
        demographics.append(basic_values)
        targets.append(target)

    return (
        np.stack(signals, axis=0),
        np.stack(demographics, axis=0),
        np.asarray(targets, dtype=np.float32),
    )


def _load_southwest_waveform_dataset(
    root: Path,
    signal_length: int = 0,
    include_basic: bool = True,
    resample: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ppg_dir = root / 'ppg'
    metadata = pd.read_excel(root / 'bg_re.xlsx', sheet_name=0, header=1)
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
            keys = basic_feature_keys('southwest')
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
            signal_path, ppg_trim_fractions('southwest')
        ).astype(np.float32, copy=False)
        if waveform.size == 0:
            print('Skipping subject', subject_id_int, 'because signal is empty')
            continue

        pending.append((waveform, basic_values, target))
        lengths.append(int(waveform.size))

    if not pending:
        raise RuntimeError(
            f'No usable waveform rows were found. Check bg_re.xlsx and matching {ppg_dir.name}/*.txt files.'
        )

    target_length = signal_length if signal_length > 0 else int(np.median(lengths))
    signals = []
    demographics = []
    targets = []
    for waveform, basic_values, target in pending:
        signals.append(_resize_waveform(waveform, target_length, resample=resample))
        demographics.append(basic_values)
        targets.append(target)

    return (
        np.stack(signals, axis=0),
        np.stack(demographics, axis=0),
        np.asarray(targets, dtype=np.float32),
    )


def split(dataset: pd.DataFrame, return_groups: bool = False):
    feature_columns = [
        column
        for column in dataset.columns
        if column not in {'subject_ID', 'session_ID', 'window_ID', 'target'}
    ]
    X = dataset[feature_columns]
    y = dataset['target']
    groups = dataset['subject_ID'] if 'subject_ID' in dataset.columns else None

    from utils import _log_outlier_exclusion, _target_outlier_mask

    mask, lower, upper = _target_outlier_mask(y)
    _log_outlier_exclusion(int((~mask).sum()), lower, upper)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    feature_columns = list(X.columns)
    if return_groups:
        if groups is None:
            return X, y, feature_columns, None
        groups = groups.loc[mask].reset_index(drop=True).to_numpy()
        return X, y, feature_columns, groups
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
    parser.add_argument('--dataset', default='MUST', help='MUST, vitaldb, or southwest')
    cli_args = parser.parse_args()

    ds = load_dataset(cli_args.dataset)
    print(ds.head())
    print('Target summary (mmol/L):')
    print(ds['target'].describe().to_string())
