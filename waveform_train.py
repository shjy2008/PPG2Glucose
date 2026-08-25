"""Shared helpers for 1D PPG waveform deep-learning trainers."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

import data_loader
import evaluation
from config import (
    DEFAULT_VITALDB_PPG_SECONDS,
    basic_feature_keys,
    ppg_sampling_rate,
    ppg_target_fs,
    resolve_config_dataset,
    torch_float_tensor,
)
from data_ppg import effective_sampling_rate, segment_signal


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_signals_per_sample(signals: np.ndarray) -> np.ndarray:
    mean = signals.mean(axis=1, keepdims=True)
    std = signals.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return ((signals - mean) / std).astype(np.float32, copy=False)


def normalize_targets(y_train: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(y_train.mean())
    std = float(y_train.std())
    if std == 0:
        std = 1.0
    normalized = ((y_train - mean) / std).astype(np.float32, copy=False)
    return normalized, mean, std


def standardize_train_test(
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (
        ((train - mean) / std).astype(np.float32, copy=False),
        ((validation - mean) / std).astype(np.float32, copy=False),
    )


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chop each waveform into windows that share the same glucose label."""
    fs = effective_sampling_rate(ppg_sampling_rate(dataset), ppg_target_fs(dataset))
    windowed_signals = []
    windowed_demographics = []
    windowed_targets = []
    windowed_groups = []
    for index in range(len(targets)):
        windows = segment_signal(signals[index], fs=fs, window_seconds=window_seconds)
        if not windows:
            continue
        demo = demographics[index]
        target = targets[index]
        group = groups[index]
        for window in windows:
            resized = data_loader._resize_waveform(
                np.asarray(window, dtype=np.float32),
                signal_length,
                resample=resample,
            )
            windowed_signals.append(resized)
            windowed_demographics.append(demo)
            windowed_targets.append(target)
            windowed_groups.append(group)

    if not windowed_signals:
        raise RuntimeError(
            f'No windows produced with window_seconds={window_seconds} at fs={fs}'
        )
    return (
        np.stack(windowed_signals, axis=0),
        np.stack(windowed_demographics, axis=0),
        np.asarray(windowed_targets, dtype=np.float32),
        np.asarray(windowed_groups),
    )


def split_train_val(
    train_index: np.ndarray,
    groups: Optional[np.ndarray],
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve an inner validation set; subject-wise when groups are given."""
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        inner_train, inner_val = next(
            splitter.split(train_index, groups=groups[train_index])
        )
        return train_index[inner_train], train_index[inner_val]

    shuffled = train_index.copy()
    np.random.RandomState(seed).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    return shuffled[n_val:], shuffled[:n_val]


def make_signal_loader(
    signals: np.ndarray,
    targets: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    *,
    demographics: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    drop_last: bool = False,
) -> DataLoader:
    tensors = [torch_float_tensor(signals[:, None, :])]
    if demographics is not None and demographics.size:
        tensors.append(torch_float_tensor(demographics))
    if targets is not None:
        tensors.append(torch_float_tensor(targets))
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=drop_last and shuffle and len(signals) >= batch_size,
    )


def default_ppg_seconds(dataset: str) -> float | None:
    if resolve_config_dataset(dataset) == 'vitaldb':
        return DEFAULT_VITALDB_PPG_SECONDS
    return None


def load_waveform_split(
    dataset: str,
    *,
    signal_length: int,
    include_basic: bool,
    window_seconds: float,
    max_samples: int | None = None,
    resample: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    load_length = 0 if window_seconds > 0 else signal_length
    signals, demographics, targets, groups = data_loader.load_waveform_dataset(
        dataset=dataset,
        signal_length=load_length,
        include_basic=include_basic,
        resample=resample,
        max_samples=max_samples,
        return_groups=True,
        ppg_seconds=default_ppg_seconds(dataset),
    )
    if window_seconds > 0:
        n_labs = len(targets)
        signals, demographics, targets, groups = expand_waveform_windows(
            signals,
            demographics,
            targets,
            groups,
            dataset=dataset,
            window_seconds=window_seconds,
            signal_length=signal_length if signal_length > 0 else 1024,
            resample=resample,
        )
        print(
            f'{dataset}: windowed {n_labs} lab draws → {len(targets)} segments '
            f'({window_seconds:g} s)'
        )
    return signals, demographics, targets, groups


def _resize_signals(signals: np.ndarray, length: int) -> np.ndarray:
    if signals.shape[1] == length:
        return signals
    return np.stack(
        [data_loader._resize_waveform(row, length, resample=True) for row in signals],
        axis=0,
    )


def align_ood_demographics(
    train_demo: np.ndarray,
    test_demo: np.ndarray,
    train_dataset: str,
    test_dataset: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    empty_train = np.empty((len(train_demo), 0), dtype=np.float32)
    empty_test = np.empty((len(test_demo), 0), dtype=np.float32)
    if train_demo.size == 0 or test_demo.size == 0:
        return empty_train, empty_test, []
    train_keys = list(basic_feature_keys(train_dataset))
    test_keys = list(basic_feature_keys(test_dataset))
    shared = [key for key in train_keys if key in test_keys]
    if not shared:
        return empty_train, empty_test, []
    train_idx = [train_keys.index(key) for key in shared]
    test_idx = [test_keys.index(key) for key in shared]
    return (
        np.asarray(train_demo[:, train_idx], dtype=np.float32),
        np.asarray(test_demo[:, test_idx], dtype=np.float32),
        shared,
    )


def prepare_ood_waveforms(
    train_dataset: str,
    test_dataset: str,
    *,
    signal_length: int,
    include_basic: bool,
    window_seconds: float,
    max_samples: int | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    train_signals, train_demo, train_y, train_groups = load_waveform_split(
        train_dataset,
        signal_length=signal_length,
        include_basic=include_basic,
        window_seconds=window_seconds,
        max_samples=max_samples,
    )
    test_signals, test_demo, test_y, test_groups = load_waveform_split(
        test_dataset,
        signal_length=signal_length,
        include_basic=include_basic,
        window_seconds=window_seconds,
        max_samples=max_samples,
    )
    test_signals = _resize_signals(test_signals, train_signals.shape[1])
    if include_basic:
        train_demo, test_demo, demo_keys = align_ood_demographics(
            train_demo, test_demo, train_dataset, test_dataset
        )
    else:
        demo_keys = []
        train_demo = np.empty((len(train_y), 0), dtype=np.float32)
        test_demo = np.empty((len(test_y), 0), dtype=np.float32)

    train_signals = normalize_signals_per_sample(train_signals)
    test_signals = normalize_signals_per_sample(test_signals)
    print(
        f'OOD waveforms: {train_dataset} → {test_dataset}\n'
        f'  train: {len(train_y)} samples, {len(np.unique(train_groups))} subjects, '
        f'length={train_signals.shape[1]}\n'
        f'  test:  {len(test_y)} samples, {len(np.unique(test_groups))} subjects, '
        f'length={test_signals.shape[1]}\n'
        f'  demographics: {demo_keys if demo_keys else "PPG only"}'
    )
    return (
        train_signals,
        train_demo,
        train_y,
        train_groups,
        test_signals,
        test_demo,
        test_y,
        test_groups,
    )


def run_waveform_ood(
    train_signals: np.ndarray,
    train_demo: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    test_signals: np.ndarray,
    test_demo: np.ndarray,
    test_y: np.ndarray,
    *,
    build_model: Callable[[int], torch.nn.Module],
    train_model: Callable,
    predict: Callable,
    args,
    device: torch.device,
    seeds: Sequence[int],
    model_name: str,
    drop_last: bool = False,
    val_fraction: float = 0.25,
) -> np.ndarray:
    """Train on the source dataset, evaluate on the held-out target dataset."""
    use_basic = train_demo.size > 0 and test_demo.size > 0
    source_index = np.arange(len(train_y))
    preds = []
    for seed_index, seed in enumerate(seeds, start=1):
        print(f'\n--- Seed {seed} ({seed_index}/{len(seeds)}) ---')
        set_seed(seed)
        try:
            inner_train, inner_val = split_train_val(
                source_index, train_groups, val_fraction, seed
            )
        except ValueError:
            inner_train, inner_val = split_train_val(
                source_index, None, val_fraction, seed
            )

        y_train, y_mean, y_std = normalize_targets(train_y[inner_train])
        train_d = val_d = test_d = None
        if use_basic:
            train_d, test_d = standardize_train_test(
                train_demo[inner_train], test_demo
            )
            _, val_d = standardize_train_test(
                train_demo[inner_train], train_demo[inner_val]
            )

        train_loader = make_signal_loader(
            train_signals[inner_train],
            y_train,
            args.batch_size,
            True,
            demographics=train_d,
            seed=seed,
            drop_last=drop_last,
        )
        y_val = ((train_y[inner_val] - y_mean) / y_std).astype(np.float32, copy=False)
        val_loader = make_signal_loader(
            train_signals[inner_val],
            y_val,
            args.batch_size,
            False,
            demographics=val_d,
        )
        test_loader = make_signal_loader(
            test_signals,
            None,
            args.batch_size,
            False,
            demographics=test_d,
        )

        n_demo = train_d.shape[1] if use_basic else 0
        model = build_model(n_demo).to(device)
        train_model(model, train_loader, val_loader, args, device, use_basic)
        pred = predict(model, test_loader, device, use_basic, y_mean, y_std)
        preds.append(pred)
        print(f'Seed {seed}: test MAE={float(np.mean(np.abs(pred - test_y))):.4f} mmol/L')

    y_pred = np.mean(np.column_stack(preds), axis=1)
    evaluation.print_regression_report(
        f'{model_name} ({args.train_dataset} → {args.test_dataset})',
        test_y,
        y_pred,
    )
    evaluation.print_regression_report(
        'Mean baseline (mean train y)',
        test_y,
        np.full(len(test_y), float(train_y.mean())),
    )
    return y_pred

