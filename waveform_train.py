"""Shared helpers for 1D PPG waveform deep-learning trainers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

import data_loader
from config import ppg_sampling_rate, ppg_target_fs, torch_float_tensor
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
