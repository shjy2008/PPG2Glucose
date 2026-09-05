"""Train Ali et al. CBHFF (JBHI 2025) on IPS / MUST / vitaldb.

  python train_cbhff.py --dataset IPS
  python train_cbhff.py --dataset IPS --sample-wise --window-seconds 15
  python train_cbhff.py --dataset IPS --include-basic
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import data_loader
import utils
from CBHFF import BloodGlucosePredictor
from config import (
    DEFAULT_SEEDS,
    DEFAULT_VITALDB_PPG_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    resolve_config_dataset,
    resolve_ppg_center_window_seconds,
    tensor_to_numpy,
    torch_device,
)
from data_loader import resolve_dataset_root
from waveform_train import (
    make_signal_loader,
    normalize_signals_per_sample,
    normalize_targets,
    set_seed,
    split_train_val,
    standardize_train_test,
    waveform_length,
    waveform_n_channels,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Train CBHFF PPG→glucose regressor')
    p.add_argument('--dataset', default='IPS')
    p.add_argument('--ppg-seconds', type=float, default=None)
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--signal-length', type=int, default=2048)
    p.add_argument(
        '--window-seconds',
        type=float,
        default=None,
        help=f'Non-overlapping windows (default: {DEFAULT_WINDOW_SECONDS}; 0 = full signal)',
    )
    p.add_argument('--sample-wise', action='store_true')
    p.add_argument(
        '--pas',
        action='store_true',
        help='Participant-aware split: hold out a session from subjects with 2+ '
             'recordings; single-session subjects stay in train. MUST and VitalDB only.',
    )
    p.add_argument(
        '--include-basic',
        action='store_true',
        help='Use BASIC_FEATURE_KEYS from config (default: PPG only)',
    )
    return p


def _batch(batch, device, with_target: bool):
    """Unpack (x[, demo][, y]) from make_signal_loader."""
    x = batch[0].to(device)
    if with_target:
        demo = batch[1].to(device) if len(batch) == 3 else None
        return x, demo, batch[-1].to(device)
    demo = batch[1].to(device) if len(batch) == 2 else None
    return x, demo, None


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    args,
    device: torch.device,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = None
    best_val = float('inf')
    remaining = args.patience if args.patience > 0 else args.epochs + 1

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            x, demo, y = _batch(batch, device, True)
            optimizer.zero_grad()
            loss = criterion(model(x, demo).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if val_loader is None:
            print(f'Epoch {epoch + 1}/{args.epochs} - train MSE: {np.mean(train_losses):.4f}')
            continue

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, demo, y = _batch(batch, device, True)
                val_losses.append(criterion(model(x, demo).squeeze(-1), y).item())
        val_loss = float(np.mean(val_losses))
        print(
            f'Epoch {epoch + 1}/{args.epochs} - '
            f'train MSE: {np.mean(train_losses):.4f}, val MSE: {val_loss:.4f}'
        )
        if args.patience <= 0:
            continue
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            remaining = args.patience
        else:
            remaining -= 1
            if remaining <= 0:
                print('Early stopping')
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def predict(model, loader, device, target_mean, target_std) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            x, demo, _ = _batch(batch, device, False)
            preds.append(tensor_to_numpy(model(x, demo).squeeze(-1)))
    return (np.concatenate(preds) * target_std + target_mean).astype(np.float32)


def run_fold(signals, demographics, targets, train_index, test_index, args, device, groups, seed):
    if args.patience > 0:
        inner_train, inner_val = split_train_val(train_index, groups, 0.25, seed)
    else:
        inner_train, inner_val = train_index, None

    y_train, y_mean, y_std = normalize_targets(targets[inner_train])
    use_basic = demographics.size > 0
    train_demo = val_demo = test_demo = None
    if use_basic:
        train_demo, test_demo = standardize_train_test(
            demographics[inner_train], demographics[test_index]
        )
        if inner_val is not None:
            _, val_demo = standardize_train_test(
                demographics[inner_train], demographics[inner_val]
            )

    # drop_last: ChannelGate BN fails on a trailing batch of size 1
    train_loader = make_signal_loader(
        signals[inner_train], y_train, args.batch_size, True,
        demographics=train_demo, seed=seed, drop_last=True,
    )
    val_loader = None
    if inner_val is not None:
        y_val = ((targets[inner_val] - y_mean) / y_std).astype(np.float32)
        val_loader = make_signal_loader(
            signals[inner_val], y_val, args.batch_size, False, demographics=val_demo
        )

    n_demo = train_demo.shape[1] if use_basic else 0
    model = BloodGlucosePredictor(
        n_demographics=n_demo,
        in_channel=waveform_n_channels(signals),
    ).to(device)
    train_model(model, train_loader, val_loader, args, device)
    test_loader = make_signal_loader(
        signals[test_index], None, args.batch_size, False, demographics=test_demo
    )
    return predict(model, test_loader, device, y_mean, y_std)


def main() -> None:
    args = build_parser().parse_args()
    window_seconds = (
        DEFAULT_WINDOW_SECONDS if args.window_seconds is None else args.window_seconds
    )
    seeds = tuple(args.seeds) if args.seeds is not None else DEFAULT_SEEDS
    set_seed(args.seed)
    device = torch_device()

    ppg_seconds = args.ppg_seconds
    if resolve_config_dataset(args.dataset) == 'vitaldb' and ppg_seconds is None:
        ppg_seconds = DEFAULT_VITALDB_PPG_SECONDS
    center = resolve_ppg_center_window_seconds(ppg_seconds)
    root = resolve_dataset_root(args.dataset)
    print(f'Using dataset: {args.dataset} (window_seconds={window_seconds})')
    print(
        f'PPG: root={root}, '
        + ('full on-disk signal' if center is None else f'center-crop {center:g} s')
    )

    signals, demographics, targets, groups, sessions = data_loader.load_waveform_dataset(
        dataset=args.dataset,
        signal_length=args.signal_length,
        include_basic=args.include_basic,
        resample=True,
        max_samples=args.max_samples,
        return_groups=True,
        return_sessions=True,
        ppg_seconds=ppg_seconds,
        window_seconds=window_seconds,
    )

    signals = normalize_signals_per_sample(signals)
    mode = utils.exclusive_split_mode(sample_wise=args.sample_wise, pas=args.pas)
    cv_sessions = None
    if mode == 'pas':
        cv_groups = groups
        cv_sessions = sessions
        split_desc = utils.pas_split_summary(groups, sessions)
    elif mode == 'sample':
        cv_groups = None
        split_desc = f'sample-wise ({len(np.unique(groups))} subjects)'
    else:
        cv_groups = groups
        split_desc = f'subject-wise ({len(np.unique(groups))} unique subjects)'
    basic_mode = 'with basic' if args.include_basic and demographics.size else 'PPG only'
    print(
        f'Training CBHFF on {device.type}: {len(targets)} samples, '
        f'len={waveform_length(signals)}, {waveform_n_channels(signals)} ch, '
        f'{basic_mode}, {split_desc}'
    )

    def predict_fold(train_index, test_index, seed):
        set_seed(seed)
        return run_fold(
            signals, demographics, targets, train_index, test_index,
            args, device, cv_groups, seed,
        )

    utils.run_repeated_kfold_array_cv(
        targets, predict_fold, model_name='cbhff',
        n_splits=args.n_splits, seeds=seeds, groups=cv_groups, sessions=cv_sessions,
    )


if __name__ == '__main__':
    main()
