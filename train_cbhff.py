"""Train Ali et al. CBHFF (JBHI 2025) on southwest / MUST / vitaldb."""

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
    expand_waveform_windows,
    make_signal_loader,
    normalize_signals_per_sample,
    normalize_targets,
    set_seed,
    split_train_val,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Train CBHFF PPG→glucose regressor')
    p.add_argument('--dataset', default='southwest')
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
    p.add_argument('--window-seconds', type=float, default=None)
    p.add_argument(
        '--sample-wise',
        action='store_true',
        help='Random KFold over rows (paper-style leakage)',
    )
    p.add_argument(
        '--paper-protocol',
        action='store_true',
        help='15 s windows, sample-wise 10-fold, batch 128, 30 epochs, seed 8',
    )
    p.add_argument('--quick', action='store_true')
    return p


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
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x).squeeze(-1), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if val_loader is None:
            print(f'Epoch {epoch + 1}/{args.epochs} - train MSE: {np.mean(train_losses):.4f}')
            continue

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_losses.append(
                    criterion(model(batch_x).squeeze(-1), batch_y).item()
                )
        val_loss = float(np.mean(val_losses))
        print(
            f'Epoch {epoch + 1}/{args.epochs} - '
            f'train MSE: {np.mean(train_losses):.4f}, val MSE: {val_loss:.4f}'
        )
        if args.patience <= 0:
            continue
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
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
        for (batch_x,) in loader:
            preds.append(tensor_to_numpy(model(batch_x.to(device)).squeeze(-1)))
    return (np.concatenate(preds) * target_std + target_mean).astype(np.float32)


def run_fold(signals, targets, train_index, test_index, args, device, groups, seed):
    if args.patience > 0:
        inner_train, inner_val = split_train_val(train_index, groups, 0.25, seed)
    else:
        inner_train, inner_val = train_index, None

    y_train, y_mean, y_std = normalize_targets(targets[inner_train])
    # drop_last: ChannelGate BN fails on a trailing batch of size 1
    train_loader = make_signal_loader(
        signals[inner_train], y_train, args.batch_size, True, seed=seed, drop_last=True
    )
    val_loader = None
    if inner_val is not None:
        y_val = ((targets[inner_val] - y_mean) / y_std).astype(np.float32)
        val_loader = make_signal_loader(
            signals[inner_val], y_val, args.batch_size, False
        )

    model = BloodGlucosePredictor().to(device)
    train_model(model, train_loader, val_loader, args, device)
    test_loader = make_signal_loader(signals[test_index], None, args.batch_size, False)
    return predict(model, test_loader, device, y_mean, y_std)


def main() -> None:
    args = build_parser().parse_args()
    if args.paper_protocol:
        args.window_seconds = 15.0
        args.sample_wise = True
        args.n_splits = 10
        args.epochs = 30
        args.batch_size = 128
        args.patience = 0
        args.seeds = [8]
        args.seed = 8
    if args.quick:
        args.n_splits = 3
        args.epochs = 2
        args.patience = 1
        args.signal_length = 512
        args.batch_size = 16
        args.max_samples = args.max_samples or 48
        args.seeds = [42]

    window_seconds = (
        DEFAULT_WINDOW_SECONDS if args.window_seconds is None else args.window_seconds
    )
    seeds = tuple(args.seeds) if args.seeds is not None else DEFAULT_SEEDS
    set_seed(args.seed)
    device = torch.device('cpu') if args.quick else torch_device()

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

    load_length = 0 if window_seconds > 0 else args.signal_length
    signals, _, targets, groups = data_loader.load_waveform_dataset(
        dataset=args.dataset,
        signal_length=load_length,
        include_basic=False,
        resample=True,
        max_samples=args.max_samples,
        return_groups=True,
        ppg_seconds=ppg_seconds,
    )
    if window_seconds > 0:
        n_labs = len(targets)
        empty_demo = np.empty((len(targets), 0), dtype=np.float32)
        signals, _, targets, groups = expand_waveform_windows(
            signals,
            empty_demo,
            targets,
            groups,
            dataset=args.dataset,
            window_seconds=window_seconds,
            signal_length=args.signal_length if args.signal_length > 0 else 1024,
        )
        print(f'Windowed {n_labs} → {len(targets)} segments ({window_seconds:g} s)')

    signals = normalize_signals_per_sample(signals)
    cv_groups = None if args.sample_wise else groups
    print(
        f'Training CBHFF on {device.type}: {len(targets)} samples, '
        f'len={signals.shape[1]}, '
        + (
            f'sample-wise ({len(np.unique(groups))} subjects)'
            if args.sample_wise
            else f'subject-wise ({len(np.unique(groups))} subjects)'
        )
    )

    def predict_fold(train_index, test_index, seed):
        set_seed(seed)
        return run_fold(
            signals, targets, train_index, test_index, args, device, cv_groups, seed
        )

    utils.run_repeated_kfold_array_cv(
        targets,
        predict_fold,
        model_name='cbhff',
        n_splits=args.n_splits,
        seeds=seeds,
        groups=cv_groups,
    )


if __name__ == '__main__':
    main()
