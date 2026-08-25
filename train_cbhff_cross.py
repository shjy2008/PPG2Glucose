"""Train CBHFF on one PPG dataset, test on another (paper OOD).

  python train_cbhff_cross.py
  python train_cbhff_cross.py --train-dataset MUST --test-dataset southwest
  python train_cbhff_cross.py --train-dataset southwest --test-dataset vitaldb
  python train_cbhff_cross.py --quick --max-samples 32
"""

from __future__ import annotations

import argparse

import torch

from CBHFF import BloodGlucosePredictor
from config import DEFAULT_SEEDS, DEFAULT_WINDOW_SECONDS, torch_device
from train_cbhff import predict, train_model
from waveform_train import prepare_ood_waveforms, run_waveform_ood, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='OOD: train CBHFF on one PPG dataset, test on another',
    )
    parser.add_argument('--train-dataset', default='vitaldb')
    parser.add_argument('--test-dataset', default='MUST')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--signal-length', type=int, default=2048)
    parser.add_argument(
        '--window-seconds',
        type=float,
        default=None,
        help=f'Non-overlapping windows (default: {DEFAULT_WINDOW_SECONDS}; 0=full)',
    )
    parser.add_argument(
        '--include-basic',
        action='store_true',
        help='Use overlapping BASIC_FEATURE_KEYS (default: PPG only)',
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='CPU smoke test with fewer epochs and shorter signals',
    )
    return parser


def apply_quick_settings(args) -> None:
    args.epochs = 3
    args.patience = 1
    args.signal_length = 512


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        apply_quick_settings(args)
    window_seconds = (
        DEFAULT_WINDOW_SECONDS if args.window_seconds is None else args.window_seconds
    )
    seeds = tuple(args.seeds) if args.seeds is not None else DEFAULT_SEEDS
    if args.quick:
        seeds = (args.seed,)
    set_seed(args.seed)
    device = torch.device('cpu') if args.quick else torch_device()

    (
        train_signals,
        train_demo,
        train_y,
        train_groups,
        test_signals,
        test_demo,
        test_y,
        _,
    ) = prepare_ood_waveforms(
        args.train_dataset,
        args.test_dataset,
        signal_length=args.signal_length,
        include_basic=args.include_basic,
        window_seconds=window_seconds,
        max_samples=args.max_samples,
    )
    mode = 'with basic' if args.include_basic and train_demo.size else 'PPG only'
    print(
        f'Training CBHFF on {device.type}: '
        f'{len(train_y)} train / {len(test_y)} test, '
        f'len={train_signals.shape[1]}, {mode}'
    )

    def train_fn(model, train_loader, val_loader, train_args, train_device, _use_basic):
        train_model(model, train_loader, val_loader, train_args, train_device)

    def predict_fn(model, loader, pred_device, _use_basic, y_mean, y_std):
        return predict(model, loader, pred_device, y_mean, y_std)

    run_waveform_ood(
        train_signals,
        train_demo,
        train_y,
        train_groups,
        test_signals,
        test_demo,
        test_y,
        build_model=lambda n_demo: BloodGlucosePredictor(n_demographics=n_demo),
        train_model=train_fn,
        predict=predict_fn,
        args=args,
        device=device,
        seeds=seeds,
        model_name='CBHFF',
        drop_last=True,
    )


if __name__ == '__main__':
    main()
