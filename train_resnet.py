import argparse
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import data_loader
import utils
from config import (
    DEFAULT_SEEDS,
    DEFAULT_WINDOW_SECONDS,
    tensor_to_numpy,
    torch_device,
)
from TinyML_code.resnet import resnet34
from waveform_train import (
    expand_waveform_windows,
    make_signal_loader,
    normalize_signals_per_sample,
    normalize_targets,
    set_seed,
    split_train_val,
    standardize_train_test,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train a 1D ResNet-34 PPG regressor')
    parser.add_argument(
        '--dataset',
        default='IPS',
        help='IPS, MUST, vitaldb (16 min files under dataset/vitaldb_16min), or a path',
    )
    parser.add_argument(
        '--ppg-seconds',
        type=float,
        default=None,
        help='Center-crop PPG to this many seconds around mid-file. '
             'Default: 60 for vitaldb, full signal for other datasets. 0 = full signal.',
    )
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--max-samples', type=int, default=None, help='Use only the first N label files')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--signal-length', type=int, default=2048, help='0 = median waveform length')
    parser.add_argument(
        '--window-seconds',
        type=float,
        default=None,
        help='Chop each PPG into non-overlapping windows (same glucose label per window). '
             f'Default: config.DEFAULT_WINDOW_SECONDS ({DEFAULT_WINDOW_SECONDS}; 0=full)',
    )
    parser.add_argument(
        '--sample-wise',
        action='store_true',
        help='Force random KFold over windows/rows (ignore subject_ID). Reproduces the '
             'leaky segment-level split used by TinyML / Biosensors papers.',
    )
    parser.add_argument(
        '--pas',
        action='store_true',
        help='Participant-aware split: hold out a session from subjects with 2+ '
             'recordings; single-session subjects stay in train. MUST and VitalDB only.',
    )
    parser.add_argument(
        '--include-basic',
        dest='include_basic',
        action='store_true',
        help='Concatenate overlapping demographics (default: PPG only)',
    )
    parser.add_argument('--no-basic', dest='include_basic', action='store_false')
    parser.add_argument(
        '--quick',
        action='store_true',
        help='CPU smoke test with fewer epochs and shorter signals',
    )
    parser.set_defaults(include_basic=False)
    return parser


def apply_quick_settings(args) -> None:
    args.n_splits = 3
    args.epochs = 3
    args.patience = 1
    args.signal_length = 512


class ResNetRegressor(nn.Module):
    def __init__(self, n_demographics: int = 0):
        super().__init__()
        self.n_demographics = n_demographics
        self.backbone = resnet34(in_channel=1)
        self.backbone.fc2 = nn.Identity()
        self.output = nn.Linear(128 + n_demographics, 1)

    def forward(
        self,
        signals: torch.Tensor,
        demographics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.backbone(signals)
        if self.n_demographics:
            if demographics is None:
                raise ValueError('Demographic features are required by this model')
            features = torch.cat((features, demographics), dim=1)
        return self.output(features).squeeze(-1)


def make_loader(
    signals: np.ndarray,
    demographics: np.ndarray,
    targets: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    include_basic: bool,
    seed: Optional[int] = None,
) -> DataLoader:
    return make_signal_loader(
        signals,
        targets,
        batch_size,
        shuffle,
        demographics=demographics if include_basic else None,
        seed=seed,
    )


def unpack_batch(batch, include_basic: bool, device: torch.device, with_target: bool):
    signals = batch[0].to(device)
    offset = 1
    demographics = None
    if include_basic:
        demographics = batch[offset].to(device)
        offset += 1
    targets = batch[offset].to(device) if with_target else None
    return signals, demographics, targets


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
    include_basic: bool,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=max(1, args.patience // 2),
    )
    best_state = None
    best_val_loss = float('inf')
    remaining_patience = args.patience

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch_x, batch_d, batch_y = unpack_batch(batch, include_basic, device, True)
            optimizer.zero_grad()
            loss = criterion(model(batch_x, batch_d), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch_x, batch_d, batch_y = unpack_batch(batch, include_basic, device, True)
                val_losses.append(criterion(model(batch_x, batch_d), batch_y).item())

        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        print(
            f'Epoch {epoch + 1}/{args.epochs} - '
            f'train MSE: {np.mean(train_losses):.4f}, val MSE: {val_loss:.4f}'
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            remaining_patience = args.patience
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                print('Early stopping')
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    include_basic: bool,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_d, _ = unpack_batch(batch, include_basic, device, False)
            predictions.append(tensor_to_numpy(model(batch_x, batch_d)))
    normalized = np.concatenate(predictions)
    return (normalized * target_std + target_mean).astype(np.float32, copy=False)


def run_fold(
    signals: np.ndarray,
    demographics: np.ndarray,
    targets: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    args,
    device: torch.device,
    groups: Optional[np.ndarray],
    seed: int,
) -> np.ndarray:
    # 60/20/20: outer fold gives the 20% test set; split the 80% pool into
    # inner-train (60% overall) and inner-val (20% overall) for early stopping.
    inner_train_index, inner_val_index = split_train_val(
        train_index,
        groups,
        val_fraction=0.25,
        seed=seed,
    )

    train_signals = signals[inner_train_index]
    val_signals = signals[inner_val_index]
    test_signals = signals[test_index]

    y_train = targets[inner_train_index]
    y_val = targets[inner_val_index]
    y_train, target_mean, target_std = normalize_targets(y_train)
    y_val = ((y_val - target_mean) / target_std).astype(np.float32, copy=False)

    use_basic = args.include_basic and demographics.size > 0
    if use_basic:
        # Standardize val and test with train-fold statistics only.
        train_demographics, val_demographics = standardize_train_test(
            demographics[inner_train_index],
            demographics[inner_val_index],
        )
        _, test_demographics = standardize_train_test(
            demographics[inner_train_index],
            demographics[test_index],
        )
    else:
        train_demographics = np.empty((len(inner_train_index), 0), dtype=np.float32)
        val_demographics = np.empty((len(inner_val_index), 0), dtype=np.float32)
        test_demographics = np.empty((len(test_index), 0), dtype=np.float32)

    train_loader = make_loader(
        train_signals,
        train_demographics,
        y_train,
        args.batch_size,
        True,
        use_basic,
        seed=seed,
    )
    val_loader = make_loader(
        val_signals,
        val_demographics,
        y_val,
        args.batch_size,
        False,
        use_basic,
    )
    prediction_loader = make_loader(
        test_signals,
        test_demographics,
        None,
        args.batch_size,
        False,
        use_basic,
    )

    n_demographics = train_demographics.shape[1] if use_basic else 0
    model = ResNetRegressor(n_demographics=n_demographics).to(device)
    train_model(model, train_loader, val_loader, args, device, use_basic)
    return predict(
        model,
        prediction_loader,
        device,
        use_basic,
        target_mean,
        target_std,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        apply_quick_settings(args)

    window_seconds = (
        DEFAULT_WINDOW_SECONDS if args.window_seconds is None else args.window_seconds
    )
    seeds = tuple(args.seeds) if args.seeds is not None else DEFAULT_SEEDS
    set_seed(args.seed)
    device = torch.device('cpu') if args.quick else torch_device()

    from config import (
        DEFAULT_VITALDB_PPG_SECONDS,
        resolve_config_dataset,
        resolve_ppg_center_window_seconds,
    )
    from data_loader import resolve_dataset_root

    ppg_seconds = args.ppg_seconds
    if resolve_config_dataset(args.dataset) == 'vitaldb' and ppg_seconds is None:
        ppg_seconds = DEFAULT_VITALDB_PPG_SECONDS
    center = resolve_ppg_center_window_seconds(ppg_seconds)
    root = resolve_dataset_root(args.dataset)

    print(f'Using dataset: {args.dataset} (window_seconds={window_seconds})')
    if center is None:
        print(f'PPG: root={root}, using full on-disk signal')
    else:
        print(f'PPG: root={root}, center-crop {center:g} s')

    # Load native-length waveforms when windowing so we can chop before resize.
    load_length = 0 if window_seconds > 0 else args.signal_length
    signals, demographics, targets, groups, sessions = data_loader.load_waveform_dataset(
        dataset=args.dataset,
        signal_length=load_length,
        include_basic=args.include_basic,
        resample=True,
        max_samples=args.max_samples,
        return_groups=True,
        return_sessions=True,
        ppg_seconds=ppg_seconds,
    )
    if window_seconds > 0:
        n_labs = len(targets)
        signals, demographics, targets, groups, sessions = expand_waveform_windows(
            signals,
            demographics,
            targets,
            groups,
            dataset=args.dataset,
            window_seconds=window_seconds,
            signal_length=args.signal_length if args.signal_length > 0 else 1024,
            resample=True,
            sessions=sessions,
        )
        print(
            f'Windowed {n_labs} lab draws → {len(targets)} segments '
            f'({window_seconds:g} s, shared glucose labels)'
        )
    signals = normalize_signals_per_sample(signals)
    basic_mode = 'with basic info' if args.include_basic and demographics.size else 'PPG only'
    print(
        f'Training ResNet-34 on {device.type}: {len(targets)} samples, '
        f'signal length {signals.shape[1]} ({basic_mode})'
    )

    mode = utils.exclusive_split_mode(sample_wise=args.sample_wise, pas=args.pas)
    cv_sessions = None
    if mode == 'pas':
        cv_groups = groups
        cv_sessions = sessions
        print(utils.pas_split_summary(groups, sessions))
    elif mode == 'sample':
        cv_groups = None
        print(
            f'Sample-wise CV (KFold over rows; subject leakage allowed): '
            f'{len(targets)} samples across {len(np.unique(groups))} subjects'
        )
    else:
        cv_groups = groups
        print(f'Subject-wise CV: {len(np.unique(groups))} unique subjects')

    def predict_fold(
        train_index: np.ndarray,
        test_index: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        set_seed(seed)
        return run_fold(
            signals,
            demographics,
            targets,
            train_index,
            test_index,
            args,
            device,
            cv_groups,
            seed,
        )

    utils.run_repeated_kfold_array_cv(
        targets,
        predict_fold,
        model_name='resnet34',
        n_splits=args.n_splits,
        seeds=seeds,
        groups=cv_groups,
        sessions=cv_sessions,
    )


if __name__ == '__main__':
    main()
