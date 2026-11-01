import argparse
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import data_loader
import utils
from config import tensor_to_numpy, torch_device
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


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='IPS', help='IPS, MUST, or vitaldb')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--max-samples', type=int, default=None, help='Use only the first N label files')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--signal-length', type=int, default=0, help='0 = median waveform length')
    p.add_argument(
        '--window-seconds',
        type=float,
        default=0.0,
        help='Non-overlapping windows before the split. 0 keeps the whole recording (LSPO).',
    )
    p.add_argument(
        '--sample-wise',
        action='store_true',
        help='KFold over rows, so the same subject can sit in train and test (RTTS).',
    )
    p.add_argument(
        '--include-basic',
        dest='include_basic',
        action='store_true',
        help='Concatenate demographics (default: PPG only)',
    )
    p.add_argument('--no-basic', dest='include_basic', action='store_false')
    p.set_defaults(include_basic=False)
    return p


class _GlobalMean(nn.Module):
    """Mean over time. Same output as AdaptiveAvgPool1d(1), deterministic on GPU."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mean(dim=-1, keepdim=True)


class PPGCNNRegressor(nn.Module):
    def __init__(self, input_length: int, n_demographics: int = 0, in_channels: int = 1):
        super().__init__()
        self.n_demographics = n_demographics
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            _GlobalMean(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 + n_demographics, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, inputs, demographics=None):
        features = self.network(inputs)
        if self.n_demographics > 0:
            features = torch.cat([features.flatten(1), demographics], dim=1)
            return self.head(features).squeeze(-1)
        outputs = self.head(features)
        return outputs.squeeze(-1)


def make_loader(
    X: np.ndarray,
    y: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
    demographics: Optional[np.ndarray] = None,
) -> DataLoader:
    return make_signal_loader(
        X,
        y,
        batch_size,
        shuffle,
        demographics=demographics,
        seed=seed,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    use_demographics: bool = False,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = None
    best_val_loss = float('inf')
    patience = 5
    remaining_patience = patience

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            if use_demographics:
                batch_x, batch_d, batch_y = batch
                batch_x = batch_x.to(device)
                batch_d = batch_d.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                predictions = model(batch_x, batch_d)
            else:
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if use_demographics:
                    batch_x, batch_d, batch_y = batch
                    batch_x = batch_x.to(device)
                    batch_d = batch_d.to(device)
                    batch_y = batch_y.to(device)
                    predictions = model(batch_x, batch_d)
                else:
                    batch_x, batch_y = batch
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    predictions = model(batch_x)
                val_losses.append(criterion(predictions, batch_y).item())

        mean_val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()}
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_demographics: bool = False,
) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            if use_demographics:
                batch_x, batch_d = batch[0].to(device), batch[1].to(device)
                batch_predictions = tensor_to_numpy(model(batch_x, batch_d))
            else:
                batch_x = batch[0].to(device)
                batch_predictions = tensor_to_numpy(model(batch_x))
            predictions.append(batch_predictions)
    if not predictions:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(predictions, axis=0)


def run_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    args,
    device: torch.device,
    groups: Optional[np.ndarray] = None,
    demographics: Optional[np.ndarray] = None,
    seed: int = 0,
) -> np.ndarray:
    # Outer fold is the reported test set. Early stopping uses 25% of the
    # training subjects only, matching TinyML and CBHFF.
    inner_train, inner_val = split_train_val(train_index, groups, 0.25, seed)
    X_train, X_val, X_test = X[inner_train], X[inner_val], X[test_index]
    y_train, y_mean, y_std = normalize_targets(y[inner_train])
    y_val = ((y[inner_val] - y_mean) / y_std).astype(np.float32)

    use_demographics = args.include_basic and demographics is not None and demographics.size > 0
    demo_train = demo_val = demo_test = None
    n_demographics = 0
    if use_demographics:
        demo_train, demo_val = standardize_train_test(
            demographics[inner_train], demographics[inner_val]
        )
        _, demo_test = standardize_train_test(
            demographics[inner_train], demographics[test_index]
        )
        n_demographics = demo_train.shape[1]

    train_loader = make_loader(
        X_train, y_train, args.batch_size, True, seed, demographics=demo_train
    )
    val_loader = make_loader(
        X_val, y_val, args.batch_size, False, seed, demographics=demo_val
    )
    test_loader = make_loader(
        X_test, None, args.batch_size, False, seed, demographics=demo_test
    )

    model = PPGCNNRegressor(
        input_length=waveform_length(X),
        n_demographics=n_demographics,
        in_channels=waveform_n_channels(X),
    ).to(device)
    train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        use_demographics=use_demographics,
    )

    predictions = predict(model, test_loader, device, use_demographics=use_demographics)
    return predictions * y_std + y_mean


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = torch_device()

    window_seconds = float(args.window_seconds or 0.0)
    sample_wise = bool(args.sample_wise)
    if sample_wise and window_seconds <= 0:
        protocol = 'row-wise KFold'
    elif sample_wise:
        protocol = 'RTTS'
    elif window_seconds > 0:
        protocol = 'STTS'
    else:
        protocol = 'LSPO'
    print(f'Using dataset: {args.dataset} ({protocol})')
    signals, demographics, y, groups = data_loader.load_waveform_dataset(
        dataset=args.dataset,
        signal_length=args.signal_length,
        include_basic=args.include_basic,
        resample=args.signal_length > 0 or window_seconds > 0,
        max_samples=args.max_samples,
        return_groups=True,
        window_seconds=window_seconds,
    )
    X = normalize_signals_per_sample(signals)
    print(
        f'Data for training - X.size: {X.shape[0]}, y.size: {y.shape[0]}, '
        f'waveform_length: {waveform_length(signals)}, '
        f'channels: {waveform_n_channels(signals)}, input_length: {waveform_length(X)}'
    )
    cv_groups = None if sample_wise else groups
    if cv_groups is None:
        print(f'Row-wise CV: {len(y)} rows, {len(np.unique(groups))} subjects')
    else:
        print(f'Subject-wise CV ({protocol}): {len(np.unique(groups))} unique subjects')

    def predict_fold(train_index: np.ndarray, val_index: np.ndarray, seed: int) -> np.ndarray:
        set_seed(seed)
        return run_fold(
            X,
            y,
            train_index,
            val_index,
            args,
            device,
            groups=cv_groups,
            demographics=demographics,
            seed=seed,
        )

    utils.run_repeated_kfold_array_cv(
        y,
        predict_fold,
        model_name='cnn',
        n_splits=args.n_splits,
        groups=cv_groups,
    )


if __name__ == '__main__':
    main()
