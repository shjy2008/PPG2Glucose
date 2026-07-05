import argparse
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import data_loader
import utils
from config import torch_device


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='southwest', help='southwest or MUST')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
    p.add_argument('--signal-length', type=int, default=2048, help='0 = median waveform length')
    p.add_argument('--hidden-size', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=1)
    p.add_argument(
        '--quick',
        action='store_true',
        help='Fast CPU smoke test: 3 folds, 8 epochs, shorter signals, smaller unidirectional LSTM',
    )
    p.add_argument('--bidirectional', dest='bidirectional', action='store_true')
    p.add_argument('--unidirectional', dest='bidirectional', action='store_false')
    p.add_argument('--include-basic', dest='include_basic', action='store_true')
    p.add_argument('--no-basic', dest='include_basic', action='store_false')
    p.set_defaults(bidirectional=True, include_basic=True)
    return p


def apply_quick_settings(args) -> None:
    args.n_splits = 5
    args.epochs = 8
    args.batch_size = 32
    args.patience = 2
    args.signal_length = 1024
    args.hidden_size = 32
    args.bidirectional = False


def normalize_signals_per_sample(signals: np.ndarray) -> np.ndarray:
    mean = signals.mean(axis=1, keepdims=True)
    std = signals.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return ((signals - mean) / std).astype(np.float32, copy=False)


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def normalize_targets(y_train: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(y_train.mean())
    std = float(y_train.std())
    if std == 0:
        std = 1.0
    return ((y_train - mean) / std).astype(np.float32, copy=False), mean, std


def denormalize_targets(y_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (y_norm * std + mean).astype(np.float32, copy=False)


class PPGLSTMRegressor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        include_basic: bool,
        n_basic: int,
    ):
        super().__init__()
        self.include_basic = include_basic
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out = hidden_size * 2 if bidirectional else hidden_size
        head_in = lstm_out + (n_basic if include_basic else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, signals: torch.Tensor, demographics: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs, _ = self.lstm(signals)
        features = outputs.mean(dim=1)
        if self.include_basic and demographics is not None:
            features = torch.cat((features, demographics), dim=1)
        return self.head(features).squeeze(-1)


def make_loader(
    signals: np.ndarray,
    demographics: np.ndarray,
    y: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    include_basic: bool,
) -> DataLoader:
    tensor_x = torch.from_numpy(signals).float().unsqueeze(-1)
    if include_basic and demographics.size:
        tensor_d = torch.from_numpy(demographics).float()
        if y is None:
            dataset = TensorDataset(tensor_x, tensor_d)
        else:
            tensor_y = torch.from_numpy(y).float()
            dataset = TensorDataset(tensor_x, tensor_d, tensor_y)
    else:
        if y is None:
            dataset = TensorDataset(tensor_x)
        else:
            tensor_y = torch.from_numpy(y).float()
            dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    include_basic: bool,
    patience: int,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = None
    best_val_loss = float('inf')
    remaining_patience = patience

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            if include_basic and len(batch) == 3:
                batch_x, batch_d, batch_y = batch
                batch_d = batch_d.to(device)
            else:
                batch_x, batch_y = batch
                batch_d = None
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x, batch_d)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if include_basic and len(batch) == 3:
                    batch_x, batch_d, batch_y = batch
                    batch_d = batch_d.to(device)
                else:
                    batch_x, batch_y = batch
                    batch_d = None
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_x, batch_d)
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
    include_basic: bool,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            if include_basic and len(batch) >= 2 and batch[0].dim() == 3:
                batch_x, batch_d = batch[0], batch[1]
                batch_d = batch_d.to(device)
            else:
                batch_x = batch[0]
                batch_d = None
            batch_x = batch_x.to(device)
            batch_predictions = model(batch_x, batch_d).detach().cpu().numpy()
            predictions.append(batch_predictions)
    if not predictions:
        return np.empty(0, dtype=np.float32)
    return denormalize_targets(np.concatenate(predictions, axis=0), y_mean, y_std)


def run_fold(
    signals: np.ndarray,
    demographics: np.ndarray,
    y: np.ndarray,
    train_index: np.ndarray,
    val_index: np.ndarray,
    args,
    device: torch.device,
) -> np.ndarray:
    sig_train, sig_val = signals[train_index], signals[val_index]
    y_train, y_val = y[train_index], y[val_index]
    y_train_norm, y_mean, y_std = normalize_targets(y_train)
    y_val_norm = ((y_val - y_mean) / y_std).astype(np.float32, copy=False)

    if args.include_basic and demographics.size:
        dem_train, dem_val = demographics[train_index], demographics[val_index]
        dem_train, dem_val = standardize_train_test(dem_train, dem_val)
    else:
        dem_train = dem_val = np.empty((len(train_index), 0), dtype=np.float32)

    train_loader = make_loader(
        sig_train, dem_train, y_train_norm, batch_size=args.batch_size, shuffle=True, include_basic=args.include_basic
    )
    val_loader = make_loader(
        sig_val, dem_val, y_val_norm, batch_size=args.batch_size, shuffle=False, include_basic=args.include_basic
    )

    n_basic = demographics.shape[1] if args.include_basic and demographics.size else 0
    model = PPGLSTMRegressor(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        include_basic=args.include_basic,
        n_basic=n_basic,
    ).to(device)
    train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        include_basic=args.include_basic,
        patience=args.patience,
    )

    predictions = predict(model, val_loader, device, include_basic=args.include_basic, y_mean=y_mean, y_std=y_std)
    return predictions


def main():
    args = build_parser().parse_args()
    if args.quick:
        apply_quick_settings(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cpu') if args.quick else torch_device()

    print(f'Using dataset: {args.dataset}')
    signals, demographics, y = data_loader.load_waveform_dataset(
        dataset=args.dataset,
        signal_length=args.signal_length,
        include_basic=args.include_basic,
        resample=True,
    )
    signals = normalize_signals_per_sample(signals)
    model_type = 'BiLSTM' if args.bidirectional else 'LSTM'
    basic_mode = 'with basic info' if args.include_basic else 'PPG only'
    print(f'Training on {device.type}')
    if args.quick:
        print(
            'Quick mode: '
            f'{args.n_splits} folds, {args.epochs} epochs, signal_length={args.signal_length}, '
            f'hidden_size={args.hidden_size}, {model_type}'
        )
    print(
        f'Data for training - X.size: {signals.shape[0]}, y.size: {y.shape[0]}, '
        f'waveform_length: {signals.shape[1]}, model: {model_type} ({basic_mode})'
    )

    def predict_fold(train_index: np.ndarray, val_index: np.ndarray, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        np.random.seed(seed)
        return run_fold(
            signals, demographics, y, train_index, val_index, args, device
        )

    utils.run_repeated_kfold_array_cv(
        y,
        predict_fold,
        model_name=model_type.lower(),
        n_splits=args.n_splits,
    )


if __name__ == '__main__':
    main()
