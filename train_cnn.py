import argparse
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import data_loader
import utils
from config import tensor_to_numpy, torch_device, torch_float_tensor


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='southwest', help='southwest or kaggle')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--signal-length', type=int, default=0, help='0 = median waveform length')
    p.add_argument('--include-basic', dest='include_basic', action='store_true')
    p.add_argument('--no-basic', dest='include_basic', action='store_false')
    p.set_defaults(include_basic=True)
    return p


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean()
    std = X_train.std()
    if std == 0:
        std = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


class PPGCNNRegressor(nn.Module):
    def __init__(self, input_length: int, n_demographics: int = 0):
        super().__init__()
        self.n_demographics = n_demographics
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
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
            nn.AdaptiveAvgPool1d(1),
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
    demographics: Optional[np.ndarray] = None,
) -> DataLoader:
    tensor_x = torch_float_tensor(X[:, None, :])
    if y is None:
        dataset = TensorDataset(tensor_x)
    elif demographics is not None:
        tensor_d = torch_float_tensor(demographics)
        tensor_y = torch_float_tensor(y)
        dataset = TensorDataset(tensor_x, tensor_d, tensor_y)
    else:
        tensor_y = torch_float_tensor(y)
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
    val_index: np.ndarray,
    args,
    device: torch.device,
    demographics: Optional[np.ndarray] = None,
) -> np.ndarray:
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    X_train, X_val = standardize_train_test(X_train, X_val)

    use_demographics = args.include_basic and demographics is not None and demographics.size > 0
    demo_train = demo_val = None
    n_demographics = 0
    if use_demographics:
        demo_train, demo_val = demographics[train_index], demographics[val_index]
        demo_train, demo_val = standardize_train_test(demo_train, demo_val)
        n_demographics = demo_train.shape[1]

    train_loader = make_loader(
        X_train, y_train, args.batch_size, True, demographics=demo_train
    )
    val_loader = make_loader(
        X_val, y_val, args.batch_size, False, demographics=demo_val
    )

    model = PPGCNNRegressor(input_length=X.shape[1], n_demographics=n_demographics).to(device)
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

    predictions = predict(model, val_loader, device, use_demographics=use_demographics)
    return predictions


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch_device()

    print(f'Using dataset: {args.dataset}')
    signals, demographics, y = data_loader.load_waveform_dataset(
        dataset=args.dataset,
        signal_length=args.signal_length,
        include_basic=args.include_basic,
    )
    X = signals
    print(
        f'Data for training - X.size: {X.shape[0]}, y.size: {y.shape[0]}, '
        f'waveform_length: {signals.shape[1]}, input_length: {X.shape[1]}'
    )

    def predict_fold(train_index: np.ndarray, val_index: np.ndarray, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        np.random.seed(seed)
        return run_fold(
            X, y, train_index, val_index, args, device, demographics=demographics
        )

    utils.run_repeated_kfold_array_cv(
        y,
        predict_fold,
        model_name='cnn',
        n_splits=args.n_splits,
    )


if __name__ == '__main__':
    main()
