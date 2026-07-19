import argparse
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import data_loader
import utils
from config import tensor_to_numpy, torch_device, torch_float_tensor
from TinyML_code.resnet import resnet34


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train a 1D ResNet-34 PPG regressor')
    parser.add_argument('--dataset', default='southwest', help='southwest, MUST, or vitaldb')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--max-samples', type=int, default=None, help='Use only the first N label files')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--signal-length', type=int, default=2048, help='0 = median waveform length')
    parser.add_argument('--include-basic', dest='include_basic', action='store_true')
    parser.add_argument('--no-basic', dest='include_basic', action='store_false')
    parser.add_argument(
        '--quick',
        action='store_true',
        help='CPU smoke test with fewer epochs and shorter signals',
    )
    parser.set_defaults(include_basic=True)
    return parser


def apply_quick_settings(args) -> None:
    args.n_splits = 3
    args.epochs = 3
    args.patience = 1
    args.signal_length = 512


def normalize_signals_per_sample(signals: np.ndarray) -> np.ndarray:
    mean = signals.mean(axis=1, keepdims=True)
    std = signals.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return ((signals - mean) / std).astype(np.float32, copy=False)


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


def normalize_targets(y_train: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(y_train.mean())
    std = float(y_train.std())
    if std == 0:
        std = 1.0
    normalized = ((y_train - mean) / std).astype(np.float32, copy=False)
    return normalized, mean, std


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
) -> DataLoader:
    # torch.tensor avoids torch.from_numpy, which breaks under NumPy 2.x + older PyTorch.
    tensors = [torch_float_tensor(signals[:, None, :])]
    if include_basic and demographics.size:
        tensors.append(torch_float_tensor(demographics))
    if targets is not None:
        tensors.append(torch_float_tensor(targets))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


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
    val_index: np.ndarray,
    args,
    device: torch.device,
) -> np.ndarray:
    train_signals, val_signals = signals[train_index], signals[val_index]
    y_train, y_val = targets[train_index], targets[val_index]
    y_train, target_mean, target_std = normalize_targets(y_train)
    y_val = ((y_val - target_mean) / target_std).astype(np.float32, copy=False)

    use_basic = args.include_basic and demographics.size > 0
    if use_basic:
        train_demographics, val_demographics = standardize_train_test(
            demographics[train_index],
            demographics[val_index],
        )
    else:
        train_demographics = np.empty((len(train_index), 0), dtype=np.float32)
        val_demographics = np.empty((len(val_index), 0), dtype=np.float32)

    train_loader = make_loader(
        train_signals,
        train_demographics,
        y_train,
        args.batch_size,
        True,
        use_basic,
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
        val_signals,
        val_demographics,
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cpu') if args.quick else torch_device()

    print(f'Using dataset: {args.dataset}')
    signals, demographics, targets, groups = data_loader.load_waveform_dataset(
        dataset=args.dataset,
        signal_length=args.signal_length,
        include_basic=args.include_basic,
        resample=True,
        max_samples=args.max_samples,
        return_groups=True,
    )
    signals = normalize_signals_per_sample(signals)
    basic_mode = 'with basic info' if args.include_basic and demographics.size else 'PPG only'
    print(
        f'Training ResNet-34 on {device.type}: {len(targets)} samples, '
        f'signal length {signals.shape[1]} ({basic_mode})'
    )
    print(f'Subject-wise CV: {len(np.unique(groups))} unique subjects')

    def predict_fold(
        train_index: np.ndarray,
        val_index: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        torch.manual_seed(seed)
        np.random.seed(seed)
        return run_fold(
            signals,
            demographics,
            targets,
            train_index,
            val_index,
            args,
            device,
        )

    utils.run_repeated_kfold_array_cv(
        targets,
        predict_fold,
        model_name='resnet34',
        n_splits=args.n_splits,
        groups=groups,
    )


if __name__ == '__main__':
    main()
