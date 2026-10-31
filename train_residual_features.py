"""Learn PPG features outside the handcrafted set, then score them with CatBoost.

Each subject-wise fold fits a linear model on the current CatBoost inputs
(demographics + handcrafted PPG). A small 1-D network is trained only on that
fold to predict the leftover glucose from the raw waveform. Its embedding is
penalized for correlating with those inputs. CatBoost is then trained twice
on the same fold: handcrafted inputs alone, and those inputs plus the new
embedding. The readout and the split stay fixed, so a gain is attributable
to the new features.

  python train_residual_features.py --dataset MUST
  python train_residual_features.py --dataset IPS
  python train_residual_features.py --dataset vitaldb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import data_loader
import evaluation
import train_catboost
import utils
from config import DEFAULT_SEEDS, torch_device
from waveform_train import as_nct, normalize_signals_per_sample, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('IPS', 'MUST', 'vitaldb'))
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument(
        '--seeds',
        default=','.join(str(seed) for seed in DEFAULT_SEEDS),
        help='Comma-separated CV seeds. Default matches train_catboost.py.',
    )
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embed-dim', type=int, default=4)
    parser.add_argument(
        '--wave-length',
        type=int,
        default=512,
        help='Temporal length of the waveform seen by the feature network.',
    )
    parser.add_argument(
        '--corr-weight',
        type=float,
        default=0.1,
        help='Penalty on correlation between new features and handcrafted inputs.',
    )
    parser.add_argument('--signal-length', type=int, default=2048)
    return parser


class ResidualFeatureNet(nn.Module):
    """Map a waveform to a few features, with a linear head for the residual."""

    def __init__(self, n_features: int, in_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(32, n_features)
        self.head = nn.Linear(n_features, 1)

    def embed(self, waveforms: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(waveforms).squeeze(-1)
        return self.proj(hidden)

    def forward(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.embed(waveforms)
        return features, self.head(features).squeeze(-1)


def _to_tensor(values: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(values, dtype=np.float32)
    try:
        return torch.from_numpy(array)
    except RuntimeError:
        # This environment's PyTorch was built against NumPy 1.x.
        copied = torch.frombuffer(bytearray(array.tobytes()), dtype=torch.float32)
        return copied.reshape(array.shape).clone()


def _to_numpy(values: torch.Tensor) -> np.ndarray:
    values = values.detach().cpu().contiguous()
    try:
        return values.numpy()
    except RuntimeError:
        flat = np.asarray(values.reshape(-1).tolist(), dtype=np.float32)
        return flat.reshape(values.shape)


def _device() -> torch.device:
    chosen = torch_device()
    if chosen.type == 'cpu' and torch.backends.mps.is_available():
        return torch.device('mps')
    return chosen


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part) for part in raw.split(',') if part.strip())
    if not seeds:
        raise SystemExit('--seeds must list at least one integer')
    return seeds


def _downsample(signals: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float32)
    source = values.shape[-1]
    if source == length:
        return values
    if source < 2:
        raise ValueError(f'waveform length {source} is too short to downsample')
    position = np.linspace(0, source - 1, length, dtype=np.float32)
    left = np.floor(position).astype(np.int64)
    right = np.minimum(left + 1, source - 1)
    weight = (position - left).astype(np.float32)
    return values[..., left] * (1.0 - weight) + values[..., right] * weight


def _row_keys(subjects: np.ndarray, sessions: np.ndarray | None) -> list[tuple[str, str]]:
    subject_text = np.asarray(subjects).astype(str)
    if sessions is None:
        session_text = np.array([''] * len(subject_text))
    else:
        session_text = np.asarray(sessions).astype(str)
    return list(zip(subject_text, session_text))


def _alignment_order(
    frame: pd.DataFrame,
    groups: np.ndarray,
    sessions: np.ndarray,
) -> np.ndarray:
    """Row order that puts each waveform next to the same tabular recording."""
    table_has_session = (
        'session_ID' in frame.columns and frame['session_ID'].nunique(dropna=True) > 1
    )
    wave_sessions = np.asarray(sessions).astype(str)
    wave_has_session = len(set(wave_sessions.tolist())) > 1
    if table_has_session and wave_has_session:
        wave_index = {
            key: index for index, key in enumerate(_row_keys(groups, sessions))
        }
        table_keys = _row_keys(
            frame['subject_ID'].to_numpy(),
            frame['session_ID'].to_numpy(),
        )
    else:
        wave_subjects = np.asarray(groups).astype(str)
        if len(wave_subjects) != len(set(wave_subjects.tolist())):
            raise SystemExit('waveforms repeat subject_ID but sessions do not identify them')
        wave_index = {subject: index for index, subject in enumerate(wave_subjects)}
        table_keys = frame['subject_ID'].astype(str).tolist()
    missing = [key for key in table_keys if key not in wave_index]
    if missing:
        raise SystemExit(
            f'{len(missing)} tabular rows have no waveform, e.g. {missing[:3]}'
        )
    return np.asarray([wave_index[key] for key in table_keys], dtype=np.int64)


def load_aligned(dataset: str, signal_length: int, wave_length: int):
    """Handcrafted table and waveforms in the same row order."""
    frame = data_loader.load_dataset(dataset, refresh_cache=False)
    signals, _demo, wave_targets, groups, sessions = data_loader.load_waveform_dataset(
        dataset,
        signal_length=signal_length,
        resample=True,
        include_basic=False,
        return_groups=True,
        return_sessions=True,
        refresh_cache=False,
    )
    order = _alignment_order(frame, groups, sessions)
    signals = _downsample(signals[order], wave_length)
    signals = normalize_signals_per_sample(as_nct(signals))
    target_gap = np.nanmax(np.abs(
        frame['target'].to_numpy(dtype=float) - np.asarray(wave_targets, dtype=float)[order]
    ))
    if target_gap > 1e-3:
        raise SystemExit(f'tabular and waveform glucose differ by up to {target_gap:.4f}')

    feature_columns = [
        column
        for column in frame.columns
        if column not in {'subject_ID', 'session_ID', 'window_ID', 'target'}
    ]
    features = frame[feature_columns].reset_index(drop=True)
    target = frame['target'].reset_index(drop=True)
    subjects = frame['subject_ID'].to_numpy()
    print(
        f'{dataset}: {len(target)} rows, {features.shape[1]} handcrafted inputs, '
        f'waveform {tuple(signals.shape)}'
    )
    return features, target, signals, subjects


def _prepare_base(
    features: pd.DataFrame,
    train_index: np.ndarray,
    val_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    train = scaler.fit_transform(imputer.fit_transform(features.iloc[train_index]))
    val = scaler.transform(imputer.transform(features.iloc[val_index]))
    return train.astype(np.float32), val.astype(np.float32)


def _residual(base_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    model = Ridge(alpha=1.0)
    model.fit(base_train, y_train)
    return (y_train - model.predict(base_train)).astype(np.float32)


def _correlation_penalty(embedding: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    if base.shape[1] == 0 or embedding.shape[0] < 3:
        return embedding.new_zeros(())
    embedding_centered = embedding - embedding.mean(dim=0, keepdim=True)
    base_centered = base - base.mean(dim=0, keepdim=True)
    embedding_scale = embedding_centered.std(dim=0, unbiased=False).clamp_min(1e-6)
    base_scale = base_centered.std(dim=0, unbiased=False)
    usable = base_scale > 1e-6
    if not bool(usable.any()):
        return embedding.new_zeros(())
    embedding_norm = embedding_centered / embedding_scale
    base_norm = base_centered[:, usable] / base_scale[usable]
    correlation = embedding_norm.transpose(0, 1) @ base_norm / embedding.shape[0]
    return correlation.square().mean()


def train_embedding(
    waveforms: np.ndarray,
    base: np.ndarray,
    residual: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    embed_dim: int,
    corr_weight: float,
    seed: int,
) -> ResidualFeatureNet:
    set_seed(seed)
    device = _device()
    model = ResidualFeatureNet(embed_dim, in_channels=int(waveforms.shape[1])).to(device)
    residual_std = float(np.std(residual))
    if residual_std < 1e-8:
        residual_std = 1.0
    target = _to_tensor(residual / residual_std)
    data = TensorDataset(
        _to_tensor(waveforms),
        _to_tensor(base),
        target,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _epoch in range(epochs):
        for wave_batch, base_batch, residual_batch in loader:
            wave_batch = wave_batch.to(device)
            base_batch = base_batch.to(device)
            residual_batch = residual_batch.to(device)
            _embedding, prediction = model(wave_batch)
            loss = nn.functional.mse_loss(prediction, residual_batch)
            loss = loss + corr_weight * _correlation_penalty(_embedding, base_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model


def embed(
    model: ResidualFeatureNet,
    waveforms: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    device = _device()
    model = model.to(device)
    loader = DataLoader(
        TensorDataset(_to_tensor(waveforms)),
        batch_size=batch_size,
        shuffle=False,
    )
    pieces = []
    with torch.no_grad():
        for (wave_batch,) in loader:
            pieces.append(_to_numpy(model.embed(wave_batch.to(device))))
    return np.concatenate(pieces, axis=0).astype(np.float32)


def _catboost_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    seed: int,
    params: dict,
) -> np.ndarray:
    model = train_catboost.build_model(seed, params)
    model.fit(train_x, train_y)
    return model.predict(val_x).astype(float)


def run(args: argparse.Namespace) -> pd.DataFrame:
    seeds = _parse_seeds(args.seeds)
    features, target, waveforms, subjects = load_aligned(
        args.dataset,
        args.signal_length,
        args.wave_length,
    )
    params = train_catboost.load_params(args.dataset)
    y = target.to_numpy(dtype=float)
    fold_rows: list[dict] = []
    names = ('handcrafted', 'handcrafted+new')

    for seed in seeds:
        for fold_id, (train_index, val_index) in enumerate(
            utils._iter_cv_splits(y, args.n_splits, seed, groups=subjects),
            start=1,
        ):
            base_train, base_val = _prepare_base(features, train_index, val_index)
            leftover = _residual(base_train, y[train_index])
            network = train_embedding(
                waveforms[train_index],
                base_train,
                leftover,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                embed_dim=args.embed_dim,
                corr_weight=args.corr_weight,
                seed=seed + fold_id,
            )
            new_train = embed(network, waveforms[train_index], args.batch_size)
            new_val = embed(network, waveforms[val_index], args.batch_size)
            y_train = y[train_index]
            y_val = y[val_index]
            raw_train = features.iloc[train_index].to_numpy(dtype=np.float64)
            raw_val = features.iloc[val_index].to_numpy(dtype=np.float64)
            raw_train[~np.isfinite(raw_train)] = np.nan
            raw_val[~np.isfinite(raw_val)] = np.nan
            predictions = {
                'handcrafted': _catboost_predict(
                    raw_train, y_train, raw_val, seed, params
                ),
                'handcrafted+new': _catboost_predict(
                    np.concatenate([raw_train, new_train], axis=1),
                    y_train,
                    np.concatenate([raw_val, new_val], axis=1),
                    seed,
                    params,
                ),
            }
            for name in names:
                metrics = evaluation.evaluate_regression(y_val, predictions[name])
                fold_rows.append({
                    'dataset': args.dataset,
                    'model': name,
                    'seed': seed,
                    'fold': fold_id,
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2'],
                })
            base_mae = fold_rows[-2]['mae']
            new_mae = fold_rows[-1]['mae']
            print(
                f'{args.dataset} seed {seed} fold {fold_id}: '
                f'handcrafted MAE {base_mae:.4f}  '
                f'+new MAE {new_mae:.4f}  '
                f'delta {new_mae - base_mae:+.4f}',
                flush=True,
            )

    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby('model', sort=False)
        .agg(
            mae_mean=('mae', 'mean'),
            mae_std=('mae', 'std'),
            rmse_mean=('rmse', 'mean'),
            rmse_std=('rmse', 'std'),
            r2_mean=('r2', 'mean'),
            r2_std=('r2', 'std'),
        )
        .reset_index()
    )
    summary.insert(0, 'dataset', args.dataset)
    print()
    for _, row in summary.iterrows():
        print(
            f"{row['model']:18s}  "
            f"MAE {row['mae_mean']:.4f}±{row['mae_std']:.4f}  "
            f"RMSE {row['rmse_mean']:.4f}±{row['rmse_std']:.4f}  "
            f"R2 {row['r2_mean']:.4f}±{row['r2_std']:.4f}"
        )
    base = summary.loc[summary['model'] == 'handcrafted'].iloc[0]
    extra = summary.loc[summary['model'] == 'handcrafted+new'].iloc[0]
    print(
        f"delta (+new − handcrafted)  "
        f"MAE {extra['mae_mean'] - base['mae_mean']:+.4f}  "
        f"R2 {extra['r2_mean'] - base['r2_mean']:+.4f}"
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'residual_features_{args.dataset}.csv'
    summary.to_csv(out_path, index=False)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
