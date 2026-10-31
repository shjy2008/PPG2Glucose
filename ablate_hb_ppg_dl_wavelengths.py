"""Ablate Hb_PPG DL wavelength subsets (load 4-channel waveforms once).

Smoke (CBHFF, 940nm vs all four, 32 subjects, 2-fold × 1 seed × 2 epochs):

  python ablate_hb_ppg_dl_wavelengths.py --smoke

Full table (TinyML + CBHFF, all 15 subsets, subject-wise 5×5, PPG only):

  python ablate_hb_ppg_dl_wavelengths.py
  python ablate_hb_ppg_dl_wavelengths.py --include-basic
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import data_loader
import evaluation
import train_cbhff
import train_resnet
import utils
from config import BASIC_FEATURE_KEYS, DEFAULT_SEEDS, ppg_csv_column, torch_device
from waveform_train import normalize_signals_per_sample, set_seed, waveform_n_channels

DATASET = 'Hb_PPG'
WAVELENGTHS = tuple(ppg_csv_column(DATASET) or ())
MODEL_LABELS = {
    'resnet': 'TinyML(ResNet34)',
    'cbhff': 'CBHFF',
}
PASTE_COLUMNS = (
    'dataset',
    'Method used',
    'subjects',
    'wavelength',
    'MAE(mmol/L)',
    'RMSE(mmol/L)',
    'R2',
    'Accuracy',
    'Zone A',
    'Zone B',
    'Zone A+B',
    'MARD',
    "Pearson's r",
)


def _combos(waves: tuple[str, ...]) -> list[tuple[str, ...]]:
    return [
        combo
        for n in range(1, len(waves) + 1)
        for combo in combinations(waves, n)
    ]


def _select_channels(signals: np.ndarray, waves: tuple[str, ...]) -> np.ndarray:
    names = list(WAVELENGTHS)
    arr = np.asarray(signals)
    if arr.ndim != 3 or arr.shape[1] != len(names):
        raise ValueError(
            f'Expected (N, {len(names)}, T) Hb_PPG waveforms, got {arr.shape}. '
            'Set PPG_CSV_COLUMN to all four wavelengths and rebuild waveform_cache.'
        )
    idx = [names.index(wave) for wave in waves]
    return np.ascontiguousarray(arr[:, idx, :])


def _method_used(model: str, include_basic: bool) -> str:
    name = MODEL_LABELS[model]
    if not include_basic:
        return name
    keys = ', '.join(repr(key) for key in BASIC_FEATURE_KEYS[DATASET])
    return f'{name}({keys})'


def _trainer_args(model: str, *, epochs: int, patience: int, batch_size: int) -> SimpleNamespace:
    if model == 'resnet':
        return SimpleNamespace(
            include_basic=False,
            patience=patience,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=1e-3,
            weight_decay=1e-4,
        )
    return SimpleNamespace(
        patience=patience,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        weight_decay=0.0,
    )


def _paste_row(method: str, n_subjects: int, wavelengths: str, summary: pd.Series) -> dict:
    return {
        'dataset': DATASET,
        'Method used': method,
        'subjects': n_subjects,
        'wavelength': wavelengths,
        'MAE(mmol/L)': f'{summary["mae_mean"]:.4f}',
        'RMSE(mmol/L)': f'{summary["rmse_mean"]:.4f}',
        'R2': f'{summary["r2_mean"]:.4f}',
        'Accuracy': f'{summary["acc_mean"]:.2f}',
        'Zone A': f'{summary["zone_a_mean"]:.2f}',
        'Zone B': f'{summary["zone_b_mean"]:.2f}',
        'Zone A+B': f'{summary["zone_ab_mean"]:.2f}',
        'MARD': f'{summary["mard_mean"]:.2f}',
        "Pearson's r": evaluation.format_pearson_summary(
            summary['pearson_oof_r'], summary['pearson_oof_p']
        ),
    }


def _run_one(
    model: str,
    signals: np.ndarray,
    demographics: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    args: SimpleNamespace,
    device,
    n_splits: int,
    seeds: tuple[int, ...],
) -> pd.Series:
    use_basic = demographics.size > 0
    if model == 'resnet':
        args.include_basic = use_basic

    def predict_fold(train_index, test_index, seed):
        set_seed(seed)
        with redirect_stdout(StringIO()):
            if model == 'resnet':
                return train_resnet.run_fold(
                    signals, demographics, targets, train_index, test_index,
                    args, device, groups, seed,
                )
            return train_cbhff.run_fold(
                signals, demographics, targets, train_index, test_index,
                args, device, groups, seed,
            )

    _, summary_df, _ = utils.run_repeated_kfold_array_cv(
        targets,
        predict_fold,
        model_name=model,
        n_splits=n_splits,
        seeds=seeds,
        groups=groups,
        print_target_distribution=False,
        verbose=False,
    )
    return summary_df.iloc[0]


def _load(include_basic: bool, signal_length: int, max_samples: int | None):
    signals, demographics, targets, groups = data_loader.load_waveform_dataset(
        dataset=DATASET,
        signal_length=signal_length,
        include_basic=include_basic,
        resample=True,
        max_samples=max_samples,
        return_groups=True,
        window_seconds=0.0,
    )
    signals = normalize_signals_per_sample(signals)
    print(
        f'Loaded {len(targets)} samples, {waveform_n_channels(signals)} ch, '
        f'len={signals.shape[-1]}, '
        f'{"+".join(BASIC_FEATURE_KEYS[DATASET]) if include_basic else "PPG only"}',
        flush=True,
    )
    return signals, demographics, targets, groups


def main() -> None:
    p = argparse.ArgumentParser(description='Hb_PPG DL wavelength ablation')
    p.add_argument('--smoke', action='store_true', help='Tiny CV to check the table pipeline')
    p.add_argument('--models', nargs='+', choices=('resnet', 'cbhff'), default=None)
    p.add_argument('--include-basic', action='store_true')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--signal-length', type=int, default=2048)
    p.add_argument('--max-samples', type=int, default=None)
    args = p.parse_args()

    if len(WAVELENGTHS) < 2:
        raise SystemExit(
            f'PPG_CSV_COLUMN[{DATASET!r}] must list multiple wavelengths, got {WAVELENGTHS!r}'
        )

    models = tuple(args.models) if args.models else ('resnet', 'cbhff')
    seeds = tuple(args.seeds) if args.seeds is not None else DEFAULT_SEEDS
    n_splits = args.n_splits
    patience = args.patience
    batch_size = args.batch_size
    max_samples = args.max_samples
    combos = _combos(WAVELENGTHS)
    default_epochs = {'resnet': 25, 'cbhff': 30}
    if args.smoke:
        models = tuple(args.models) if args.models else ('cbhff',)
        seeds = (42,)
        n_splits = 2
        patience = 1
        batch_size = 8 if args.batch_size == 32 else args.batch_size
        max_samples = 32 if args.max_samples is None else args.max_samples
        default_epochs = {'resnet': 2, 'cbhff': 2}
        combos = [('940nm',), WAVELENGTHS]
        print(
            f'Smoke: {models}, {len(combos)} subsets, {max_samples} subjects, '
            f'{n_splits}-fold × {len(seeds)} seed',
            flush=True,
        )

    device = torch_device()
    signals, demographics, targets, groups = _load(
        args.include_basic, args.signal_length, max_samples
    )
    n_subjects = int(len(targets))
    method_by_model = {model: _method_used(model, args.include_basic) for model in models}

    paste_rows = []
    n_jobs = len(models) * len(combos)
    job = 0
    for model in models:
        trainer_args = _trainer_args(
            model,
            epochs=args.epochs if args.epochs is not None else default_epochs[model],
            patience=patience,
            batch_size=batch_size,
        )
        print(
            f'\n=== {method_by_model[model]}: {len(combos)} wavelength subsets '
            f'on {device.type} ===\n',
            flush=True,
        )
        for waves in combos:
            job += 1
            label = '+'.join(waves)
            subset = _select_channels(signals, waves)
            print(
                f'[{job}/{n_jobs}] {MODEL_LABELS[model]:18s} {label:28s} '
                f'{waveform_n_channels(subset)} ch ...',
                end=' ',
                flush=True,
            )
            summary = _run_one(
                model, subset, demographics, targets, groups,
                trainer_args, device, n_splits, seeds,
            )
            print(
                f'MAE={summary["mae_mean"]:.4f}  RMSE={summary["rmse_mean"]:.4f}  '
                f'R2={summary["r2_mean"]:+.4f}  ACC={summary["acc_mean"]:.2f}  '
                f'ZoneA={summary["zone_a_mean"]:.2f}',
                flush=True,
            )
            paste_rows.append(
                _paste_row(method_by_model[model], n_subjects, label, summary)
            )

    paste = pd.DataFrame(paste_rows, columns=list(PASTE_COLUMNS))
    Path('results').mkdir(exist_ok=True)
    stem = 'hb_ppg_dl_wavelength_ablation' + ('_smoke' if args.smoke else '')
    csv_path = Path('results') / f'{stem}.csv'
    tsv_path = Path('results') / f'{stem}.tsv'
    paste.to_csv(csv_path, index=False)
    paste.to_csv(tsv_path, index=False, sep='\t')

    print('\n=== Copy-paste (TSV) ===')
    print(paste.to_csv(sep='\t', index=False), end='')
    print(f'Saved {csv_path} and {tsv_path}')


if __name__ == '__main__':
    main()
