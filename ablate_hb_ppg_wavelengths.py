"""Ablate Hb_PPG CatBoost wavelength subsets (load 4-channel features once)."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import data_loader
import train_catboost
import utils
from config import BASIC_FEATURE_KEYS
from data_ppg import csv_column_tag

DATASET = 'Hb_PPG'
WAVELENGTHS = ('660nm', '730nm', '850nm', '940nm')


def _select_columns(
    all_cols: list[str],
    waves: tuple[str, ...],
    include_demo: bool,
) -> list[str]:
    keep: list[str] = []
    if include_demo:
        keep.extend(c for c in BASIC_FEATURE_KEYS[DATASET] if c in all_cols)
    tags = [f'_{csv_column_tag(wave)}_' for wave in waves]
    keep.extend(c for c in all_cols if any(tag in c for tag in tags))
    return keep


def _metrics(summary_df: pd.DataFrame) -> dict:
    row = summary_df.iloc[0]
    return {
        'mae': float(row['mae_mean']),
        'rmse': float(row['rmse_mean']),
        'r2': float(row['r2_mean']),
        'acc': float(row['acc_mean']),
        'zone_a': float(row['zone_a_mean']),
        'mard': float(row['mard_mean']),
    }


def _run_sweep(X_full, y, feature_names, params, include_demo: bool) -> pd.DataFrame:
    combos = [
        combo
        for n in range(1, len(WAVELENGTHS) + 1)
        for combo in combinations(WAVELENGTHS, n)
    ]
    results = []
    mode = 'demo+PPG' if include_demo else 'PPG only'
    print(f'\n=== {mode}: {len(combos)} wavelength subsets ===\n', flush=True)
    for i, waves in enumerate(combos, start=1):
        cols = _select_columns(feature_names, waves, include_demo)
        label = '+'.join(waves)
        print(
            f'[{i}/{len(combos)}] {label:28s} n={len(cols):3d} ...',
            end=' ',
            flush=True,
        )
        summary_df, _ = utils.run_repeated_kfold_training(
            X_full[cols],
            y,
            lambda seed: train_catboost.build_model(seed, params),
            cols,
            model_name='catboost',
            n_splits=5,
            verbose=False,
        )
        metrics = _metrics(summary_df)
        print(
            f'MAE={metrics["mae"]:.4f}  RMSE={metrics["rmse"]:.4f}  '
            f'R2={metrics["r2"]:+.4f}  ACC={metrics["acc"]:.2f}  '
            f'ZoneA={metrics["zone_a"]:.2f}',
            flush=True,
        )
        results.append({
            'mode': mode,
            'wavelengths': label,
            'n_wavelengths': len(waves),
            'n_features': len(cols),
            **metrics,
        })
    return pd.DataFrame(results)


def main() -> None:
    ds = data_loader.load_dataset(DATASET)
    X_full, y, feature_names = data_loader.split(ds)
    params = train_catboost.load_params(DATASET)
    print(
        f'Loaded {len(y)} samples, {X_full.shape[1]} features '
        f'(including {list(BASIC_FEATURE_KEYS[DATASET])}).'
    )

    with_demo = _run_sweep(X_full, y, feature_names, params, include_demo=True)
    ppg_only = _run_sweep(X_full, y, feature_names, params, include_demo=False)
    df = pd.concat([with_demo, ppg_only], ignore_index=True)

    Path('results').mkdir(exist_ok=True)
    out = Path('results/hb_ppg_wavelength_ablation.csv')
    df.to_csv(out, index=False)

    print('\n=== With demographics, by #wavelengths (mean MAE) ===')
    grouped = with_demo.groupby('n_wavelengths')['mae']
    print(grouped.agg(['count', 'mean', 'min', 'max']).to_string(float_format=lambda x: f'{x:.4f}'))
    print('\n=== Best / worst with demographics ===')
    ranked = with_demo.sort_values(['mae', 'r2'], ascending=[True, False])
    print(ranked.head(5).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('...')
    print(ranked.tail(3).to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    singles = with_demo[with_demo['n_wavelengths'] == 1]
    all_four = with_demo[with_demo['n_wavelengths'] == 4].iloc[0]
    best_single = singles.sort_values('mae').iloc[0]
    print(
        f'\nBest single: {best_single["wavelengths"]} MAE={best_single["mae"]:.4f} '
        f'R2={best_single["r2"]:+.4f} ACC={best_single["acc"]:.2f}'
    )
    print(
        f'All four:    {all_four["wavelengths"]} MAE={all_four["mae"]:.4f} '
        f'R2={all_four["r2"]:+.4f} ACC={all_four["acc"]:.2f}  '
        f'(delta vs best single {all_four["mae"] - best_single["mae"]:+.4f} MAE)'
    )
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
