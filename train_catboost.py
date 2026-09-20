import argparse
import json
from pathlib import Path
from typing import Optional

from catboost import CatBoostRegressor

import data_loader
import utils
import numpy as np
from config import DEFAULT_SEEDS, DEFAULT_WINDOW_SECONDS, catboost_kwargs
from data_loader import add_prediction_label_argument, retarget_tabular_frame

DEFAULT_PARAMS = {
    # 'iterations': 400,
    # 'learning_rate': 0.05,
    # 'depth': 3,
    # 'subsample': 0.9,
    # 'reg_lambda': 1.0,

    # 'iterations': 450,
    # 'learning_rate': 0.025629380669206163,
    # 'depth': 3,
    # 'subsample': 0.7318547877163166,
    # 'reg_lambda': 0.1617396356788761,
    # 'min_data_in_leaf': 7,
    # 'border_count': 192,

    'iterations': 500,
    'learning_rate': 0.025,
    'depth': 3,
    'subsample': 0.75,
    'reg_lambda': 0.15,
    'min_data_in_leaf': 7,
    'border_count': 190,
}

# Dataset-specific overrides; edit these independently after tuning each dataset.
DATASET_PARAMS = {
    'IPS': {'depth': 4,},
    # MUST: depth 3 → 4 with age+weight + g3-g6: MAE 0.7109 → 0.7075, R2 -0.0140 → +0.0003.
    'MUST': {
        # 'iterations': 500,
        'depth': 4,
        },
    'vitaldb': {},
    'Hb_PPG': {'depth': 4},
}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--dataset',
        default='IPS',
        help='IPS, MUST, vitaldb, Hb_PPG, or a path',
    )
    p.add_argument(
        '--ppg-seconds',
        type=float,
        default=None,
        help='Center-crop PPG to this many seconds around mid-file. '
             'Default: 60 for vitaldb, full signal for other datasets. 0 = full signal.',
    )
    p.add_argument(
        '--params',
        default=None,
        help='JSON file with tuned hyperparameters (e.g. results/catboost_best_params.json)',
    )
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--max-samples', type=int, default=None, help='Use only the first N label files')
    p.add_argument(
        '--window-seconds',
        type=float,
        default=None,
        help='Non-overlapping PPG window length in seconds (default: config.DEFAULT_WINDOW_SECONDS; 0=full)',
    )
    p.add_argument(
        '--subject-wise',
        action='store_true',
        help='Use GroupKFold so the same subject_ID never appears in both train and val '
             '(auto-enabled when any subject has multiple samples, e.g. MUST or windowed PPG)',
    )
    p.add_argument(
        '--sample-wise',
        action='store_true',
        help='Force random KFold over rows (ignore subject_ID). Useful to reproduce '
             'leaky segment-level splits; overrides auto subject-wise CV.',
    )
    p.add_argument(
        '--pas',
        action='store_true',
        help='Participant-aware split: hold out a session from subjects with 2+ '
             'recordings; single-session subjects stay in train. MUST and VitalDB only.',
    )
    p.add_argument(
        '--select-top',
        type=int,
        default=None,
        metavar='N',
        help='Keep top N features per fold (ranked on train split only)',
    )
    p.add_argument('--top', type=int, default=20, help='Number of top features to print')
    p.add_argument(
        '--refresh-cache',
        action='store_true',
        help='Re-extract PPG features and overwrite the feature cache',
    )
    p.add_argument(
        '--only-basic',
        action='store_true',
        help='Drop PPG features; train on BASIC_FEATURE_KEYS only (demographics baseline)',
    )
    add_prediction_label_argument(p, keep_flag='--only-basic')
    return p


def load_params(dataset: str, path: Optional[str] = None) -> dict:
    if path is None:
        from config import resolve_config_dataset

        return {**DEFAULT_PARAMS, **DATASET_PARAMS.get(resolve_config_dataset(dataset), {})}
    payload = json.loads(Path(path).read_text())
    if 'params' in payload:
        return dict(payload['params'])
    return dict(payload)


def build_model(seed: int, params: Optional[dict] = None):
    model_params = dict(DEFAULT_PARAMS if params is None else params)
    return CatBoostRegressor(
        loss_function='MAE',
        random_seed=seed,
        verbose=False,
        **model_params,
        **catboost_kwargs(),
    )


def main():
    args = build_parser().parse_args()
    window_seconds = (
        DEFAULT_WINDOW_SECONDS if args.window_seconds is None else args.window_seconds
    )
    from config import (
        DEFAULT_VITALDB_PPG_SECONDS,
        basic_feature_keys,
        ppg_best_segment_seconds,
        ppg_csv_column,
        ppg_feature_groups,
        ppg_filter,
        ppg_sqa_method,
        ppg_sqa_threshold,
        ppg_target_fs,
        resolve_config_dataset,
        resolve_ppg_center_window_seconds,
    )
    from data_loader import resolve_dataset_root

    ppg_seconds = args.ppg_seconds
    if resolve_config_dataset(args.dataset) == 'vitaldb' and ppg_seconds is None:
        ppg_seconds = DEFAULT_VITALDB_PPG_SECONDS
    center = resolve_ppg_center_window_seconds(ppg_seconds)
    root = resolve_dataset_root(args.dataset)
    if center is None:
        print(f'PPG: root={root}, using full on-disk signal')
    else:
        print(f'PPG: root={root}, center-crop {center:g} s')

    ds = data_loader.load_dataset(
        args.dataset,
        max_samples=args.max_samples,
        window_seconds=window_seconds,
        refresh_cache=args.refresh_cache,
        ppg_seconds=ppg_seconds,
    )
    ds, label, y_unit, plot_clarke = retarget_tabular_frame(ds, args.label)
    # Leakage if the same subject appears in train and val with multiple rows
    # (MUST/vitaldb sessions, or windowed PPG).
    multi_sample_subjects = (
        'subject_ID' in ds.columns and ds.groupby('subject_ID').size().max() > 1
    )
    mode = utils.exclusive_split_mode(
        sample_wise=args.sample_wise,
        subject_wise=args.subject_wise,
        pas=args.pas,
    )
    print(
        f"Using dataset: {args.dataset} "
        f"(window_seconds={window_seconds}, ppg_filter={ppg_filter(args.dataset)}, "
        f"target_fs={ppg_target_fs(args.dataset)}, "
        f"best_segment_seconds={ppg_best_segment_seconds(args.dataset)}, "
        f"feature_groups={ppg_feature_groups(args.dataset)}, "
        f"ppg_csv_column={ppg_csv_column(args.dataset)}, "
        f"sqa_method={ppg_sqa_method(args.dataset)}, "
        f"sqa_threshold={ppg_sqa_threshold(args.dataset)})"
    )
    sessions = None
    if mode == 'pas':
        X, y, feature_names, groups, sessions = data_loader.split(
            ds, return_groups=True, return_sessions=True
        )
        if groups is None or sessions is None:
            raise SystemExit('PAS requires subject_ID and session_ID (MUST or vitaldb)')
        print(utils.pas_split_summary(groups, sessions))
    elif mode != 'sample' and (mode == 'subject' or multi_sample_subjects):
        X, y, feature_names, groups = data_loader.split(ds, return_groups=True)
        if groups is None:
            raise RuntimeError('subject-wise CV requires subject_ID in the dataset')
        print(
            f"Subject-wise CV: {len(np.unique(groups))} unique subjects "
            f"across {len(y)} samples"
        )
    else:
        X, y, feature_names = data_loader.split(ds)
        groups = None
        if mode == 'sample':
            print(
                f"Sample-wise CV (KFold over rows; subject leakage allowed): "
                f"{len(y)} samples"
            )
    if label != 'glucose' and label in X.columns:
        X = X.drop(columns=[label])
        feature_names = [name for name in feature_names if name != label]
        print(f'Dropped {label} from X (would leak the prediction target)')
    if args.only_basic:
        keys = [k for k in basic_feature_keys(args.dataset) if k in X.columns]
        missing = [k for k in basic_feature_keys(args.dataset) if k not in X.columns]
        if missing:
            print(f'--only-basic: dropping missing demographic keys {missing}')
        if not keys:
            raise SystemExit(
                '--only-basic requires demographic columns from BASIC_FEATURE_KEYS; '
                f'none found in X ({list(X.columns)[:12]}…)'
            )
        X = X[keys]
        feature_names = keys
        print(f'PPG dropped; demographics only: {keys}')
    elif label != 'glucose':
        demo_keys = [k for k in ('sex', 'age', 'height', 'weight', 'bmi') if k in X.columns]
        if demo_keys:
            X = X.drop(columns=demo_keys)
            feature_names = [name for name in feature_names if name not in demo_keys]
            print(f'PPG-only {label}: dropped demographics {demo_keys}')
    else:
        # Glucose: keep only BASIC_FEATURE_KEYS demos (extra attach cols are for --label).
        allowed = set(basic_feature_keys(args.dataset))
        extra = [
            k for k in ('sex', 'age', 'height', 'weight', 'bmi')
            if k in X.columns and k not in allowed
        ]
        if extra:
            X = X.drop(columns=extra)
            feature_names = [name for name in feature_names if name not in extra]
            print(f'Glucose inputs: dropped non-BASIC demos {extra}')

    print(
        f"Data for training - X.size: {X.shape[0]}, y.size: {y.shape[0]}, "
        f"features: {X.shape[1]}"
        + (f' (using top {args.select_top} per fold)' if args.select_top else '')
    )

    params = load_params(args.dataset, args.params)
    if args.params:
        print(f'Using hyperparameters from {args.params}')

    def train_and_predict(seed):
        model = build_model(seed, params)
        model.fit(X, y)
        return model.predict(X)

    all_preds = []
    for seed in DEFAULT_SEEDS:
        all_preds.append(train_and_predict(seed))
    training_pred = np.mean(np.column_stack(all_preds), axis=1)
    mae = np.mean(np.abs(training_pred - y))
    print(f"Training MAE: {mae:.4f} {y_unit}")

    summary_df, fi = utils.run_repeated_kfold_training(
        X,
        y,
        lambda seed: build_model(seed, params),
        feature_names,
        model_name='catboost',
        n_splits=args.n_splits,
        select_top_n=args.select_top,
        groups=groups,
        sessions=sessions,
        y_unit=y_unit,
        plot_clarke=plot_clarke,
    )
    n_evaluations = int(summary_df['n_evaluations'].iloc[0])
    if args.select_top is not None:
        fi_used = fi[fi['importance'] > 0].reset_index(drop=True)
        n_print = min(args.top, len(fi_used))
        top = fi_used.head(n_print)[['feature']].copy()
        top['%'] = 100.0 * fi_used.head(n_print)['importance'].to_numpy()
        print(
            f'\nTop {n_print} features by mean importance when selected '
            f'({n_evaluations} fits, top {args.select_top} per fold; '
            f'{len(fi_used)} unique features used across folds):'
        )
    else:
        top = fi.head(args.top)[['feature']].copy()
        top['%'] = 100.0 * fi.head(args.top)['importance'].to_numpy()
        print(f'\nTop {args.top} features (mean % importance over {n_evaluations} fits):')
    print(top.to_string(index=False, formatters={'%': '{:.2f}'.format}))


if __name__ == '__main__':
    main()
