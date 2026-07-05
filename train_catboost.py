import argparse
import json
from pathlib import Path
from typing import Optional

from catboost import CatBoostRegressor

import data_loader
import utils
import numpy as np
from config import DEFAULT_SEEDS, catboost_kwargs

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


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='southwest', help='southwest, MUST, or vitaldb')
    p.add_argument(
        '--params',
        default=None,
        help='JSON file with tuned hyperparameters (e.g. results/catboost_best_params.json)',
    )
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--max-samples', type=int, default=None, help='Use only the first N label files')
    p.add_argument(
        '--select-top',
        type=int,
        default=None,
        metavar='N',
        help='Keep top N features per fold (ranked on train split only)',
    )
    p.add_argument('--top', type=int, default=20, help='Number of top features to print')
    return p


def load_params(path: Optional[str] = None) -> dict:
    if path is None:
        return dict(DEFAULT_PARAMS)
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
    ds = data_loader.load_dataset(args.dataset, max_samples=args.max_samples)
    print(f"Using dataset: {args.dataset}")
    X, y, feature_names = data_loader.split(ds)
    print(
        f"Data for training - X.size: {X.shape[0]}, y.size: {y.shape[0]}, "
        f"features: {X.shape[1]}"
        + (f' (using top {args.select_top} per fold)' if args.select_top else '')
    )

    params = load_params(args.params)
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
    print(f"Training MAE: {mae:.4f} mmol/L")

    summary_df, fi = utils.run_repeated_kfold_training(
        X,
        y,
        lambda seed: build_model(seed, params),
        feature_names,
        model_name='catboost',
        n_splits=args.n_splits,
        select_top_n=args.select_top,
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
