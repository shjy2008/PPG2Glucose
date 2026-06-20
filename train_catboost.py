import argparse
import json
from pathlib import Path
from typing import Optional

from catboost import CatBoostRegressor

import data_loader
import utils
from config import catboost_kwargs

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
    p.add_argument('--dataset', default='southwest', help='southwest or kaggle')
    p.add_argument(
        '--params',
        default=None,
        help='JSON file with tuned hyperparameters (e.g. results/catboost_best_params.json)',
    )
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=list(utils.DEFAULT_SEEDS),
        help='Random seeds for fold shuffling (default: 42 52 62 72 82)',
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
    ds = data_loader.load_dataset(args.dataset)
    print(f"Using dataset: {args.dataset}")
    X, y, feature_names = data_loader.split(ds)
    n_evaluations = args.n_splits * len(args.seeds)
    print(f"Data for training - X.size: {X.shape[0]}, y.size: {y.shape[0]}, features: {X.shape[1]}")

    params = load_params(args.params)
    if args.params:
        print(f'Using hyperparameters from {args.params}')

    _, fi = utils.run_repeated_kfold_training(
        X,
        y,
        feature_names,
        model_factory=lambda seed: build_model(seed, params),
        model_name='catboost',
        n_splits=args.n_splits,
        seeds=args.seeds,
    )
    top = fi.head(args.top)[['feature']].copy()
    top['%'] = 100.0 * fi.head(args.top)['importance'].to_numpy()
    print(f'\nTop {args.top} features (mean % importance over {n_evaluations} fits):')
    print(top.to_string(index=False, formatters={'%': '{:.2f}'.format}))


if __name__ == '__main__':
    main()
