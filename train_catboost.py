import argparse

from catboost import CatBoostRegressor

import data_loader
import utils
from config import catboost_kwargs


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='southwest', help='southwest or kaggle')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=list(utils.DEFAULT_SEEDS),
        help='Random seeds for fold shuffling (default: 42 52 62 72 82)',
    )
    p.add_argument('--top', type=int, default=10, help='Number of top features to print')
    return p


def build_model(seed: int):
    return CatBoostRegressor(
        loss_function='MAE',#'RMSE',
        iterations=400,
        learning_rate=0.05,
        depth=3,
        subsample=0.9,
        reg_lambda=1.0,
        random_seed=seed,
        verbose=False,
        **catboost_kwargs(),
    )


def main():
    args = build_parser().parse_args()
    ds = data_loader.load_dataset(args.dataset)
    print(f"Using dataset: {args.dataset}")
    X, y, feature_names = data_loader.split(ds)
    n_evaluations = args.n_splits * len(args.seeds)
    print(f"Data for training - X.size: {X.shape[0]}, y.size: {y.shape[0]}")

    _, fi = utils.run_repeated_kfold_training(
        X,
        y,
        feature_names,
        model_factory=build_model,
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
