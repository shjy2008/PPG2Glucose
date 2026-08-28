import argparse

from lightgbm import LGBMRegressor

import data_loader
import utils
from config import lightgbm_kwargs


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='IPS', help='IPS or MUST')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--top', type=int, default=10, help='Number of top features to print')
    return p


def build_model(seed: int):
    return LGBMRegressor(
        objective='regression',
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
        **lightgbm_kwargs(),
    )


def main():
    args = build_parser().parse_args()
    ds = data_loader.load_dataset(args.dataset)
    print(f"Using dataset: {args.dataset}")
    X, y, feature_names = data_loader.split(ds)
    print(f"Data for training - X.size: {X.shape[0]}, y.size: {y.shape[0]}")

    summary_df, fi = utils.run_repeated_kfold_training(
        X,
        y,
        build_model,
        feature_names,
        model_name='lgbm',
        n_splits=args.n_splits,
    )
    n_evaluations = int(summary_df['n_evaluations'].iloc[0])
    top = fi.head(args.top)[['feature']].copy()
    top['%'] = 100.0 * fi.head(args.top)['importance'].to_numpy()
    print(f'\nTop {args.top} features (mean % importance over {n_evaluations} fits):')
    print(top.to_string(index=False, formatters={'%': '{:.2f}'.format}))


if __name__ == '__main__':
    main()
