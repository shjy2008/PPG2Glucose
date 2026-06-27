import argparse

import numpy as np
from sklearn.tree import DecisionTreeRegressor

import data_loader
import utils


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='southwest', help='southwest or kaggle')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--top', type=int, default=10, help='Number of top features to print')
    return p


def build_model(seed: int, n_features: int):
    # WEKA RandomTree default: int(log2(#predictors) + 1) features per split.
    max_features = max(1, int(np.log2(n_features) + 1))
    return DecisionTreeRegressor(
        max_features=max_features,
        min_samples_leaf=2,
        min_samples_split=2,
        max_depth=None,
        random_state=seed,
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
        lambda seed: build_model(seed, X.shape[1]),
        feature_names,
        model_name='randomtree',
        n_splits=args.n_splits,
    )
    n_evaluations = int(summary_df['n_evaluations'].iloc[0])
    top = fi.head(args.top)[['feature']].copy()
    top['%'] = 100.0 * fi.head(args.top)['importance'].to_numpy()
    print(f'\nTop {args.top} features (mean % importance over {n_evaluations} fits):')
    print(top.to_string(index=False, formatters={'%': '{:.2f}'.format}))


if __name__ == '__main__':
    main()
