import argparse

import numpy as np
from sklearn.tree import DecisionTreeRegressor

import data_loader
import evaluation
import utils


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='southwest', help='southwest or kaggle')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
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

    final_model, _, _ = utils.run_kfold_regression(
        X,
        y,
        model_factory=lambda: build_model(args.seed, X.shape[1]),
        n_splits=args.n_splits,
        seed=args.seed,
    )

    fi = evaluation.feature_importances_df(final_model, feature_names)
    if fi is not None:
        print('\nTop features:')
        print(fi.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
