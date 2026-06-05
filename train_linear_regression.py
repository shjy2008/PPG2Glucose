import argparse
from pathlib import Path

from sklearn.linear_model import LinearRegression

import data_loader
import utils


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='southwest', help='southwest or kaggle')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    ds = data_loader.load_dataset(args.dataset)
    print(f"Using dataset: {args.dataset}")
    X, y = data_loader.get_X_y(ds)
    print(f"Data for training - X.size: {X.shape[0]}, y.size: {y.shape[0]}")

    utils.run_kfold_regression(
        X,
        y,
        model_factory=lambda: LinearRegression(),
        n_splits=args.n_splits,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
