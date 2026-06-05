import argparse

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import data_loader
import utils


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='southwest', help='southwest or kaggle')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    return p


def build_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(C=10.0, gamma='scale', epsilon=0.1, kernel='rbf')),
    ])


def main():
    args = build_parser().parse_args()
    ds = data_loader.load_dataset(args.dataset)
    print(f"Using dataset: {args.dataset}")
    X, y, _ = data_loader.split(ds)
    print(f"Data for training - X.size: {X.shape[0]}, y.size: {y.shape[0]}")

    utils.run_kfold_regression(
        X,
        y,
        model_factory=build_model,
        n_splits=args.n_splits,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()