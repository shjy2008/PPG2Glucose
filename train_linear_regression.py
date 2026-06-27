import argparse

from sklearn.linear_model import LinearRegression

import data_loader
import utils


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='southwest', help='southwest or kaggle')
    p.add_argument('--n-splits', type=int, default=5)
    return p


def build_model(_seed: int):
    return LinearRegression()


def main():
    args = build_parser().parse_args()
    ds = data_loader.load_dataset(args.dataset)
    print(f"Using dataset: {args.dataset}")
    X, y, _ = data_loader.split(ds)
    print(f"Data for training - X.size: {X.shape[0]}, y.size: {y.shape[0]}")

    utils.run_repeated_kfold_training(
        X,
        y,
        build_model,
        feature_names=None,
        model_name='linear',
        n_splits=args.n_splits,
    )


if __name__ == '__main__':
    main()
