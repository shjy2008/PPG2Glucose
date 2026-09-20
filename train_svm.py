import argparse

from sklearn.svm import SVR

import data_loader
import utils


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='IPS', help='IPS or MUST')
    p.add_argument('--n-splits', type=int, default=5)
    return p


def build_model(_seed: int):
    # Feature z-score is applied in utils._impute_features (fit on train fold).
    return SVR(C=10.0, gamma='scale', epsilon=0.1, kernel='rbf')


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
        model_name='svm',
        n_splits=args.n_splits,
    )


if __name__ == '__main__':
    main()
