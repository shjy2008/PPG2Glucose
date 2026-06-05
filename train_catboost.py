import argparse

from catboost import CatBoostRegressor

import data_loader
import evaluation
import utils


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='southwest', help='southwest or kaggle')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    return p


def build_model(seed: int):
    return CatBoostRegressor(
        loss_function='RMSE',
        iterations=400,
        learning_rate=0.05,
        depth=4,
        subsample=0.9,
        reg_lambda=1.0,
        random_seed=seed,
        verbose=False,
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
        model_factory=lambda: build_model(args.seed),
        n_splits=args.n_splits,
        seed=args.seed,
    )

    fi = evaluation.feature_importances_df(final_model, feature_names)
    if fi is not None:
        print('\nTop features:')
        print(fi.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
