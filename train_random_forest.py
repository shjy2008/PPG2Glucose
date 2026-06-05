import argparse

from sklearn.ensemble import RandomForestRegressor

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
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=seed,
        n_jobs=-1,
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

    if hasattr(final_model, 'oob_score_'):
        print(f"\nOOB R2: {final_model.oob_score_:.4f}")

    fi = evaluation.feature_importances_df(final_model, feature_names)
    if fi is not None:
        print('\nTop features:')
        print(fi.head(10).to_string(index=False))


if __name__ == '__main__':
    main()