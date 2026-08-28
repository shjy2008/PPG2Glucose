import argparse

from sklearn.ensemble import RandomForestRegressor

import data_loader
import utils


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='IPS', help='IPS or MUST')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--top', type=int, default=10, help='Number of top features to print')
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

    summary_df, fi = utils.run_repeated_kfold_training(
        X,
        y,
        build_model,
        feature_names,
        model_name='rf',
        n_splits=args.n_splits,
    )
    n_evaluations = int(summary_df['n_evaluations'].iloc[0])
    top = fi.head(args.top)[['feature']].copy()
    top['%'] = 100.0 * fi.head(args.top)['importance'].to_numpy()
    print(f'\nTop {args.top} features (mean % importance over {n_evaluations} fits):')
    print(top.to_string(index=False, formatters={'%': '{:.2f}'.format}))


if __name__ == '__main__':
    main()
