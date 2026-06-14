"""Fair repeated k-fold comparison for tabular PPG → glucose models."""

import argparse

import data_loader
import utils
from train_catboost import build_model as build_catboost
from train_lightGBM import build_model as build_lgbm
from train_random_forest import build_model as build_rf
from train_random_tree import build_model as build_randomtree
from train_svm import build_model as build_svm
from train_xgboost import build_model as build_xgb

AVAILABLE_MODELS = (
    'catboost',
    'xgb',
    'rf',
    'svm',
    'lgbm',
    'randomtree',
)

def build_parser():
    p = argparse.ArgumentParser(
        description=(
            'Compare tabular ML models with repeated k-fold CV. '
            'All models share the same folds within each seed.'
        ),
    )
    p.add_argument('--dataset', default='southwest', help='southwest, kaggle, or dataset path')
    p.add_argument('--n-splits', type=int, default=5, help='Number of CV folds')
    p.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=list(utils.DEFAULT_SEEDS),
        help='Random seeds for fold shuffling (default: 42 52 62 72 82)',
    )
    p.add_argument(
        '--models',
        nargs='+',
        choices=AVAILABLE_MODELS,
        default=['catboost', 'xgb', 'rf'],
        metavar='MODEL',
        help=f'Models to compare. Choices: {", ".join(AVAILABLE_MODELS)}',
    )
    p.add_argument(
        '--ensemble',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Include mean-prediction bagging ensemble across selected models (default: on)',
    )
    p.add_argument(
        '--output',
        default=None,
        help='Optional path to save the summary table as CSV',
    )
    return p


def model_factories(n_features: int, models: list[str]):
    all_factories = {
        'catboost': build_catboost,
        'xgb': build_xgb,
        'rf': build_rf,
        'svm': lambda _seed: build_svm(),
        'lgbm': build_lgbm,
        'randomtree': lambda seed: build_randomtree(seed, n_features),
    }
    return {name: all_factories[name] for name in models}


def main():
    args = build_parser().parse_args()
    ds = data_loader.load_dataset(args.dataset)
    print(f'Using dataset: {args.dataset}')
    X, y, _ = data_loader.split(ds)
    print(f'Data for evaluation - X.size: {X.shape[0]}, y.size: {y.shape[0]}')
    print(f'Models: {", ".join(args.models)}')

    _, summary_df = utils.run_repeated_kfold_comparison(
        X,
        y,
        factories=model_factories(X.shape[1], args.models),
        n_splits=args.n_splits,
        seeds=args.seeds,
        include_ensemble=args.ensemble,
    )

    if args.output:
        summary_df.to_csv(args.output, index=False)
        print(f'\nSaved summary to {args.output}')


if __name__ == '__main__':
    main()
