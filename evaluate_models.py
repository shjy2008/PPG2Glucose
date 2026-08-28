"""Fair repeated k-fold comparison for tabular PPG → glucose models."""

import argparse

import numpy as np
from sklearn.linear_model import LinearRegression

import data_loader
import utils
from config import DEFAULT_WINDOW_SECONDS
from train_catboost import build_model as build_catboost
from train_catboost import load_params as load_catboost_params
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
    'linear',
)


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            'Compare tabular ML models with repeated k-fold CV. '
            'All models share the same folds within each seed.'
        ),
    )
    p.add_argument('--dataset', default='IPS', help='IPS, MUST, or dataset path')
    p.add_argument('--n-splits', type=int, default=5, help='Number of CV folds')
    p.add_argument(
        '--window-seconds',
        type=float,
        default=None,
        help='Non-overlapping PPG window length in seconds (default: config.DEFAULT_WINDOW_SECONDS; 0=full)',
    )
    p.add_argument(
        '--subject-wise',
        action='store_true',
        help='Use GroupKFold so the same subject_ID never appears in both train and val '
             '(auto-enabled when any subject has multiple samples, e.g. MUST or windowed PPG)',
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
        '--ensemble-weights',
        nargs='+',
        type=float,
        default=None,
        metavar='W',
        help=(
            'Ensemble weights matching --models order '
            '(e.g. --models catboost xgb rf --ensemble-weights 0.6 0.2 0.2). '
            'Default: equal weights. Weights are normalized to sum to 1.'
        ),
    )
    p.add_argument(
        '--output',
        default=None,
        help='Optional path to save the summary table as CSV',
    )
    return p


def model_factories(n_features: int, models: list[str], dataset: str):
    catboost_params = load_catboost_params(dataset)
    all_factories = {
        'catboost': lambda seed: build_catboost(seed, catboost_params),
        'xgb': build_xgb,
        'rf': build_rf,
        'svm': build_svm,
        'lgbm': build_lgbm,
        'randomtree': lambda seed: build_randomtree(seed, n_features),
        'linear': lambda _seed: LinearRegression(),
    }
    return {name: all_factories[name] for name in models}


def main():
    args = build_parser().parse_args()
    window_seconds = (
        DEFAULT_WINDOW_SECONDS if args.window_seconds is None else args.window_seconds
    )
    ds = data_loader.load_dataset(args.dataset, window_seconds=window_seconds)
    multi_sample_subjects = (
        'subject_ID' in ds.columns and ds.groupby('subject_ID').size().max() > 1
    )
    subject_wise = args.subject_wise or multi_sample_subjects
    print(f'Using dataset: {args.dataset} (window_seconds={window_seconds})')
    if subject_wise:
        X, y, _, groups = data_loader.split(ds, return_groups=True)
        if groups is None:
            raise RuntimeError('subject-wise CV requires subject_ID in the dataset')
        print(
            f'Subject-wise CV: {len(np.unique(groups))} unique subjects '
            f'across {len(y)} samples'
        )
    else:
        X, y, _ = data_loader.split(ds)
        groups = None
    print(f'Data for evaluation - X.size: {X.shape[0]}, y.size: {y.shape[0]}')
    print(f'Models: {", ".join(args.models)}')
    if args.ensemble_weights is not None:
        if len(args.ensemble_weights) != len(args.models):
            raise SystemExit(
                f'--ensemble-weights has {len(args.ensemble_weights)} value(s) '
                f'but --models has {len(args.models)} model(s)'
            )
        if not args.ensemble:
            raise SystemExit('--ensemble-weights requires --ensemble')

    _, summary_df = utils.run_repeated_kfold_comparison(
        X,
        y,
        factories=model_factories(X.shape[1], args.models, args.dataset),
        n_splits=args.n_splits,
        include_ensemble=args.ensemble,
        ensemble_weights=args.ensemble_weights,
        groups=groups,
    )

    if args.output:
        summary_df.to_csv(args.output, index=False)
        print(f'\nSaved summary to {args.output}')


if __name__ == '__main__':
    main()
