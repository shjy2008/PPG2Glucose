import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import data_loader
import evaluation
import utils
from train_catboost import build_model as build_catboost
from train_lightGBM import build_model as build_lgbm
from train_random_forest import build_model as build_rf
from train_random_tree import build_model as build_randomtree
from train_svm import build_model as build_svm
from train_xgboost import build_model as build_xgb

AVAILABLE_METHODS = (
    'catboost',
    'rf',
    'xgb',
    'lgbm',
    'randomtree',
    'svm',
    'linear',
)


class AverageEnsemble:
    def __init__(self, models):
        self.models = list(models)

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.asarray([model.predict(X) for model in self.models], dtype=float)
        return predictions.mean(axis=0)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='southwest', help='southwest or kaggle')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument(
        '--methods',
        nargs='+',
        choices=AVAILABLE_METHODS,
        default=['catboost', 'rf', 'xgb'],
        metavar='METHOD',
        help=(
            'Models to average in the ensemble. '
            f'Choices: {", ".join(AVAILABLE_METHODS)}. '
            'Default: catboost rf xgb'
        ),
    )
    return p


def model_factories(seed: int, n_features: int, methods: list[str]):
    all_factories = {
        'catboost': lambda: build_catboost(seed),
        'rf': lambda: build_rf(seed),
        'xgb': lambda: build_xgb(seed),
        'lgbm': lambda: build_lgbm(seed),
        'randomtree': lambda: build_randomtree(seed, n_features),
        'svm': build_svm,
        'linear': lambda: LinearRegression(),
    }
    return {name: all_factories[name] for name in methods}


def run_kfold_average_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    factories: dict[str, callable],
    n_splits: int,
    seed: int,
):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X, y = utils._prepare_xy(X, y)

    if X.isna().any().any():
        nan_columns = X.columns[X.isna().any()].tolist()
        print(
            f"Warning: imputing missing values in {len(nan_columns)} feature(s): "
            f"{', '.join(nan_columns)}"
        )

    evaluation.print_target_stats(y.values)

    model_names = list(factories.keys())
    oof_predictions = {name: np.empty(len(y), dtype=float) for name in model_names}
    oof_predictions['ensemble'] = np.empty(len(y), dtype=float)

    fold_metrics = {name: [] for name in [*model_names, 'ensemble']}

    for fold_index, (train_index, val_index) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_val = X.iloc[val_index]
        y_val = y.iloc[val_index]

        X_train, X_val, _ = utils._impute_features(X_train, X_val)

        fold_preds = []
        for name in model_names:
            model = factories[name]()
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            oof_predictions[name][val_index] = predictions
            fold_preds.append(predictions)

            metrics = evaluation.evaluate_regression(y_val.values, predictions)
            fold_metrics[name].append(metrics)

        ensemble_pred = np.mean(fold_preds, axis=0)
        oof_predictions['ensemble'][val_index] = ensemble_pred
        ensemble_metrics = evaluation.evaluate_regression(y_val.values, ensemble_pred)
        fold_metrics['ensemble'].append(ensemble_metrics)

        print(f"\nFold {fold_index}:")
        for name in model_names:
            metrics = fold_metrics[name][-1]
            print(
                f"  {name:9s} RMSE={metrics['rmse']:.4f} mmol/L "
                f"MAE={metrics['mae']:.4f} mmol/L R2={metrics['r2']:.4f}"
            )
        print(
            f"  {'ensemble':9s} RMSE={ensemble_metrics['rmse']:.4f} mmol/L "
            f"MAE={ensemble_metrics['mae']:.4f} mmol/L R2={ensemble_metrics['r2']:.4f}"
        )

    print('\nCross-validation summary:')
    for name in [*model_names, 'ensemble']:
        print(f"\n{name}:")
        for metric_name in ('rmse', 'mae', 'r2'):
            values = np.asarray(
                [metrics[metric_name] for metrics in fold_metrics[name]],
                dtype=float,
            )
            suffix = ' mmol/L' if metric_name in ('rmse', 'mae') else ''
            print(f"  {metric_name.upper()}{suffix}: {values.mean():.4f} ± {values.std(ddof=1):.4f}")

    print('\nOut-of-fold ensemble report:')
    evaluation.print_regression_report(y.values, oof_predictions['ensemble'])

    X_imputed, _, _ = utils._impute_features(X, X)
    final_models = [factories[name]() for name in model_names]
    final_ensemble = AverageEnsemble(final_models)
    final_ensemble.fit(X_imputed, y)
    return final_ensemble, fold_metrics, oof_predictions


def main():
    args = build_parser().parse_args()
    ds = data_loader.load_dataset(args.dataset)
    print(f"Using dataset: {args.dataset}")
    X, y, _ = data_loader.split(ds)
    print(f"Data for training - X.size: {X.shape[0]}, y.size: {y.shape[0]}")
    print(f"Ensemble methods: {', '.join(args.methods)}")

    run_kfold_average_ensemble(
        X,
        y,
        factories=model_factories(args.seed, X.shape[1], args.methods),
        n_splits=args.n_splits,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
