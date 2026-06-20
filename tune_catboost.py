"""Optuna hyperparameter search for CatBoost."""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

import data_loader
import evaluation
import utils
from train_catboost import DEFAULT_PARAMS, build_model

SEEDS = list(utils.DEFAULT_SEEDS)
RESULTS_PATH = Path('results/catboost_best_params.json')


def build_parser():
    p = argparse.ArgumentParser(description='Tune CatBoost hyperparameters with Optuna.')
    p.add_argument('--dataset', default='southwest', help='southwest, kaggle, or dataset path')
    p.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials')
    p.add_argument('--n-splits', type=int, default=5, help='CV folds per trial')
    return p


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        'iterations': trial.suggest_int('iterations', 200, 800, step=50),
        'learning_rate': trial.suggest_categorical(
            'learning_rate', [0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15]
        ),
        'depth': trial.suggest_int('depth', 2, 6),
        'subsample': trial.suggest_categorical(
            'subsample', [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        ),
        'reg_lambda': trial.suggest_categorical('reg_lambda', [0.1, 0.15, 0.2, 0.5, 1.0, 2.0, 5.0]),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 15),
        'border_count': trial.suggest_int('border_count', 32, 224, step=32),
    }


def cv_mae(X, y, params: dict, n_splits: int) -> float:
    scores = []
    for seed in SEEDS:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_index, val_index in cv.split(X, y):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_val = X.iloc[val_index]
            y_val = y.iloc[val_index]
            X_train, X_val, _ = utils._impute_features(X_train, X_val)

            model = build_model(seed, params)
            model.fit(X_train, y_train)
            metrics = evaluation.evaluate_regression(y_val.values, model.predict(X_val))
            scores.append(metrics['mae'])
    return float(np.mean(scores))


def main():
    args = build_parser().parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ds = data_loader.load_dataset(args.dataset)
    X, y, _ = data_loader.split(ds)
    X, y = utils._prepare_xy(X, y)

    print(f'Dataset: {args.dataset} ({X.shape[0]} samples)')
    print(f'Tuning: {args.n_trials} trials, {args.n_splits}-fold x {len(SEEDS)} seeds')
    print(f'Default params: {DEFAULT_PARAMS}')

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEEDS[0]),
    )
    study.optimize(
        lambda trial: cv_mae(X, y, suggest_params(trial), args.n_splits),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best_params = dict(study.best_params)
    print(f'\nBest CV MAE: {study.best_value:.4f}')
    print(f'Best params: {best_params}')

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({'params': best_params}, indent=2) + '\n')
    print(f'Saved to {RESULTS_PATH}')

    print('\nComparing default vs tuned...')
    default_summary, _ = utils.run_repeated_kfold_training(
        X, y, X.columns.tolist(),
        model_factory=lambda seed: build_model(seed),
        model_name='catboost_default',
        n_splits=args.n_splits,
        seeds=SEEDS,
        verbose=False,
    )
    tuned_summary, _ = utils.run_repeated_kfold_training(
        X, y, X.columns.tolist(),
        model_factory=lambda seed: build_model(seed, best_params),
        model_name='catboost_tuned',
        n_splits=args.n_splits,
        seeds=SEEDS,
        verbose=False,
    )
    utils.print_comparison_table(pd.concat([default_summary, tuned_summary], ignore_index=True))


if __name__ == '__main__':
    main()
