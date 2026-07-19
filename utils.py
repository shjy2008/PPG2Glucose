import numpy as np
import pandas as pd

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold

import evaluation
from config import DEFAULT_SEEDS, TARGET_OUTLIER_LOWER_Q, TARGET_OUTLIER_UPPER_Q

MG_DL_TO_MMOL_L = 1.0 / 18.0


def mg_dl_to_mmol_l(value: float) -> float:
    return float(value) * MG_DL_TO_MMOL_L


def safe_float(value):
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _target_outlier_mask(y):
    values = np.asarray(y, dtype=float)
    lower, upper = np.quantile(values, [TARGET_OUTLIER_LOWER_Q, TARGET_OUTLIER_UPPER_Q])
    return (values >= lower) & (values <= upper), lower, upper


def _log_outlier_exclusion(n_removed: int, lower: float, upper: float) -> None:
    if n_removed:
        print(
            f'Excluded {n_removed} target outlier(s) outside '
            f'[{lower:.4f}, {upper:.4f}] mmol/L'
        )


def exclude_target_outliers(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    mask, lower, upper = _target_outlier_mask(y)
    _log_outlier_exclusion(int((~mask).sum()), lower, upper)
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def exclude_target_outlier_arrays(y: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mask, lower, upper = _target_outlier_mask(y)
    _log_outlier_exclusion(int((~mask).sum()), lower, upper)
    return (y[mask], *(array[mask] for array in arrays))


def _prepare_xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = y.notna()
    if not mask.all():
        dropped = int((~mask).sum())
        print(f"Warning: dropping {dropped} rows with missing target before CV")
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    return X, y


def _split_impute_tabular(
    X: pd.DataFrame,
    train_index: np.ndarray,
    val_index: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train, X_val, _ = _impute_features(X.iloc[train_index], X.iloc[val_index])
    return X_train, X_val


def _impute_features(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    imputer: Optional[SimpleImputer] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    if imputer is None:
        imputer = SimpleImputer(strategy='median')
        train_values = imputer.fit_transform(X_train)
    else:
        train_values = imputer.transform(X_train)

    X_train_imputed = pd.DataFrame(train_values, columns=X_train.columns, index=X_train.index)
    other_values = imputer.transform(X_other)
    X_other_imputed = pd.DataFrame(other_values, columns=X_other.columns, index=X_other.index)
    return X_train_imputed, X_other_imputed, imputer


def _normalized_feature_importances(model) -> Optional[np.ndarray]:
    estimator = model
    if hasattr(estimator, 'named_steps'):
        named_steps = getattr(estimator, 'named_steps')
        estimator = named_steps.get('model', list(named_steps.values())[-1])
    if not hasattr(estimator, 'feature_importances_'):
        return None
    importances = np.asarray(estimator.feature_importances_, dtype=float)
    total = importances.sum()
    if total > 0:
        importances = importances / total
    return importances


def _warn_nan_features(X: pd.DataFrame, verbose: bool) -> None:
    if verbose and X.isna().any().any():
        nan_columns = X.columns[X.isna().any()].tolist()
        print(
            f"Warning: imputing missing values in {len(nan_columns)} feature(s): "
            f"{', '.join(nan_columns)}"
        )


def _iter_cv_splits(
    y: np.ndarray,
    n_splits: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
):
    """Yield (train_index, val_index). With groups, keep subjects out of both folds."""
    if groups is None:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from cv.split(y)
        return

    groups = np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError(f'groups length ({len(groups)}) must match y ({len(y)})')
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(
            f'Need at least {n_splits} unique subjects for GroupKFold, found {n_groups}'
        )

    # GroupKFold has no shuffle; remap group ids with a seeded permutation.
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    remap = {group: index for index, group in enumerate(shuffled)}
    remapped = np.asarray([remap[group] for group in groups], dtype=int)
    cv = GroupKFold(n_splits=n_splits)
    yield from cv.split(y, groups=remapped)


def _run_repeated_kfold_cv(
    y: np.ndarray,
    run_fold: Callable[[np.ndarray, np.ndarray, int], Mapping[str, np.ndarray]],
    model_names: Sequence[str],
    *,
    n_splits: int = 5,
    seeds: Optional[Sequence[int]] = None,
    groups: Optional[np.ndarray] = None,
    verbose: bool = True,
    fold_log: Optional[Callable[[Dict[str, dict]], str]] = None,
) -> Tuple[Dict[str, list[dict]], Dict[str, np.ndarray]]:
    y = np.asarray(y, dtype=float).reshape(-1)
    if seeds is None:
        seeds = DEFAULT_SEEDS

    fold_metrics = {name: [] for name in model_names}
    oof_sum = {name: np.zeros(len(y), dtype=float) for name in model_names}
    oof_count = {name: np.zeros(len(y), dtype=float) for name in model_names}

    for seed_index, seed in enumerate(seeds, start=1):
        if verbose:
            mode = 'GroupKFold (subject-wise)' if groups is not None else 'KFold'
            print(f'--- Seed {seed} ({seed_index}/{len(seeds)}) [{mode}] ---')

        for fold_index, (train_index, val_index) in enumerate(
            _iter_cv_splits(y, n_splits, seed, groups=groups),
            start=1,
        ):
            predictions_by_model = run_fold(train_index, val_index, seed)
            metrics_by_model = {}
            for name in model_names:
                predictions = predictions_by_model[name]
                metrics = evaluation.evaluate_regression(y[val_index], predictions)
                fold_metrics[name].append(metrics)
                oof_sum[name][val_index] += predictions
                oof_count[name][val_index] += 1.0
                metrics_by_model[name] = metrics

            if verbose:
                metrics = next(iter(metrics_by_model.values()))
                message = (
                    fold_log(metrics_by_model)
                    if fold_log is not None
                    else (
                        f'RMSE={metrics["rmse"]:.4f} mmol/L '
                        f'MAE={metrics["mae"]:.4f} mmol/L R2={metrics["r2"]:.4f}'
                    )
                )
                print(f'  Fold {fold_index}: {message}')

    oof_by_model = {name: oof_sum[name] / oof_count[name] for name in model_names}
    return fold_metrics, oof_by_model


def run_repeated_kfold_array_cv(
    y: np.ndarray,
    predict_fold: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    *,
    model_name: str = 'model',
    n_splits: int = 5,
    seeds: Optional[Sequence[int]] = None,
    print_target_distribution: bool = True,
    verbose: bool = True,
) -> Tuple[list[dict], pd.DataFrame, np.ndarray]:
    y = np.asarray(y, dtype=float).reshape(-1)
    if seeds is None:
        seeds = DEFAULT_SEEDS

    if verbose and print_target_distribution:
        evaluation.print_cv_intro(y, n_splits, seeds)

    fold_metrics_by_model, oof_by_model = _run_repeated_kfold_cv(
        y,
        lambda train_index, val_index, seed: {
            model_name: predict_fold(train_index, val_index, seed),
        },
        [model_name],
        n_splits=n_splits,
        seeds=seeds,
        verbose=verbose,
    )
    fold_metrics_by_model, summary_df, oof_predictions = evaluation.report_cv(
        fold_metrics_by_model,
        oof_by_model,
        y,
        plot_model=model_name,
        verbose=verbose,
    )
    return fold_metrics_by_model[model_name], summary_df, oof_predictions


def run_repeated_kfold_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    factories: Mapping[str, Callable[[int], object]],
    n_splits: int = 5,
    include_ensemble: bool = False,
    ensemble_name: str = 'ensemble',
    ensemble_weights: Optional[Sequence[float]] = None,
    groups: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, list[dict]], pd.DataFrame]:
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError(f'groups length ({len(groups)}) must match y ({len(y)})')
        X = X.replace([np.inf, -np.inf], np.nan)
        mask = y.notna().to_numpy()
        if not mask.all():
            dropped = int((~mask).sum())
            print(f"Warning: dropping {dropped} rows with missing target before CV")
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
        groups = groups[mask]
    else:
        X, y = _prepare_xy(X, y)
    seeds = DEFAULT_SEEDS
    model_names = list(factories.keys())
    use_ensemble = include_ensemble and len(model_names) > 1
    tracked_names = [*model_names, ensemble_name] if use_ensemble else model_names
    y_values = y.values

    weights = None
    if use_ensemble:
        if ensemble_weights is None:
            weights = np.full(len(model_names), 1.0 / len(model_names), dtype=float)
        else:
            weights = np.asarray(ensemble_weights, dtype=float)
            if weights.shape != (len(model_names),):
                raise ValueError(
                    f'ensemble_weights length ({len(weights)}) must match '
                    f'number of models ({len(model_names)})'
                )
            if np.any(weights < 0):
                raise ValueError('ensemble_weights must be non-negative')
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                raise ValueError('ensemble_weights must sum to a positive value')
            weights = weights / weight_sum

    _warn_nan_features(X, verbose)
    if verbose:
        extras = []
        if groups is not None:
            extras.append(
                f'Split: GroupKFold by subject_ID '
                f'({len(np.unique(groups))} unique subjects)'
            )
        if use_ensemble:
            weight_desc = ', '.join(
                f'{name}={weight:.3f}' for name, weight in zip(model_names, weights)
            )
            extras.append(f'Ensemble: weighted mean ({weight_desc})')
        evaluation.print_cv_intro(
            y_values,
            n_splits,
            seeds,
            per_model=True,
            models=model_names,
            extra_lines=extras,
        )

    def run_fold(train_index: np.ndarray, val_index: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
        X_train, X_val = _split_impute_tabular(X, train_index, val_index)
        y_train = y.iloc[train_index]

        predictions_by_model: Dict[str, np.ndarray] = {}
        fold_predictions = []
        for name in model_names:
            model = factories[name](seed)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            fold_predictions.append(predictions)
            predictions_by_model[name] = predictions

        if use_ensemble:
            stacked = np.column_stack(fold_predictions)
            predictions_by_model[ensemble_name] = stacked @ weights
        return predictions_by_model

    fold_metrics_by_model, oof_by_model = _run_repeated_kfold_cv(
        y_values,
        run_fold,
        tracked_names,
        n_splits=n_splits,
        seeds=seeds,
        groups=groups,
        verbose=verbose,
        fold_log=lambda metrics_by_model: ' | '.join(
            f'{name} MAE={metrics["mae"]:.4f}' for name, metrics in metrics_by_model.items()
        ),
    )
    _, summary_df, _ = evaluation.report_cv(
        fold_metrics_by_model,
        oof_by_model,
        y_values,
        verbose=verbose,
    )
    return fold_metrics_by_model, summary_df


def run_repeated_kfold_training(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[int], object],
    feature_names: Optional[Sequence[str]] = None,
    model_name: str = 'model',
    n_splits: int = 5,
    select_top_n: Optional[int] = None,
    groups: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError(f'groups length ({len(groups)}) must match y ({len(y)})')
        X = X.replace([np.inf, -np.inf], np.nan)
        mask = y.notna().to_numpy()
        if not mask.all():
            dropped = int((~mask).sum())
            print(f"Warning: dropping {dropped} rows with missing target before CV")
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
        groups = groups[mask]
    else:
        X, y = _prepare_xy(X, y)
    seeds = DEFAULT_SEEDS
    y_values = y.values
    track_importances = feature_names is not None and len(feature_names) > 0
    use_selection = (
        select_top_n is not None
        and track_importances
        and select_top_n < X.shape[1]
    )

    accum = np.zeros(len(feature_names), dtype=float) if track_importances else None
    selection_counts = np.zeros(len(feature_names), dtype=int) if track_importances else None
    importance_runs = 0
    n_runs = 0

    _warn_nan_features(X, verbose)
    if verbose:
        extras = []
        if groups is not None:
            extras.append(
                f'Split: GroupKFold by subject_ID '
                f'({len(np.unique(groups))} unique subjects)'
            )
        if use_selection:
            extras.append(f'Feature selection: top {select_top_n} per fold (train-only ranking)')
        evaluation.print_cv_intro(y_values, n_splits, seeds, extra_lines=extras)

    def run_fold(train_index: np.ndarray, val_index: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
        nonlocal accum, selection_counts, importance_runs, n_runs

        X_train, X_val = _split_impute_tabular(X, train_index, val_index)
        y_train = y.iloc[train_index]

        if use_selection:
            selector = model_factory(seed)
            selector.fit(X_train, y_train)
            importances = np.asarray(selector.feature_importances_, dtype=float)
            selected_idx = np.argsort(importances)[::-1][: min(select_top_n, X_train.shape[1])]
            selected_columns = X_train.columns[selected_idx]
            selection_counts[selected_idx] += 1

            model = model_factory(seed)
            model.fit(X_train[selected_columns], y_train)
            predictions = model.predict(X_val[selected_columns])
            fold_importances = _normalized_feature_importances(model)
            importances = np.zeros(len(feature_names), dtype=float)
            if fold_importances is not None:
                importances[selected_idx] = fold_importances
        else:
            model = model_factory(seed)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            importances = _normalized_feature_importances(model)

        if track_importances and importances is not None:
            accum += importances
            importance_runs += 1
        n_runs += 1
        return {model_name: predictions}

    fold_metrics_by_model, oof_by_model = _run_repeated_kfold_cv(
        y_values,
        run_fold,
        [model_name],
        n_splits=n_splits,
        seeds=seeds,
        groups=groups,
        verbose=verbose,
    )
    _, summary_df, oof_predictions = evaluation.report_cv(
        fold_metrics_by_model,
        oof_by_model,
        y_values,
        plot_model=model_name,
        verbose=False,
    )

    if track_importances and importance_runs > 0:
        importance_df = pd.DataFrame({
            'feature': list(feature_names),
            'importance': accum / importance_runs,
        }).sort_values('importance', ascending=False).reset_index(drop=True)
    else:
        importance_df = pd.DataFrame(columns=['feature', 'importance'])

    if verbose:
        if use_selection:
            freq = pd.DataFrame({
                'feature': list(feature_names),
                'selected_in_folds': selection_counts,
            }).sort_values(
                ['selected_in_folds', 'feature'],
                ascending=[False, True],
            ).reset_index(drop=True)
            print(
                f'\nMost frequently selected features '
                f'(top {select_top_n} per fold, {n_runs} fits):'
            )
            print(
                freq.head(select_top_n)[['feature', 'selected_in_folds']].to_string(
                    index=False,
                )
            )
        evaluation.print_cv_results(summary_df, y_values, oof_predictions)

    return summary_df, importance_df
