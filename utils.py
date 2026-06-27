import numpy as np
import pandas as pd

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

import evaluation
from config import DEFAULT_SEEDS, TARGET_OUTLIER_LOWER_Q, TARGET_OUTLIER_UPPER_Q

# Labels store glucose in mg/dL; convert to mmol/L for modeling and reporting.
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
    mask = (values >= lower) & (values <= upper)
    return mask, lower, upper


def exclude_target_outliers(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    mask, lower, upper = _target_outlier_mask(y)
    n_removed = int((~mask).sum())
    if n_removed:
        print(
            f'Excluded {n_removed} target outlier(s) outside '
            f'[{lower:.4f}, {upper:.4f}] mmol/L'
        )
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def exclude_target_outlier_arrays(y: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mask, lower, upper = _target_outlier_mask(y)
    n_removed = int((~mask).sum())
    if n_removed:
        print(
            f'Excluded {n_removed} target outlier(s) outside '
            f'[{lower:.4f}, {upper:.4f}] mmol/L'
        )
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


def run_kfold_regression(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], object],
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[object, Sequence[dict], np.ndarray]:
    """Run k-fold CV for a regression estimator.

    - Drops rows with missing targets before CV.
    - Median-imputes missing feature values per fold (fit on train only).
    - Fits model_factory() per-fold on training rows and predicts validation rows.
    - Prints per-fold and aggregate metrics and returns final model refit on full data.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X, y = _prepare_xy(X, y)

    if X.isna().any().any():
        nan_columns = X.columns[X.isna().any()].tolist()
        print(
            f"Warning: imputing missing values in {len(nan_columns)} feature(s): "
            f"{', '.join(nan_columns)}"
        )

    evaluation.print_target_stats(y.values)

    fold_metrics = []
    oof_predictions = np.empty(len(y), dtype=float)

    for fold_index, (train_index, val_index) in enumerate(cv.split(X, y), start=1):
        model = model_factory()
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_val = X.iloc[val_index]
        y_val = y.iloc[val_index]

        X_train, X_val, _ = _impute_features(X_train, X_val)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        oof_predictions[val_index] = predictions

        metrics = evaluation.evaluate_regression(y_val.values, predictions)
        fold_metrics.append(metrics)
        print(
            f"Fold {fold_index}: RMSE={metrics['rmse']:.4f} mmol/L "
            f"MAE={metrics['mae']:.4f} mmol/L R2={metrics['r2']:.4f}"
        )

    print('\nCross-validation summary:')
    for metric_name in ('rmse', 'mae', 'r2'):
        values = np.asarray([metrics[metric_name] for metrics in fold_metrics], dtype=float)
        suffix = ' mmol/L' if metric_name in ('rmse', 'mae') else ''
        print(f"{metric_name.upper()}{suffix}: {values.mean():.4f} ± {values.std(ddof=1):.4f}")

    evaluation.print_regression_report(y.values, oof_predictions)

    X_imputed, _, imputer = _impute_features(X, X)
    final_model = model_factory()
    final_model.fit(X_imputed, y)
    return final_model, fold_metrics, oof_predictions


def run_repeated_kfold_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    factories: Mapping[str, Callable[[int], object]],
    n_splits: int = 5,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    include_ensemble: bool = False,
    ensemble_name: str = 'ensemble',
    verbose: bool = True,
) -> Tuple[Dict[str, list[dict]], pd.DataFrame]:
    """Compare multiple regressors with repeated k-fold cross-validation.

    Fairness guarantees:
    - Within each seed, every model uses the exact same train/validation splits.
    - Missing values are median-imputed per fold using training data only.
    - Stochastic models should consume the current seed via their factory.

    Returns:
        fold_metrics_by_model: per-model list of metric dicts (one per fold × seed)
        summary_df: mean ± std of per-fold scores
    """
    X, y = _prepare_xy(X, y)
    model_names = list(factories.keys())
    use_ensemble = include_ensemble and len(model_names) > 1
    tracked_names = [*model_names, ensemble_name] if use_ensemble else model_names

    if X.isna().any().any() and verbose:
        nan_columns = X.columns[X.isna().any()].tolist()
        print(
            f"Warning: imputing missing values in {len(nan_columns)} feature(s): "
            f"{', '.join(nan_columns)}"
        )

    if verbose:
        evaluation.print_target_stats(y.values)
        total_runs = len(seeds) * n_splits
        print(
            f'\nRepeated CV: {n_splits}-fold × {len(seeds)} seeds '
            f'= {total_runs} evaluations per model'
        )
        print(f'Models: {", ".join(model_names)}')
        if use_ensemble:
            print(f'Ensemble: mean of {", ".join(model_names)}')
        print(f'Seeds: {list(seeds)}\n')

    fold_metrics_by_model: Dict[str, list[dict]] = {name: [] for name in tracked_names}

    for seed_index, seed in enumerate(seeds, start=1):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if verbose:
            print(f'--- Seed {seed} ({seed_index}/{len(seeds)}) ---')

        for fold_index, (train_index, val_index) in enumerate(cv.split(X, y), start=1):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_val = X.iloc[val_index]
            y_val = y.iloc[val_index]
            X_train, X_val, _ = _impute_features(X_train, X_val)

            fold_line = []
            fold_predictions = []
            for name in model_names:
                model = factories[name](seed)
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                fold_predictions.append(predictions)

                metrics = evaluation.evaluate_regression(y_val.values, predictions)
                fold_metrics_by_model[name].append(metrics)
                fold_line.append(f'{name} MAE={metrics["mae"]:.4f}')

            if use_ensemble:
                ensemble_predictions = np.mean(fold_predictions, axis=0)
                ensemble_metrics = evaluation.evaluate_regression(
                    y_val.values,
                    ensemble_predictions,
                )
                fold_metrics_by_model[ensemble_name].append(ensemble_metrics)
                fold_line.append(f'{ensemble_name} MAE={ensemble_metrics["mae"]:.4f}')

            if verbose:
                print(f'  Fold {fold_index}: {" | ".join(fold_line)}')

    summary_rows = []
    for name in tracked_names:
        summary = evaluation.summarize_repeated_cv_metrics(fold_metrics_by_model[name])
        summary_rows.append({
            'model': name,
            'n_evaluations': summary['mae']['n'],
            'mae_mean': summary['mae']['mean'],
            'mae_std': summary['mae']['std'],
            'rmse_mean': summary['rmse']['mean'],
            'rmse_std': summary['rmse']['std'],
            'r2_mean': summary['r2']['mean'],
            'r2_std': summary['r2']['std'],
            'acc_mean': 100.0 * summary['acc']['mean'],
            'acc_std': 100.0 * summary['acc']['std'],
            'zone_a_mean': 100.0 * summary['zone_a']['mean'],
            'zone_a_std': 100.0 * summary['zone_a']['std'],
            'zone_b_mean': 100.0 * summary['zone_b']['mean'],
            'zone_b_std': 100.0 * summary['zone_b']['std'],
        })

    summary_df = pd.DataFrame(summary_rows).reset_index(drop=True)

    if verbose:
        print('\n=== Repeated cross-validation summary (mean ± std) ===')
        print_comparison_table(summary_df)

    return fold_metrics_by_model, summary_df


def _top_feature_indices(model, n_features: int, top_n: int) -> np.ndarray:
    importances = np.asarray(model.feature_importances_, dtype=float)
    ranked = np.argsort(importances)[::-1]
    return ranked[: min(top_n, n_features)]


def run_repeated_kfold_training(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Sequence[str],
    model_factory: Callable[[int], object],
    model_name: str = 'model',
    n_splits: int = 5,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    select_top_n: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated k-fold CV for one model: metrics and feature importances.

    Uses the same fold splits, imputation, and seed handling as
    run_repeated_kfold_comparison. Each fold fit contributes one validation
    score and one normalized feature-importance vector.

    When select_top_n is set (and less than the feature count), feature ranking
    is fit on each training fold only, then the model is refit on those columns.
    """
    X, y = _prepare_xy(X, y)
    accum = np.zeros(len(feature_names), dtype=float)
    selection_counts = np.zeros(len(feature_names), dtype=int)
    fold_metrics: list[dict] = []
    oof_sum = np.zeros(len(y), dtype=float)
    oof_count = np.zeros(len(y), dtype=float)
    n_runs = 0
    use_selection = select_top_n is not None and select_top_n < X.shape[1]

    if X.isna().any().any() and verbose:
        nan_columns = X.columns[X.isna().any()].tolist()
        print(
            f"Warning: imputing missing values in {len(nan_columns)} feature(s): "
            f"{', '.join(nan_columns)}"
        )

    if verbose:
        evaluation.print_target_stats(y.values)
        total_runs = len(seeds) * n_splits
        print(
            f'\nRepeated CV: {n_splits}-fold × {len(seeds)} seeds '
            f'= {total_runs} evaluations'
        )
        if use_selection:
            print(f'Feature selection: top {select_top_n} per fold (train-only ranking)')
        print(f'Seeds: {list(seeds)}\n')

    for seed_index, seed in enumerate(seeds, start=1):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if verbose:
            print(f'--- Seed {seed} ({seed_index}/{len(seeds)}) ---')

        for fold_index, (train_index, val_index) in enumerate(cv.split(X, y), start=1):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_val = X.iloc[val_index]
            y_val = y.iloc[val_index]
            X_train, X_val, _ = _impute_features(X_train, X_val)

            if use_selection:
                selector = model_factory(seed)
                selector.fit(X_train, y_train)
                selected_idx = _top_feature_indices(selector, X_train.shape[1], select_top_n)
                selected_columns = X_train.columns[selected_idx]
                selection_counts[selected_idx] += 1

                model = model_factory(seed)
                model.fit(X_train[selected_columns], y_train)
                predictions = model.predict(X_val[selected_columns])
                importances = np.zeros(len(feature_names), dtype=float)
                fold_importances = np.asarray(model.feature_importances_, dtype=float)
                total = fold_importances.sum()
                if total > 0:
                    fold_importances = fold_importances / total
                importances[selected_idx] = fold_importances
            else:
                model = model_factory(seed)
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                importances = np.asarray(model.feature_importances_, dtype=float)
                total = importances.sum()
                if total > 0:
                    importances = importances / total

            metrics = evaluation.evaluate_regression(y_val.values, predictions)
            fold_metrics.append(metrics)
            accum += importances
            oof_sum[val_index] += predictions
            oof_count[val_index] += 1.0
            n_runs += 1

            if verbose:
                print(
                    f'  Fold {fold_index}: RMSE={metrics["rmse"]:.4f} mmol/L '
                    f'MAE={metrics["mae"]:.4f} mmol/L R2={metrics["r2"]:.4f}'
                )

    summary = evaluation.summarize_repeated_cv_metrics(fold_metrics)
    summary_df = pd.DataFrame([{
        'model': model_name,
        'n_evaluations': summary['mae']['n'],
        'mae_mean': summary['mae']['mean'],
        'mae_std': summary['mae']['std'],
        'rmse_mean': summary['rmse']['mean'],
        'rmse_std': summary['rmse']['std'],
        'r2_mean': summary['r2']['mean'],
        'r2_std': summary['r2']['std'],
        'acc_mean': 100.0 * summary['acc']['mean'],
        'acc_std': 100.0 * summary['acc']['std'],
        'zone_a_mean': 100.0 * summary['zone_a']['mean'],
        'zone_a_std': 100.0 * summary['zone_a']['std'],
        'zone_b_mean': 100.0 * summary['zone_b']['mean'],
        'zone_b_std': 100.0 * summary['zone_b']['std'],
    }])

    mean_importance = accum / n_runs
    importance_df = pd.DataFrame({
        'feature': list(feature_names),
        'importance': mean_importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    if verbose:
        print('\n=== Repeated cross-validation summary (mean ± std) ===')
        print_comparison_table(summary_df)
        if use_selection:
            freq = pd.DataFrame({
                'feature': list(feature_names),
                'selected_in_folds': selection_counts,
            })
            freq = freq.sort_values(
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

        oof_predictions = oof_sum / oof_count
        evaluation.print_clarke_plot(y.values, oof_predictions)

    return summary_df, importance_df


def aggregate_feature_importances(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Sequence[str],
    model_factory: Callable[[int], object],
    n_splits: int = 5,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> pd.DataFrame:
    """Mean normalized feature importances over repeated k-fold fits."""
    _, importance_df = run_repeated_kfold_training(
        X,
        y,
        feature_names,
        model_factory,
        n_splits=n_splits,
        seeds=seeds,
        verbose=False,
    )
    return importance_df


def print_comparison_table(summary_df: pd.DataFrame) -> None:
    """Pretty-print a repeated-CV comparison table."""
    rows = []
    for _, row in summary_df.iterrows():
        rows.append({
            'Model': row['model'],
            'MAE (mmol/L)': evaluation.format_metric_summary(row['mae_mean'], row['mae_std']),
            'RMSE (mmol/L)': evaluation.format_metric_summary(row['rmse_mean'], row['rmse_std']),
            'R2': evaluation.format_metric_summary(row['r2_mean'], row['r2_std']),
            'ACC (%)': evaluation.format_metric_summary(row['acc_mean'], row['acc_std'], decimals=2),
            'Zone A (%)': evaluation.format_metric_summary(
                row['zone_a_mean'], row['zone_a_std'], decimals=2
            ),
            'Zone B (%)': evaluation.format_metric_summary(
                row['zone_b_mean'], row['zone_b_std'], decimals=2
            ),
            'N': int(row['n_evaluations']),
        })
    print(pd.DataFrame(rows).to_string(index=False))