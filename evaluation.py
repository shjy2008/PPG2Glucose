import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def target_stats(y: np.ndarray) -> dict:
    """Summary statistics for glucose targets in mmol/L."""
    values = np.asarray(y, dtype=float).reshape(-1)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {'n': 0, 'mean': np.nan, 'variance': np.nan, 'std': np.nan}
    if values.size == 1:
        return {
            'n': 1,
            'mean': float(values[0]),
            'variance': 0.0,
            'std': 0.0,
        }
    return {
        'n': int(values.size),
        'mean': float(np.mean(values)),
        'variance': float(np.var(values, ddof=1)),
        'std': float(np.std(values, ddof=1)),
    }


def print_target_stats(y: np.ndarray) -> None:
    stats = target_stats(y)
    print('\n---Target distribution (mmol/L)---')
    print(f"n:        {stats['n']}")
    print(f"mean:     {stats['mean']:.4f}")
    print(f"variance: {stats['variance']:.4f} (mmol/L)²")
    print(f"std:      {stats['std']:.4f}")
    print(f"(Mean predictor baseline RMSE ≈ {stats['std']:.4f} mmol/L)")
    print('----------------------------------\n')


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def print_regression_report(y_true: np.ndarray, y_pred: np.ndarray, preview_n: int = 10) -> None:
    metrics = evaluate_regression(y_true, y_pred)
    unit = 'mmol/L'
    print('\nEvaluation:')
    print(f"RMSE ({unit}): {metrics['rmse']:.4f}")
    print(f"MAE ({unit}):  {metrics['mae']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")

    preview = pd.DataFrame({
        f'actual ({unit})': np.asarray(y_true).reshape(-1),
        f'predicted ({unit})': np.asarray(y_pred).reshape(-1),
    })
    print('\nPredictions preview:')
    print(preview.head(preview_n).to_string(index=False))


def summarize_metric_values(values: np.ndarray) -> dict:
    """Return mean and sample std for a 1-D array of metric scores."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return {'mean': np.nan, 'std': np.nan, 'n': 0}
    if arr.size == 1:
        return {'mean': float(arr[0]), 'std': 0.0, 'n': 1}
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)),
        'n': int(arr.size),
    }


def format_metric_summary(mean: float, std: float, decimals: int = 4) -> str:
    return f'{mean:.{decimals}f} ± {std:.{decimals}f}'


def summarize_repeated_cv_metrics(
    fold_metrics: list[dict],
    metric_names: tuple[str, ...] = ('rmse', 'mae', 'r2'),
) -> dict:
    """Aggregate per-fold metric dicts into mean/std summaries."""
    return {
        metric_name: summarize_metric_values(
            [metrics[metric_name] for metrics in fold_metrics]
        )
        for metric_name in metric_names
    }


def feature_importances_df(model, feature_names):
    # Works for XGBoost and sklearn estimators that expose feature_importances_
    estimator = model
    if hasattr(estimator, 'named_steps'):
        # prefer explicit 'model' step, otherwise take last step
        named_steps = getattr(estimator, 'named_steps')
        estimator = named_steps.get('model', list(named_steps.values())[-1])
    if not hasattr(estimator, 'feature_importances_'):
        return None
    importances = estimator.feature_importances_
    return pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
