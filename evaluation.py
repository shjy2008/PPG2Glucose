from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MMOL_L_TO_MG_DL = 18.0
CLARKE_AXIS_MAX_MG_DL = 400.0
CLARKE_ZONE_LABELS = ('A', 'B', 'C', 'D', 'E')
CLARKE_ZONE_COLORS = {
    'A': '#C6EFCE',
    'B': '#FFEB9C',
    'C': '#FFC7CE',
    'D': '#FF8C8C',
    'E': '#C00000',
}

CLARKE_PLOT_PATH = 'clarke_error_grid.png'


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


def _to_mg_dl(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1) * MMOL_L_TO_MG_DL


def clarke_zone_label(ref_mg_dl: float, pred_mg_dl: float) -> str:
    """Assign a single Clarke Error Grid zone (reference and prediction in mg/dL)."""
    if (
        (ref_mg_dl <= 70 and pred_mg_dl <= 70)
        or (0.8 * ref_mg_dl <= pred_mg_dl <= 1.2 * ref_mg_dl)
    ):
        return 'A'
    if (ref_mg_dl >= 180 and pred_mg_dl <= 70) or (ref_mg_dl <= 70 and pred_mg_dl >= 180):
        return 'E'
    if (
        (70 <= ref_mg_dl <= 290 and pred_mg_dl >= ref_mg_dl + 110)
        or (130 <= ref_mg_dl <= 180 and pred_mg_dl <= (7 / 5) * ref_mg_dl - 182)
    ):
        return 'C'
    if (
        (ref_mg_dl >= 240 and 70 <= pred_mg_dl <= 180)
        or (ref_mg_dl <= 175 / 3 and 70 <= pred_mg_dl <= 180)
        or (175 / 3 <= ref_mg_dl <= 70 and pred_mg_dl >= (6 / 5) * ref_mg_dl)
    ):
        return 'D'
    return 'B'


def clarke_zones(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return Clarke zones for paired glucose values in mmol/L."""
    ref_mg_dl = _to_mg_dl(y_true)
    pred_mg_dl = _to_mg_dl(y_pred)
    return np.array(
        [clarke_zone_label(ref, pred) for ref, pred in zip(ref_mg_dl, pred_mg_dl)],
        dtype=object,
    )


def clarke_zone_fractions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the fraction of points in each Clarke zone."""
    zones = clarke_zones(y_true, y_pred)
    n = zones.size
    if n == 0:
        return {f'zone_{label.lower()}': np.nan for label in CLARKE_ZONE_LABELS}
    return {
        f'zone_{label.lower()}': float(np.mean(zones == label))
        for label in CLARKE_ZONE_LABELS
    }


def plot_clarke_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = 'Clarke Error Grid',
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Scatter plot with Clarke Error Grid zones (axes in mmol/L)."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    axis_max_mmol = CLARKE_AXIS_MAX_MG_DL / MMOL_L_TO_MG_DL

    mesh = np.linspace(0.0, CLARKE_AXIS_MAX_MG_DL, 401)
    zone_grid = np.empty((mesh.size, mesh.size), dtype=object)
    for row_index, pred_value in enumerate(mesh):
        for col_index, ref_value in enumerate(mesh):
            zone_grid[row_index, col_index] = clarke_zone_label(ref_value, pred_value)

    color_grid = np.zeros((mesh.size, mesh.size, 3), dtype=float)
    for label, color in CLARKE_ZONE_COLORS.items():
        mask = zone_grid == label
        rgb = np.array(plt.matplotlib.colors.to_rgb(color))
        color_grid[mask] = rgb

    fig, ax = plt.subplots(figsize=(7, 7))
    extent = (0.0, axis_max_mmol, 0.0, axis_max_mmol)
    ax.imshow(
        color_grid,
        origin='lower',
        extent=extent,
        aspect='equal',
        interpolation='nearest',
        zorder=0,
    )

    ref_mmol = y_true
    pred_mmol = y_pred
    ax.scatter(ref_mmol, pred_mmol, s=24, c='black', alpha=0.75, zorder=3)

    scale = 1.0 / MMOL_L_TO_MG_DL
    boundary_lines = [
        ([0, 175 / 3], [70, 70]),
        ([175 / 3, CLARKE_AXIS_MAX_MG_DL / 1.2], [70, CLARKE_AXIS_MAX_MG_DL]),
        ([70, 70], [84, CLARKE_AXIS_MAX_MG_DL]),
        ([0, 70], [180, 180]),
        ([70, 290], [180, CLARKE_AXIS_MAX_MG_DL]),
        ([70, 70], [0, 56]),
        ([70, CLARKE_AXIS_MAX_MG_DL], [56, 320]),
        ([180, 180], [0, 70]),
        ([180, CLARKE_AXIS_MAX_MG_DL], [70, 70]),
        ([240, 240], [70, 180]),
        ([240, CLARKE_AXIS_MAX_MG_DL], [180, 180]),
        ([130, 180], [0, 70]),
    ]
    for x_values, y_values in boundary_lines:
        ax.plot(
            np.asarray(x_values) * scale,
            np.asarray(y_values) * scale,
            color='black',
            linewidth=0.8,
            zorder=2,
        )

    ax.plot([0, axis_max_mmol], [0, axis_max_mmol], ':', color='black', linewidth=1.0, zorder=2)
    ax.set_xlim(0.0, axis_max_mmol)
    ax.set_ylim(0.0, axis_max_mmol)
    ax.set_xlabel('Reference glucose (mmol/L)')
    ax.set_ylabel('Predicted glucose (mmol/L)')
    ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def glucose_prediction_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ACC fraction: |error| < 0.8 if y <= 5.55 mmol/L, else relative error <= 15%."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = np.abs(y_pred - y_true)
    ok = np.where(y_true <= 5.55, err < 0.8, err / y_true <= 0.15)
    return float(np.mean(ok)) if ok.size else np.nan


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    zone_fractions = clarke_zone_fractions(y_true, y_pred)
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'acc': glucose_prediction_acc(y_true, y_pred),
        'zone_a': zone_fractions['zone_a'],
        'zone_b': zone_fractions['zone_b'],
    }


def print_clarke_plot(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save Clarke Error Grid scatter plot (metrics come from CV summary)."""
    plot_clarke_error_grid(y_true, y_pred, save_path=CLARKE_PLOT_PATH)
    print(f'\nClarke Error Grid plot saved to {CLARKE_PLOT_PATH}')


def print_regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    preview_n: int = 10,
) -> None:
    metrics = evaluate_regression(y_true, y_pred)
    unit = 'mmol/L'
    print('\nEvaluation:')
    print(f"RMSE ({unit}): {metrics['rmse']:.4f}")
    print(f"MAE ({unit}):  {metrics['mae']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"ACC (%): {100.0 * metrics['acc']:.2f}")
    print('Clarke Error Grid:')
    print(f"Zone A (%): {100.0 * metrics['zone_a']:.2f}")
    print(f"Zone B (%): {100.0 * metrics['zone_b']:.2f}")
    print(
        f"Zone A+B (%): {100.0 * (metrics['zone_a'] + metrics['zone_b']):.2f}"
    )
    print_clarke_plot(y_true, y_pred)

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
    metric_names: tuple[str, ...] = (
        'rmse',
        'mae',
        'r2',
        'acc',
        'zone_a',
        'zone_b',
    ),
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
