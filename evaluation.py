from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import matplotlib

matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
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
SEG_PLOT_PATH = 'seg_error_grid.png'
SEG_GRID_IMAGE = Path(__file__).resolve().parent / 'evaluation_data' / 'seg_grid.png'
SEG_IMAGE_MAX_MG_DL = 600.0
SEG_AXIS_MAX_MG_DL = 400.0


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


def _add_mg_dl_axes(ax: plt.Axes) -> None:
    """Add mg/dL labels on the top (reference) and right (predicted) axes."""
    mmol_to_mg = lambda values: values * MMOL_L_TO_MG_DL
    mg_to_mmol = lambda values: values / MMOL_L_TO_MG_DL

    ax_top = ax.secondary_xaxis('top', functions=(mmol_to_mg, mg_to_mmol))
    ax_top.set_xlabel('Reference glucose (mg/dL)')

    ax_right = ax.secondary_yaxis('right', functions=(mmol_to_mg, mg_to_mmol))
    ax_right.set_ylabel('Predicted glucose (mg/dL)')


def plot_clarke_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = 'Clarke Error Grid',
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Scatter plot with Clarke Error Grid zones (bottom/left mmol/L, top/right mg/dL)."""
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
    _add_mg_dl_axes(ax)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _seg_colormap() -> mcolors.LinearSegmentedColormap:
    colors = [
        (0, 165 / 256, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 0, 0),
        (128 / 256, 0, 0),
    ]
    nodes = [0.0, 0.4375, 1.0625, 2.75, 4.0]
    return mcolors.LinearSegmentedColormap.from_list(
        'seg',
        list(zip([node / 4.0 for node in nodes], colors)),
    )


def plot_seg_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = 'Surveillance Error Grid',
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Scatter plot on the Surveillance Error Grid (bottom/left mmol/L, top/right mg/dL)."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    axis_max_mmol = SEG_AXIS_MAX_MG_DL / MMOL_L_TO_MG_DL
    image_max_mmol = SEG_IMAGE_MAX_MG_DL / MMOL_L_TO_MG_DL

    fig, ax = plt.subplots(figsize=(7.5, 7))
    grid = np.flipud(plt.imread(SEG_GRID_IMAGE))
    image = ax.imshow(
        grid,
        origin='lower',
        extent=(0.0, image_max_mmol, 0.0, image_max_mmol),
        cmap=_seg_colormap(),
        vmin=0.0,
        vmax=4.0,
        aspect='equal',
        zorder=0,
    )
    ax.scatter(
        y_true,
        y_pred,
        s=28,
        facecolors='white',
        edgecolors='black',
        linewidths=0.6,
        alpha=0.9,
        zorder=3,
    )
    ax.plot([0, axis_max_mmol], [0, axis_max_mmol], ':', color='black', linewidth=1.0, zorder=2)
    ax.set_xlim(0.0, axis_max_mmol)
    ax.set_ylim(0.0, axis_max_mmol)
    ax.set_xlabel('Reference glucose (mmol/L)')
    ax.set_ylabel('Predicted glucose (mmol/L)')
    ax.set_title(title)
    _add_mg_dl_axes(ax)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Risk level')
    cbar.set_ticks([0.25, 1.0, 2.0, 3.0, 3.75])
    cbar.set_ticklabels(['None', 'Slight', 'Moderate', 'High', 'Extreme'])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def glucose_prediction_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ACC fraction: |error| < 0.83 if y <= 5.55 mmol/L, else relative error <= 15%."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = np.abs(y_pred - y_true)
    ok = np.where(y_true <= 5.55, err < 0.83, err / y_true <= 0.15)
    return float(np.mean(ok)) if ok.size else np.nan


def glucose_mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Relative Difference (%)."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = y_true > 0
    if not np.any(mask):
        return np.nan
    return float(100.0 * np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])))


def format_pearson_p(p_value: float) -> str:
    if np.isnan(p_value):
        return 'p=nan'
    if p_value < 0.001:
        return 'p<0.001'
    return f'p={p_value:.4g}'


def format_pearson_summary(r: float, p_value: float) -> str:
    return f'{r:.4f} ({format_pearson_p(p_value)})'


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    if y_true.size < 2:
        pearson_r, pearson_p = np.nan, np.nan
    else:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        pearson_r, pearson_p = float(pearson_r), float(pearson_p)
    zone_fractions = clarke_zone_fractions(y_true, y_pred)
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'acc': glucose_prediction_acc(y_true, y_pred),
        'mard': glucose_mard(y_true, y_pred),
        'zone_a': zone_fractions['zone_a'],
        'zone_b': zone_fractions['zone_b'],
        'zone_ab': zone_fractions['zone_a'] + zone_fractions['zone_b'],
    }


def print_regression_report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = evaluate_regression(y_true, y_pred)
    print(f'\n{name}')
    print(f"  MAE:      {metrics['mae']:.4f} mmol/L")
    print(f"  RMSE:     {metrics['rmse']:.4f} mmol/L")
    print(f"  R2:       {metrics['r2']:.4f}")
    print(f"  ACC:      {100 * metrics['acc']:.2f}%")
    print(f"  Zone A:   {100 * metrics['zone_a']:.2f}%")
    print(f"  Zone B:   {100 * metrics['zone_b']:.2f}%")
    print(f"  Zone A+B: {100 * (metrics['zone_a'] + metrics['zone_b']):.2f}%")
    print(f"  MARD:     {metrics['mard']:.2f}%")
    print(f"  Pearson:  {metrics['pearson_r']:.4f}")
    return metrics


def print_clarke_plot(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save Clarke and SEG scatter plots (metrics come from CV summary)."""
    plot_clarke_error_grid(y_true, y_pred, save_path=CLARKE_PLOT_PATH)
    plot_seg_error_grid(y_true, y_pred, save_path=SEG_PLOT_PATH)
    print(f'\nClarke Error Grid plot saved to {CLARKE_PLOT_PATH}')
    print(f'Surveillance Error Grid plot saved to {SEG_PLOT_PATH}')


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
        'pearson_r',
        'acc',
        'mard',
        'zone_a',
        'zone_b',
        'zone_ab',
    ),
) -> dict:
    """Aggregate per-fold metric dicts into mean/std summaries."""
    return {
        metric_name: summarize_metric_values(
            [metrics[metric_name] for metrics in fold_metrics]
        )
        for metric_name in metric_names
    }


def cv_summary_row(
    fold_metrics: list[dict],
    y: np.ndarray,
    oof_predictions: np.ndarray,
    model_name: str,
) -> dict:
    summary = summarize_repeated_cv_metrics(fold_metrics)
    oof_metrics = evaluate_regression(y, oof_predictions)
    return {
        'model': model_name,
        'n_evaluations': summary['mae']['n'],
        'mae_mean': summary['mae']['mean'],
        'mae_std': summary['mae']['std'],
        'rmse_mean': summary['rmse']['mean'],
        'rmse_std': summary['rmse']['std'],
        'r2_mean': summary['r2']['mean'],
        'r2_std': summary['r2']['std'],
        'pearson_r_mean': summary['pearson_r']['mean'],
        'pearson_r_std': summary['pearson_r']['std'],
        'pearson_oof_r': oof_metrics['pearson_r'],
        'pearson_oof_p': oof_metrics['pearson_p'],
        'acc_mean': 100.0 * summary['acc']['mean'],
        'acc_std': 100.0 * summary['acc']['std'],
        'mard_mean': summary['mard']['mean'],
        'mard_std': summary['mard']['std'],
        'zone_a_mean': 100.0 * summary['zone_a']['mean'],
        'zone_a_std': 100.0 * summary['zone_a']['std'],
        'zone_b_mean': 100.0 * summary['zone_b']['mean'],
        'zone_b_std': 100.0 * summary['zone_b']['std'],
        'zone_ab_mean': 100.0 * summary['zone_ab']['mean'],
        'zone_ab_std': 100.0 * summary['zone_ab']['std'],
    }


def cv_summary_df(
    fold_metrics_by_model: Mapping[str, list[dict]],
    y: np.ndarray,
    oof_by_model: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    return pd.DataFrame([
        cv_summary_row(fold_metrics_by_model[name], y, oof_by_model[name], name)
        for name in fold_metrics_by_model
    ])


def print_cv_intro(
    y: np.ndarray,
    n_splits: int,
    seeds: Sequence[int],
    *,
    per_model: bool = False,
    models: Optional[Sequence[str]] = None,
    extra_lines: Sequence[str] = (),
) -> None:
    print_target_stats(y)
    total_runs = len(seeds) * n_splits
    scope = ' per model' if per_model else ''
    print(f'\nRepeated CV: {n_splits}-fold × {len(seeds)} seeds = {total_runs} evaluations{scope}')
    if models:
        print(f'Models: {", ".join(models)}')
    for line in extra_lines:
        print(line)
    print(f'Seeds: {list(seeds)}\n')


def report_cv(
    fold_metrics_by_model: Mapping[str, list[dict]],
    oof_by_model: Mapping[str, np.ndarray],
    y: np.ndarray,
    *,
    plot_model: Optional[str] = None,
    verbose: bool = True,
) -> tuple[Dict[str, list[dict]], pd.DataFrame, Optional[np.ndarray]]:
    summary_df = cv_summary_df(fold_metrics_by_model, y, oof_by_model)
    oof_predictions = oof_by_model[plot_model] if plot_model else None
    if verbose:
        print_cv_results(
            summary_df,
            y if oof_predictions is not None else None,
            oof_predictions,
        )
    return fold_metrics_by_model, summary_df, oof_predictions


def print_comparison_table(summary_df: pd.DataFrame) -> None:
    rows = []
    for _, row in summary_df.iterrows():
        rows.append({
            'Model': row['model'],
            'MAE (mmol/L)': format_metric_summary(row['mae_mean'], row['mae_std']),
            'RMSE (mmol/L)': format_metric_summary(row['rmse_mean'], row['rmse_std']),
            'R2': format_metric_summary(row['r2_mean'], row['r2_std']),
            'Pearson r': format_pearson_summary(row['pearson_oof_r'], row['pearson_oof_p']),
            'ACC (%)': format_metric_summary(row['acc_mean'], row['acc_std'], decimals=2),
            'MARD (%)': format_metric_summary(row['mard_mean'], row['mard_std'], decimals=2),
            'Zone A (%)': format_metric_summary(row['zone_a_mean'], row['zone_a_std'], decimals=2),
            'Zone B (%)': format_metric_summary(row['zone_b_mean'], row['zone_b_std'], decimals=2),
            'N': int(row['n_evaluations']),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def print_cv_metric_block(row: pd.Series) -> None:
    print(f"  MAE:      {format_metric_summary(row['mae_mean'], row['mae_std'])} mmol/L")
    print(f"  RMSE:     {format_metric_summary(row['rmse_mean'], row['rmse_std'])} mmol/L")
    print(f"  R2:       {format_metric_summary(row['r2_mean'], row['r2_std'])}")
    print(f"  ACC:      {format_metric_summary(row['acc_mean'], row['acc_std'], decimals=2)}%")
    print(f"  Zone A:   {format_metric_summary(row['zone_a_mean'], row['zone_a_std'], decimals=2)}%")
    print(f"  Zone B:   {format_metric_summary(row['zone_b_mean'], row['zone_b_std'], decimals=2)}%")
    print(f"  Zone A+B: {format_metric_summary(row['zone_ab_mean'], row['zone_ab_std'], decimals=2)}%")
    print(f"  MARD:     {format_metric_summary(row['mard_mean'], row['mard_std'], decimals=2)}%")
    print(f"  Pearson:  {format_pearson_summary(row['pearson_oof_r'], row['pearson_oof_p'])}")


def print_cv_results(
    summary_df: pd.DataFrame,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    preview_n: int = 10,
) -> None:
    n_evaluations = int(summary_df['n_evaluations'].iloc[0])
    label = 'per model' if len(summary_df) > 1 else ''
    suffix = f' {label}' if label else ''
    print(f'\n=== Cross-validation results (mean ± std, N={n_evaluations}{suffix}) ===')
    for _, row in summary_df.iterrows():
        if len(summary_df) > 1:
            print(f"\n{row['model']}")
        print_cv_metric_block(row)

    if y_true is None or y_pred is None:
        return

    print_clarke_plot(y_true, y_pred)
    preview = pd.DataFrame({
        'actual (mmol/L)': np.asarray(y_true).reshape(-1),
        'predicted (mmol/L)': np.asarray(y_pred).reshape(-1),
    })
    print('\nPredictions preview:')
    print(preview.head(preview_n).to_string(index=False))
