import numpy as np
import pandas as pd

from typing import Callable, Sequence, Tuple
from sklearn.model_selection import KFold
import evaluation

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


def run_kfold_regression(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], object],
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[object, Sequence[dict], np.ndarray]:
    """Run k-fold CV for a regression estimator.

    - Drops rows with missing values before CV (minimal approach).
    - Fits model_factory() per-fold on training rows and predicts validation rows.
    - Prints per-fold and aggregate metrics and returns final model refit on full data.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # drop any rows with missing values for simple, minimal pipeline
    mask = X.notna().all(axis=1) & y.notna()
    if not mask.all():
        dropped = int((~mask).sum())
        print(f"Warning: dropping {dropped} rows with missing values before CV")
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    fold_metrics = []
    oof_predictions = np.empty(len(y), dtype=float)

    for fold_index, (train_index, val_index) in enumerate(cv.split(X, y), start=1):
        model = model_factory()
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_val = X.iloc[val_index]
        y_val = y.iloc[val_index]

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

    final_model = model_factory()
    final_model.fit(X, y)
    return final_model, fold_metrics, oof_predictions