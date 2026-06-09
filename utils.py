import numpy as np
import pandas as pd

from typing import Callable, Optional, Sequence, Tuple

from sklearn.impute import SimpleImputer
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