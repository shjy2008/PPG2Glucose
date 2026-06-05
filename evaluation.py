import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
