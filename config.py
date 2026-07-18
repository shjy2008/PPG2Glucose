"""Global runtime settings."""

USE_GPU = False

# Per-dataset demographic features for tabular ML, CNN, LSTM, etc.
# Options: (), ('age',), ('age', 'bmi'), ('age', 'height', 'weight'),
# ('sex', 'age', 'height', 'weight'), ('age', 'height', 'weight', 'bmi')
BASIC_FEATURE_KEYS = {
    'southwest': ('age', 'height', 'weight', 'bmi'),
    'MUST': (),
    'vitaldb': (),
}
DEFAULT_BASIC_FEATURE_KEYS = ()

# Shared default for repeated k-fold CV (5-fold x 5 seeds = 25 evaluations).
DEFAULT_SEEDS = (42, 52, 62, 72, 82)

# PPG sampling rate (Hz) used for frequency-domain and heart-rate features.
PPG_SAMPLING_RATE = {
    'MUST': 2175,
    'southwest': 200,
    'vitaldb': 500,
}
DEFAULT_PPG_SAMPLING_RATE = 200

# 0 = full signal (one sample). >0 = non-overlapping windows in seconds.
DEFAULT_WINDOW_SECONDS = 0.0

# Fractions trimmed from each dataset's PPG signals before feature extraction.
PPG_TRIM_FRACTIONS = {
    'MUST': (0.0, 0.0),
    'southwest': (0.1, 0.0),
    'vitaldb': (0.0, 0.0),
}
DEFAULT_PPG_TRIM_FRACTIONS = (0.0, 0.0)


def basic_feature_keys(dataset: str) -> tuple:
    return BASIC_FEATURE_KEYS.get(dataset, DEFAULT_BASIC_FEATURE_KEYS)


def ppg_sampling_rate(dataset: str) -> float:
    return PPG_SAMPLING_RATE.get(dataset, DEFAULT_PPG_SAMPLING_RATE)


def ppg_trim_fractions(dataset: str) -> tuple[float, float]:
    return PPG_TRIM_FRACTIONS.get(dataset, DEFAULT_PPG_TRIM_FRACTIONS)

# Drop glucose targets outside this quantile range (mmol/L).
TARGET_OUTLIER_LOWER_Q = 0
TARGET_OUTLIER_UPPER_Q = 1
# TARGET_OUTLIER_LOWER_Q = 0.01
# TARGET_OUTLIER_UPPER_Q = 0.99

def catboost_kwargs() -> dict:
    if USE_GPU:
        # GPU uses Bayesian bootstrap by default, which rejects subsample.
        return {'task_type': 'GPU', 'bootstrap_type': 'Bernoulli'}
    return {'task_type': 'CPU'}


def xgboost_kwargs() -> dict:
    if USE_GPU:
        return {'tree_method': 'hist', 'device': 'cuda'}
    return {'tree_method': 'hist', 'device': 'cpu', 'n_jobs': -1}


def xgboost_train_params() -> dict:
    if USE_GPU:
        return {'tree_method': 'hist', 'device': 'cuda'}
    return {'tree_method': 'hist', 'device': 'cpu', 'nthread': -1}


def lightgbm_kwargs() -> dict:
    if USE_GPU:
        return {'device': 'gpu'}
    return {'device': 'cpu', 'n_jobs': -1}


def torch_device():
    import torch

    if USE_GPU and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def torch_float_tensor(array):
    import torch

    return torch.tensor(array, dtype=torch.float32)


def tensor_to_numpy(tensor):
    import numpy as np

    return np.asarray(tensor.detach().cpu().tolist(), dtype=np.float32)
