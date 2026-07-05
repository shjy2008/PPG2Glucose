"""Global runtime settings."""

USE_GPU = False

# Edit this list for tabular ML, CNN, LSTM, etc.
BASIC_FEATURE_KEYS = ('age',)
# BASIC_FEATURE_KEYS = ('age', 'bmi')
# BASIC_FEATURE_KEYS = ('age', 'height', 'weight')
# BASIC_FEATURE_KEYS = ('sex', 'age', 'height', 'weight')
# BASIC_FEATURE_KEYS = ('age', 'height', 'weight', 'bmi')

# Shared default for repeated k-fold CV (5-fold x 5 seeds = 25 evaluations).
DEFAULT_SEEDS = (42, 52, 62, 72, 82)

# PPG sampling rate (Hz) used for frequency-domain and heart-rate features.
PPG_SAMPLING_RATE = {
    'kaggle': 2175,
    'southwest': 200,
}
DEFAULT_PPG_SAMPLING_RATE = 200


def ppg_sampling_rate(dataset: str) -> float:
    return PPG_SAMPLING_RATE.get(dataset, DEFAULT_PPG_SAMPLING_RATE)

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
