"""Global runtime settings."""

USE_GPU = False


def catboost_task_type() -> str:
    return 'GPU' if USE_GPU else 'CPU'


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
