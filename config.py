"""Global runtime settings."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

# TinyML / CBHFF use CUDA when available. Set False to force CPU.
# CatBoost (and other tabular models) always use CPU.
USE_GPU = True

# Named dataset roots. Edit paths per machine, or override with env vars.
# --dataset vitaldb loads dataset/vitaldb (60 s windows from download_vitaldb.py).
_VITALDB_ROOT = Path(os.environ.get('VITALDB_DIR', 'dataset/vitaldb'))
DATASET_ROOTS = {
    'IPS': Path('dataset/IPS'),
    'MUST': Path('dataset/MUST'),
    'vitaldb': _VITALDB_ROOT,
    'Hb_PPG': Path('dataset/Hb_PPG_Dataset'),
}

# Config profile used for PPG/demographics when the CLI name is an alias/path.
DATASET_CONFIG_PROFILE = {
    'IPS': 'IPS',
    'MUST': 'MUST',
    'vitaldb': 'vitaldb',
    'Hb_PPG': 'Hb_PPG',
    'Hb_PPG_Dataset': 'Hb_PPG',
}

# Default --ppg-seconds when --dataset vitaldb and flag omitted. Other datasets: full signal.
DEFAULT_VITALDB_PPG_SECONDS = 60.0

# Per-dataset demographic features for tabular ML, CNN, LSTM, etc.
# Options: (), ('age',), ('age', 'bmi'), ('age', 'height', 'weight'),
# ('sex', 'age', 'height', 'weight'), ('age', 'height', 'weight', 'bmi')
BASIC_FEATURE_KEYS = {
    'IPS': ('age', 'height', 'weight', 'bmi'),
    # MUST glucose+demo baseline: age+weight only (depth=4 → MAE 0.7075, R2 +0.0003).
    'MUST': ('age', 'weight'),
    'vitaldb': ('age', 'height', 'weight', 'bmi'),
    # Height/weight missing for 77/217 glucose-labeled subjects (marked "/").
    'Hb_PPG': ('age', 'height', 'weight', 'bmi'),
}
DEFAULT_BASIC_FEATURE_KEYS = ()

# Joined at load for --label age|height|weight|bmi. Glucose model inputs still use
# BASIC_FEATURE_KEYS only (extra attached demos are dropped from X).
ATTACH_DEMO_KEYS = ('age', 'height', 'weight', 'bmi')

# Shared default for repeated k-fold CV (5-fold x 5 seeds = 25 evaluations).
DEFAULT_SEEDS = (42, 52, 62, 72, 82)

# PPG sampling rate (Hz) used for frequency-domain and heart-rate features.
PPG_SAMPLING_RATE = {
    'MUST': 2175,
    'IPS': 200,
    'vitaldb': 500,
    'Hb_PPG': 200,
}
DEFAULT_PPG_SAMPLING_RATE = 200

# Optional rate-based downsample after filter (Hz). None keeps native PPG_SAMPLING_RATE.
# Applied in shared PPG prep so feature extraction uses the reduced rate.
# MUST downsample-only 2175→100 Hz raised MAE 0.7311 → 0.7633.
PPG_TARGET_FS = {
    'MUST': None,  # 100.0
    'IPS': None,
    'vitaldb': None,
    'Hb_PPG': None,
}
DEFAULT_PPG_TARGET_FS = None

# Hand-crafted PPG feature groups enabled per dataset.
# g1=distribution, g2=derivatives, g3=pulse, g4=frequency,
# g5=periodicity, g6=signal segments, g7=optional signal-quality indicators.
DEFAULT_PPG_FEATURE_GROUPS = ('g1', 'g2', 'g3', 'g4', 'g5', 'g6')
PPG_FEATURE_GROUPS = {
    # MUST ablation: g3-g6 best among group sets; age+weight demo improves further.
    'MUST': (
        # 'g1',
        # 'g2',
        'g3',
        'g4',
        'g5',
        'g6',
        ),
    'IPS': DEFAULT_PPG_FEATURE_GROUPS,
    'vitaldb': DEFAULT_PPG_FEATURE_GROUPS,
    'Hb_PPG': DEFAULT_PPG_FEATURE_GROUPS,
}

# 0 = full signal (one sample). >0 = non-overlapping windows in seconds.
DEFAULT_WINDOW_SECONDS = 0.0

# Fractions trimmed from each dataset's PPG signals before feature extraction.
PPG_TRIM_FRACTIONS = {
    'MUST': (0.0, 0.0),
    'IPS': (0.1, 0.0),
    'vitaldb': (0.0, 0.0),
    'Hb_PPG': (0.0, 0.0),
}
DEFAULT_PPG_TRIM_FRACTIONS = (0.0, 0.0)

# PPG time-domain filter applied after trim, before feature extraction.
# Options:
#   'none'
#   'mswf' (Chen et al. JBHI 2024 multi-size weighted fitting)
#   'butterworth' / 'butterworth_0.5_8' (3rd-order band-pass 0.5–8 Hz; common PPG→BG)
#   'butterworth_0.5_5' (2nd/3rd-order lit. variant; we use order 3)
# Try-1 IPS CatBoost: mswf raised MAE 0.5785 → 0.6119, so default off.
# MUST CatBoost ablation (subject-wise 5x5 CV, baseline MAE 0.7311):
# target_fs=100 only 0.7633; mswf only 0.7964; combined 0.7950. Keep both off.
# Try-4 MUST: fs-aware physiological feature units (IBI/lag in seconds, cardiac
# bands 0.5–4 Hz, rate-normalized derivatives) → MAE 0.7468 (no gain). Keep fixes.
# Try-5 IPS CatBoost: butterworth 0.5–8 Hz raised MAE 0.5786 → 0.6047 (same
# direction as mswf). Band-pass strips DC and breaks DC-sensitive features.
# Keep filtering off for handcrafted-feature CatBoost.
PPG_FILTER = {
    'MUST': 'none',
    'IPS': 'none',
    'vitaldb': 'none',
    'Hb_PPG': 'none',
}
DEFAULT_PPG_FILTER = 'none'

# Per-channel amplitude normalization after trim/filter/downsample/crop.
# 'zscore' = (x - mean) / std (common in PPG DL papers).
# 'none' keeps raw amplitudes (needed for amplitude-sensitive handcrafted features).
# Try-6 CatBoost: zscore hurt — IPS MAE 0.578→0.608, MUST 0.731→0.749. Keep off.
PPG_NORMALIZE = {
    'MUST': 'none',
    'IPS': 'none',
    'vitaldb': 'none',
    'Hb_PPG': 'none',
}
DEFAULT_PPG_NORMALIZE = 'none'

# Optional best-window selection before feature extraction.
# A positive duration selects the overlapping window with the highest SQA score
# (see PPG_SQA_METHOD). MUST 4 s / 1 s hop raised CatBoost MAE 0.7311 → 0.7685.
PPG_BEST_SEGMENT_SECONDS = {
    'MUST': None,  # 4.0
    'IPS': None,
    'vitaldb': None,
    'Hb_PPG': None,
}
DEFAULT_PPG_BEST_SEGMENT_SECONDS = None
PPG_BEST_SEGMENT_HOP_SECONDS = 1.0

# Signal-quality gate before feature extraction. None disables hard filtering.
# Accept non-overlapping PPG_SQA_WINDOW_SECONDS windows with score > threshold,
# then concatenate for full-signal features.
# Methods:
#   'template' — mean cosine similarity of beats to mean-pulse template
#                (Orphanidou / Sci Rep 2024 PPG→BG; typical cut ≈ 0.85)
#   'ssqi'     — skewness (Elgendi / Chen et al. JBHI 2024; typical cut ≈ 0.3)
# Try-2 IPS CatBoost: ssqi=0.3 kept 191/252, MAE 0.5785 → 0.6001, so default off.
PPG_SQA_METHOD = {
    'MUST': 'template',
    'IPS': 'template',
    'vitaldb': 'template',
    'Hb_PPG': 'template',
}
DEFAULT_PPG_SQA_METHOD = 'template'
PPG_SQA_THRESHOLD = {
    'MUST': None,  # 0.85 for template, 0.3 for ssqi
    'IPS': None,
    'vitaldb': None,
    'Hb_PPG': None,
}
DEFAULT_PPG_SQA_THRESHOLD = None
PPG_SQA_WINDOW_SECONDS = 6.0

# Named CSV column(s) for multi-wavelength files. None = first column (MUST / vitaldb).
# A string selects one channel. A tuple/list extracts each channel: tabular features
# are prefixed (g1_660nm_ppg_mean, …); waveform models stack channels as (N, C, T).
# Revert Hb_PPG to '940nm' to reproduce the single-wavelength baseline.
PPG_CSV_COLUMN = {
    'Hb_PPG': ('660nm', '730nm', '850nm', '940nm'),
}
DEFAULT_PPG_CSV_COLUMN = None

# Drop glucose targets outside this quantile range (mmol/L).
TARGET_OUTLIER_LOWER_Q = 0
TARGET_OUTLIER_UPPER_Q = 1
# TARGET_OUTLIER_LOWER_Q = 0.01
# TARGET_OUTLIER_UPPER_Q = 0.99


def resolve_config_dataset(dataset: str | Path | None) -> str:
    """Map CLI dataset name/path to a settings profile (vitaldb/MUST/IPS)."""
    if dataset is None:
        return 'IPS'
    raw = str(dataset)
    name = Path(raw).name
    if raw in DATASET_CONFIG_PROFILE:
        return DATASET_CONFIG_PROFILE[raw]
    if name in DATASET_CONFIG_PROFILE:
        return DATASET_CONFIG_PROFILE[name]

    lower_name = name.lower()
    lower_path = raw.replace('\\', '/').lower()
    if lower_name.startswith('vitaldb') or '/vitaldb' in lower_path:
        return 'vitaldb'
    if lower_name == 'must' or '/must/' in f'/{lower_path}/':
        return 'MUST'
    if lower_name.startswith('ips') or '/ips' in lower_path:
        return 'IPS'
    if 'hb_ppg' in lower_name or 'hb-ppg' in lower_path or 'hb_ppg' in lower_path:
        return 'Hb_PPG'

    root = Path(raw)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    if root.is_dir():
        if (root / 'manifest.json').is_file() and (root / 'Labels').is_dir():
            return 'vitaldb'
        if (root / 'Labels').is_dir() and (root / 'RawData').is_dir():
            return 'MUST'
        if (root / 'bg_re.xlsx').is_file():
            return 'IPS'
        if (root / 'subject information.xlsx').is_file() and (root / 'data_csv').is_dir():
            return 'Hb_PPG'
    return raw


def basic_feature_keys(dataset: str) -> tuple:
    return BASIC_FEATURE_KEYS.get(resolve_config_dataset(dataset), DEFAULT_BASIC_FEATURE_KEYS)


def attach_demo_keys(dataset: str | None = None) -> tuple:
    """Demographics columns to join from labels (superset of BASIC_FEATURE_KEYS)."""
    del dataset  # Same attach set for IPS / MUST / VitalDB / Hb_PPG.
    return ATTACH_DEMO_KEYS


def ppg_sampling_rate(dataset: str) -> float:
    return PPG_SAMPLING_RATE.get(resolve_config_dataset(dataset), DEFAULT_PPG_SAMPLING_RATE)


def ppg_feature_groups(dataset: str) -> tuple[str, ...]:
    return PPG_FEATURE_GROUPS.get(resolve_config_dataset(dataset), DEFAULT_PPG_FEATURE_GROUPS)


# Explicit demo×PPG products for CatBoost (trees don't form products freely).
# Ablation: MUST MAE 0.708→0.696 / R2 -0.000→+0.052; IPS got worse — MUST only.
USE_INTERACTION_FEATURES = {
    'MUST': True,
    'IPS': False,
    'vitaldb': False,
    'Hb_PPG': False,
}
DEFAULT_USE_INTERACTION_FEATURES = False
INTERACTION_FEATURE_PAIRS = (
    ('age', 'g3_ppg_ibi_mean'),
    ('age', 'g3_ppg_ibi_std'),
    ('age', 'g4_ppg_spectral_entropy'),
    ('age', 'g4_ppg_band_power_low'),
    ('age', 'weight'),
    ('weight', 'g3_ppg_perfusion_index'),
    ('weight', 'g3_ppg_peak_amp_mean'),
)


def use_interaction_features(dataset: str) -> bool:
    return USE_INTERACTION_FEATURES.get(
        resolve_config_dataset(dataset), DEFAULT_USE_INTERACTION_FEATURES
    )


def ppg_trim_fractions(dataset: str) -> tuple[float, float]:
    return PPG_TRIM_FRACTIONS.get(resolve_config_dataset(dataset), DEFAULT_PPG_TRIM_FRACTIONS)


def ppg_filter(dataset: str) -> str:
    return PPG_FILTER.get(resolve_config_dataset(dataset), DEFAULT_PPG_FILTER)


def ppg_normalize(dataset: str) -> str:
    return PPG_NORMALIZE.get(resolve_config_dataset(dataset), DEFAULT_PPG_NORMALIZE)


def ppg_target_fs(dataset: str):
    return PPG_TARGET_FS.get(resolve_config_dataset(dataset), DEFAULT_PPG_TARGET_FS)


def ppg_best_segment_seconds(dataset: str):
    return PPG_BEST_SEGMENT_SECONDS.get(
        resolve_config_dataset(dataset), DEFAULT_PPG_BEST_SEGMENT_SECONDS
    )


def ppg_sqa_threshold(dataset: str):
    return PPG_SQA_THRESHOLD.get(resolve_config_dataset(dataset), DEFAULT_PPG_SQA_THRESHOLD)


def ppg_sqa_method(dataset: str) -> str:
    return PPG_SQA_METHOD.get(resolve_config_dataset(dataset), DEFAULT_PPG_SQA_METHOD)


def normalize_ppg_csv_column(
    value: str | Sequence[str] | None,
) -> str | tuple[str, ...] | None:
    """None, one column name, or two-plus names as a tuple."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    columns = tuple(str(column) for column in value)
    if not columns:
        return None
    if len(columns) == 1:
        return columns[0]
    return columns


def ppg_csv_column(dataset: str) -> str | tuple[str, ...] | None:
    return normalize_ppg_csv_column(
        PPG_CSV_COLUMN.get(resolve_config_dataset(dataset), DEFAULT_PPG_CSV_COLUMN)
    )


def ppg_csv_column_fingerprint(dataset: str) -> str | list[str] | None:
    """JSON-stable form of ``ppg_csv_column`` for feature/waveform cache keys."""
    value = ppg_csv_column(dataset)
    if isinstance(value, tuple):
        return list(value)
    return value


def resolve_ppg_center_window_seconds(ppg_seconds: float | None = None) -> float | None:
    """Center-crop length in seconds for any dataset.

    ``ppg_seconds`` from --ppg-seconds:
      None / <=0 → no crop (full on-disk signal)
      >0         → keep that many seconds centered on the file midpoint
    """
    if ppg_seconds is None:
        return None
    seconds = float(ppg_seconds)
    if seconds <= 0:
        return None
    return seconds


def catboost_kwargs() -> dict:
    return {'task_type': 'CPU'}


def xgboost_kwargs() -> dict:
    return {'tree_method': 'hist', 'device': 'cpu', 'n_jobs': -1}


def xgboost_train_params() -> dict:
    return {'tree_method': 'hist', 'device': 'cpu', 'nthread': -1}


def lightgbm_kwargs() -> dict:
    return {'device': 'cpu', 'n_jobs': -1}


def torch_device():
    import torch

    if USE_GPU is not False and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def torch_float_tensor(array):
    import torch

    return torch.tensor(array, dtype=torch.float32)


def tensor_to_numpy(tensor):
    import numpy as np

    return np.asarray(tensor.detach().cpu().tolist(), dtype=np.float32)
