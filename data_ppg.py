import pandas as pd
import numpy as np
from pathlib import Path
from utils import safe_float

def extract_ppg_bpm(row: pd.Series) -> dict:
    ppg_bpm = safe_float(row.get('PPG/BPM'))
    sppg_bpm = safe_float(row.get('SPPG/BPM'))
    if np.isnan(ppg_bpm) and np.isnan(sppg_bpm):
        return {}
    return {
        'ppg_bpm': ppg_bpm,
        'sppg_bpm': sppg_bpm,
    }

def read_signal_waveform(signal_path: Path) -> np.ndarray:
    """Return the raw PPG samples from a signal CSV or text file."""
    return _read_signal_values(signal_path)


def _read_signal_values(signal_path: Path) -> np.ndarray:
    if signal_path.suffix.lower() == '.csv':
        values = pd.read_csv(signal_path, header=None).iloc[:, 0]
        return values.to_numpy(dtype=float)
    with open(signal_path, 'r') as f:
        return np.asarray([float(x) for x in f.read().split()], dtype=float)


def extract_raw_ppg(signal_path: str) -> dict:
    signal = _read_signal_values(Path(signal_path))
    if signal.size == 0:
        raise ValueError('empty signal')

    mean_value = float(np.mean(signal))
    centered = signal - mean_value
    std_value = float(np.std(signal))
    diff = np.diff(signal)

    if std_value > 0:
        skewness = float(np.mean(centered ** 3) / (std_value ** 3))
        kurtosis = float(np.mean(centered ** 4) / (std_value ** 4) - 3.0)
    else:
        skewness = 0.0
        kurtosis = 0.0

    zero_crossings = float(np.sum(centered[:-1] * centered[1:] < 0)) if signal.size > 1 else 0.0

    return {
        'ppg_length': float(signal.size),
        'ppg_mean': mean_value,
        'ppg_std': std_value,
        'ppg_range': float(np.ptp(signal)),
        'ppg_rms': float(np.sqrt(np.mean(signal ** 2))),
        'ppg_abs_mean': float(np.mean(np.abs(signal))),
        'ppg_skew': skewness,
        'ppg_kurtosis': kurtosis,
        'ppg_zero_crossings': zero_crossings,
        'ppg_diff_mean': float(np.mean(np.abs(diff))) if diff.size else 0.0,
        'ppg_diff_std': float(np.std(diff)) if diff.size else 0.0,
    }
