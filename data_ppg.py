from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import safe_float

# Fraction trimmed from the start before feature extraction.
PPG_START_TRIM_FRACTION = 0.1


def trim_signal_start(
    signal: np.ndarray,
    trim_fraction: float = PPG_START_TRIM_FRACTION,
) -> np.ndarray:
    """Return the signal after trimming the noisy start portion."""
    values = np.asarray(signal, dtype=float).reshape(-1)
    if values.size == 0 or trim_fraction <= 0:
        return values
    if trim_fraction >= 1.0:
        raise ValueError('trim_fraction must be less than 1.0')

    start = int(values.size * trim_fraction)
    if start >= values.size:
        return values
    return values[start:]


def extract_ppg_bpm(row: pd.Series) -> dict:
    ppg_bpm = safe_float(row.get('PPG/BPM'))
    sppg_bpm = safe_float(row.get('SPPG/BPM'))
    if np.isnan(ppg_bpm) and np.isnan(sppg_bpm):
        return {}
    return {
        'ppg_bpm': ppg_bpm,
        'sppg_bpm': sppg_bpm,
    }


def read_signal_waveform(
    signal_path: Path,
    trim_fraction: float = PPG_START_TRIM_FRACTION,
) -> np.ndarray:
    """Return PPG samples from a signal file, skipping the noisy start."""
    signal = _read_signal_values(signal_path)
    return trim_signal_start(signal, trim_fraction=trim_fraction)


def _read_signal_values(signal_path: Path) -> np.ndarray:
    if signal_path.suffix.lower() == '.csv':
        values = pd.read_csv(signal_path, header=None).iloc[:, 0]
        return values.to_numpy(dtype=float)
    with open(signal_path, 'r') as f:
        return np.asarray([float(x) for x in f.read().split()], dtype=float)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window < 2 or values.size < window:
        return values.astype(float, copy=True)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode='same')


def _find_peaks(signal: np.ndarray, min_distance: int) -> np.ndarray:
    try:
        from scipy.signal import find_peaks

        prominence = max(float(np.std(signal)) * 0.25, 1e-6)
        peaks, _ = find_peaks(signal, distance=max(min_distance, 1), prominence=prominence)
        return peaks.astype(int)
    except Exception:
        peaks = []
        last_peak = -min_distance
        for index in range(1, signal.size - 1):
            if signal[index] <= signal[index - 1] or signal[index] <= signal[index + 1]:
                continue
            if index - last_peak < min_distance:
                if peaks and signal[index] > signal[peaks[-1]]:
                    peaks[-1] = index
                    last_peak = index
                continue
            peaks.append(index)
            last_peak = index
        return np.asarray(peaks, dtype=int)


def _safe_stats(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(values)), float(np.std(values)), float(np.median(values))


def _rmssd(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    diffs = np.diff(values.astype(float))
    return float(np.sqrt(np.mean(diffs ** 2)))


def _distribution_features(signal: np.ndarray, prefix: str) -> dict:
    mean_value = float(np.mean(signal))
    std_value = float(np.std(signal))
    centered = signal - mean_value
    if std_value > 0:
        skewness = float(np.mean(centered ** 3) / (std_value ** 3))
        kurtosis = float(np.mean(centered ** 4) / (std_value ** 4) - 3.0)
    else:
        skewness = 0.0
        kurtosis = 0.0

    q25, q75 = np.percentile(signal, [25, 75])
    iqr = float(q75 - q25)
    cv = float(std_value / abs(mean_value)) if abs(mean_value) > 1e-8 else 0.0
    zero_crossings = (
        float(np.sum(centered[:-1] * centered[1:] < 0)) if signal.size > 1 else 0.0
    )

    return {
        f'{prefix}length': float(signal.size),
        f'{prefix}mean': mean_value,
        f'{prefix}median': float(np.median(signal)),
        f'{prefix}std': std_value,
        f'{prefix}range': float(np.ptp(signal)),
        f'{prefix}iqr': iqr,
        f'{prefix}cv': cv,
        f'{prefix}rms': float(np.sqrt(np.mean(signal ** 2))),
        f'{prefix}abs_mean': float(np.mean(np.abs(signal))),
        f'{prefix}skew': skewness,
        f'{prefix}kurtosis': kurtosis,
        f'{prefix}zero_crossings': zero_crossings,
    }


def _pulse_features(signal: np.ndarray) -> dict:
    baseline_window = max(5, signal.size // 20)
    baseline = _moving_average(signal, baseline_window)
    ac = signal - baseline

    min_distance = max(3, signal.size // 120)
    peaks = _find_peaks(ac, min_distance=min_distance)

    if peaks.size < 2:
        return {
            'ppg_peak_count': float(peaks.size),
            'ppg_beat_density': 0.0,
            'ppg_ibi_mean': 0.0,
            'ppg_ibi_std': 0.0,
            'ppg_ibi_rmssd': 0.0,
            'ppg_peak_amp_mean': 0.0,
            'ppg_peak_amp_std': 0.0,
            'ppg_peak_amp_cv': 0.0,
            'ppg_max_slope': 0.0,
            'ppg_mean_rise_slope': 0.0,
            'ppg_perfusion_index': 0.0,
            'ppg_ac_dc_ratio': 0.0,
        }

    ibi = np.diff(peaks).astype(float)
    peak_amps = ac[peaks]
    amp_mean, amp_std, _ = _safe_stats(peak_amps)
    amp_cv = float(amp_std / amp_mean) if amp_mean > 1e-8 else 0.0

    rise_slopes = []
    for peak in peaks:
        start = max(0, int(peak) - min_distance)
        segment = ac[start:peak + 1]
        if segment.size < 2:
            continue
        rise_slopes.append(float(np.max(np.diff(segment)) / max(np.std(ac), 1e-6)))

    dc_mean = float(np.mean(np.abs(baseline)))
    ac_std = float(np.std(ac))
    perfusion_index = float(ac_std / dc_mean) if dc_mean > 1e-8 else 0.0

    return {
        'ppg_peak_count': float(peaks.size),
        'ppg_beat_density': float(peaks.size / signal.size),
        'ppg_ibi_mean': float(np.mean(ibi)),
        'ppg_ibi_std': float(np.std(ibi)),
        'ppg_ibi_rmssd': _rmssd(ibi),
        'ppg_peak_amp_mean': amp_mean,
        'ppg_peak_amp_std': amp_std,
        'ppg_peak_amp_cv': amp_cv,
        'ppg_max_slope': float(np.max(np.abs(np.diff(ac)))) if ac.size > 1 else 0.0,
        'ppg_mean_rise_slope': float(np.mean(rise_slopes)) if rise_slopes else 0.0,
        'ppg_perfusion_index': perfusion_index,
        'ppg_ac_dc_ratio': perfusion_index,
    }


def _spectral_features(signal: np.ndarray) -> dict:
    if signal.size < 16:
        return {
            'ppg_dom_freq': 0.0,
            'ppg_spectral_entropy': 0.0,
            'ppg_band_power_low': 0.0,
            'ppg_band_power_mid': 0.0,
            'ppg_band_power_high': 0.0,
            'ppg_snr': 0.0,
        }

    centered = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(centered.size, d=1.0)
    total_power = float(np.sum(spectrum))
    if total_power <= 1e-12:
        return {
            'ppg_dom_freq': 0.0,
            'ppg_spectral_entropy': 0.0,
            'ppg_band_power_low': 0.0,
            'ppg_band_power_mid': 0.0,
            'ppg_band_power_high': 0.0,
            'ppg_snr': 0.0,
        }

    norm_spectrum = spectrum / total_power
    dom_freq = float(freqs[int(np.argmax(spectrum[1:])) + 1]) if spectrum.size > 1 else 0.0
    entropy = float(-np.sum(norm_spectrum[norm_spectrum > 1e-12] * np.log(norm_spectrum[norm_spectrum > 1e-12])))

    low = float(np.sum(spectrum[freqs <= 0.05]) / total_power)
    mid = float(np.sum(spectrum[(freqs > 0.05) & (freqs <= 0.2)]) / total_power)
    high = float(np.sum(spectrum[freqs > 0.2]) / total_power)

    if signal.size > 8:
        smooth = _moving_average(signal, max(3, signal.size // 64))
        noise = signal - smooth
        signal_power = float(np.var(smooth))
        noise_power = float(np.var(noise))
        snr = float(signal_power / noise_power) if noise_power > 1e-12 else 0.0
    else:
        snr = 0.0

    return {
        'ppg_dom_freq': dom_freq,
        'ppg_spectral_entropy': entropy,
        'ppg_band_power_low': low,
        'ppg_band_power_mid': mid,
        'ppg_band_power_high': high,
        'ppg_snr': snr,
    }


def _autocorr_features(signal: np.ndarray) -> dict:
    if signal.size < 8:
        return {
            'ppg_autocorr_peak_lag': 0.0,
            'ppg_autocorr_peak_val': 0.0,
            'ppg_pulse_rate': 0.0,
        }

    centered = signal - np.mean(signal)
    autocorr = np.correlate(centered, centered, mode='full')
    autocorr = autocorr[autocorr.size // 2 :]
    if autocorr[0] <= 0:
        return {
            'ppg_autocorr_peak_lag': 0.0,
            'ppg_autocorr_peak_val': 0.0,
            'ppg_pulse_rate': 0.0,
        }

    autocorr = autocorr / autocorr[0]
    min_lag = max(3, signal.size // 200)
    max_lag = max(min_lag + 1, signal.size // 2)
    search = autocorr[min_lag:max_lag]
    if search.size == 0:
        return {
            'ppg_autocorr_peak_lag': 0.0,
            'ppg_autocorr_peak_val': 0.0,
            'ppg_pulse_rate': 0.0,
        }

    peak_lag = float(min_lag + int(np.argmax(search)))
    peak_val = float(autocorr[int(peak_lag)])
    pulse_rate = float(1.0 / peak_lag) if peak_lag > 0 else 0.0
    return {
        'ppg_autocorr_peak_lag': peak_lag,
        'ppg_autocorr_peak_val': peak_val,
        'ppg_pulse_rate': pulse_rate,
    }


def _derivative_features(signal: np.ndarray) -> dict:
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1) if diff1.size > 1 else np.empty(0, dtype=float)
    return {
        'ppg_diff_mean': float(np.mean(np.abs(diff1))) if diff1.size else 0.0,
        'ppg_diff_std': float(np.std(diff1)) if diff1.size else 0.0,
        'ppg_diff2_mean': float(np.mean(np.abs(diff2))) if diff2.size else 0.0,
        'ppg_diff2_std': float(np.std(diff2)) if diff2.size else 0.0,
    }


def _segment_features(signal: np.ndarray) -> dict:
    midpoint = signal.size // 2
    if midpoint == 0:
        return {
            'ppg_half_mean_diff': 0.0,
            'ppg_half_std_ratio': 0.0,
            'ppg_edge_mean_diff': 0.0,
        }

    first_half = signal[:midpoint]
    second_half = signal[midpoint:]
    edge = max(1, signal.size // 10)
    start_mean = float(np.mean(signal[:edge]))
    end_mean = float(np.mean(signal[-edge:]))
    half_std_ratio = (
        float(np.std(second_half) / np.std(first_half))
        if np.std(first_half) > 1e-8
        else 0.0
    )
    return {
        'ppg_half_mean_diff': float(np.mean(second_half) - np.mean(first_half)),
        'ppg_half_std_ratio': half_std_ratio,
        'ppg_edge_mean_diff': float(end_mean - start_mean),
    }


def extract_ppg_features(signal: np.ndarray) -> dict:
    """Extract hand-crafted PPG features from a 1-D waveform."""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size == 0:
        raise ValueError('empty signal')

    features = {}
    features.update(_distribution_features(signal, prefix='ppg_'))
    features.update(_derivative_features(signal))
    features.update(_pulse_features(signal))
    features.update(_spectral_features(signal))
    features.update(_autocorr_features(signal))
    features.update(_segment_features(signal))
    return features


def extract_raw_ppg(
    signal_path: str | Path,
    trim_fraction: float = PPG_START_TRIM_FRACTION,
) -> dict:
    signal = read_signal_waveform(Path(signal_path), trim_fraction=trim_fraction)
    return extract_ppg_features(signal)
