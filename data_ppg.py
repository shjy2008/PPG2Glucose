from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DEFAULT_PPG_BEST_SEGMENT_SECONDS,
    DEFAULT_PPG_FEATURE_GROUPS,
    DEFAULT_PPG_FILTER,
    DEFAULT_PPG_NORMALIZE,
    DEFAULT_PPG_SAMPLING_RATE,
    DEFAULT_PPG_TARGET_FS,
    DEFAULT_PPG_TRIM_FRACTIONS,
    PPG_BEST_SEGMENT_HOP_SECONDS,
    PPG_SQA_WINDOW_SECONDS,
    normalize_ppg_csv_column,
)
from utils import safe_float


def trim_signal(
    signal: np.ndarray,
    start_trim_fraction: float = DEFAULT_PPG_TRIM_FRACTIONS[0],
    end_trim_fraction: float = DEFAULT_PPG_TRIM_FRACTIONS[1],
) -> np.ndarray:
    """Return the signal after trimming configurable portions from start and/or end."""
    values = np.asarray(signal, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    if start_trim_fraction < 0 or end_trim_fraction < 0:
        raise ValueError('trim fractions must be non-negative')
    if start_trim_fraction + end_trim_fraction >= 1.0:
        raise ValueError('start_trim_fraction + end_trim_fraction must be less than 1.0')

    start = int(values.size * start_trim_fraction) if start_trim_fraction > 0 else 0
    end = (
        values.size - int(values.size * end_trim_fraction)
        if end_trim_fraction > 0
        else values.size
    )
    if end <= start:
        return values
    return values[start:end]


def _odd_window(samples: int, minimum: int = 5) -> int:
    size = max(int(samples), minimum)
    if size % 2 == 0:
        size += 1
    return size


def mswf_filter(
    signal: np.ndarray,
    fs: float,
    *,
    poly_order: int = 2,
    small_window_seconds: float = 0.05,
    large_window_seconds: float = 1.0,
    n_iterations: int = 3,
) -> np.ndarray:
    """Multi-size weighted fitting (Chen et al., JBHI 2024, Algorithm 1).

    Stage 1: local polynomial smoothing (Savitzky–Golay).
    Stage 2: iterative Gaussian-weighted large-window baseline estimate, then subtract.
    """
    values = np.asarray(signal, dtype=float).reshape(-1)
    n = values.size
    if n < 8 or not np.isfinite(values).all():
        return values

    sw = _odd_window(int(round(small_window_seconds * fs)), minimum=5)
    lw = _odd_window(int(round(large_window_seconds * fs)), minimum=sw + 2)
    sw = min(sw, n if n % 2 == 1 else n - 1)
    lw = min(lw, n if n % 2 == 1 else n - 1)
    if sw < 5 or lw <= sw:
        return values

    k_smooth = min(poly_order, sw - 1)
    k_base = min(poly_order, lw - 1)
    try:
        from scipy.signal import savgol_filter

        smoothed = savgol_filter(values, window_length=sw, polyorder=k_smooth, mode='interp')
    except Exception:
        smoothed = values.copy()

    # Coarse-grid IRLS baseline (Gaussian weights), then interpolate to full length.
    half = lw // 2
    step = max(1, half // 2)
    centers = np.arange(0, n, step, dtype=int)
    if centers[-1] != n - 1:
        centers = np.append(centers, n - 1)

    x_all = np.arange(n, dtype=float)
    baseline_at_centers = np.empty(centers.size, dtype=float)
    for ci, center in enumerate(centers):
        left = max(0, center - half)
        right = min(n, center + half + 1)
        x = x_all[left:right]
        y = smoothed[left:right]
        if y.size <= k_base:
            baseline_at_centers[ci] = float(np.mean(y))
            continue

        # Map local x to [-1, 1] as in the paper's Stage 2.
        x_norm = 2.0 * (x - x[0]) / max(x[-1] - x[0], 1.0) - 1.0
        # Initial Gaussian weights around the center in normalized coords.
        xc = 2.0 * (center - x[0]) / max(x[-1] - x[0], 1.0) - 1.0
        w = np.exp(-0.5 * ((x_norm - xc) / 0.45) ** 2)
        w = np.maximum(w, 1e-6)

        for _ in range(max(1, n_iterations)):
            coeffs = np.polynomial.polynomial.polyfit(x_norm, y, deg=k_base, w=w)
            y_hat = np.polynomial.polynomial.polyval(x_norm, coeffs)
            resid = np.abs(y - y_hat)
            dm = float(np.median(resid))
            if dm < 1e-12:
                break
            # Gaussian reweight on residual / median residual.
            w = np.exp(-0.5 * (resid / dm) ** 2)
            w = np.maximum(w, 1e-6)

        baseline_at_centers[ci] = float(
            np.polynomial.polynomial.polyval(
                np.asarray([xc]),
                np.polynomial.polynomial.polyfit(x_norm, y, deg=k_base, w=w),
            )[0]
        )

    baseline = np.interp(x_all, centers.astype(float), baseline_at_centers)
    return smoothed - baseline


def _parse_butterworth_name(name: str) -> tuple[float, float, int] | None:
    """Return (low_hz, high_hz, order) for butterworth aliases, else None.

    Accepted: butterworth / butter / bw (default 0.5–8 Hz, order 3),
    or butterworth_0.5_8 / butter_0.5_5 (order still 3).
    """
    if name in {'butterworth', 'butter', 'bw'}:
        return 0.5, 8.0, 3
    for prefix in ('butterworth_', 'butter_', 'bw_'):
        if not name.startswith(prefix):
            continue
        parts = name[len(prefix) :].split('_')
        if len(parts) != 2:
            return None
        try:
            low_hz, high_hz = float(parts[0]), float(parts[1])
        except ValueError:
            return None
        if not (0.0 < low_hz < high_hz):
            return None
        return low_hz, high_hz, 3
    return None


def butterworth_bandpass_filter(
    signal: np.ndarray,
    fs: float,
    *,
    low_hz: float = 0.5,
    high_hz: float = 8.0,
    order: int = 3,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass (common PPG→glucose preprocessing)."""
    values = np.asarray(signal, dtype=float).reshape(-1)
    rate = float(fs)
    if values.size < 16 or rate <= 0 or not np.isfinite(values).all():
        return values

    nyquist = 0.5 * rate
    low = float(low_hz)
    high = min(float(high_hz), nyquist * 0.99)
    if not (0.0 < low < high < nyquist):
        return values

    try:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(int(order), [low, high], btype='bandpass', fs=rate, output='sos')
        # sosfiltfilt needs enough samples for its default padlen.
        if values.size <= 3 * (2 * int(order) + 1):
            return values
        return sosfiltfilt(sos, values).astype(float, copy=False)
    except Exception:
        return values


def apply_ppg_filter(
    signal: np.ndarray,
    fs: float,
    filter_name: str = DEFAULT_PPG_FILTER,
) -> np.ndarray:
    name = (filter_name or 'none').strip().lower()
    if name in {'', 'none', 'off', 'false'}:
        return np.asarray(signal, dtype=float).reshape(-1)
    if name == 'mswf':
        return mswf_filter(signal, fs)
    butter = _parse_butterworth_name(name)
    if butter is not None:
        low_hz, high_hz, order = butter
        return butterworth_bandpass_filter(
            signal, fs, low_hz=low_hz, high_hz=high_hz, order=order
        )
    raise ValueError(
        f'Unknown PPG filter: {filter_name!r} '
        f'(expected none|mswf|butterworth|butterworth_0.5_8|butterworth_0.5_5)'
    )


def zscore_normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Per-channel zero-mean / unit-variance normalization."""

    def _one(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            return values
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std < 1e-12:
            return values - mean
        return (values - mean) / std

    return _apply_per_channel(signal, _one)


def apply_ppg_normalize(
    signal: np.ndarray,
    normalize: str = DEFAULT_PPG_NORMALIZE,
) -> np.ndarray:
    name = (normalize or 'none').strip().lower()
    if name in {'', 'none', 'off', 'false'}:
        arr = np.asarray(signal, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(-1)
        return arr
    if name in {'zscore', 'z-score', 'standard', 'true', 'on', '1'}:
        return zscore_normalize_signal(signal)
    raise ValueError(
        f'Unknown PPG normalize mode: {normalize!r} (expected none|zscore)'
    )


def effective_sampling_rate(source_fs: float, target_fs: float | None) -> float:
    """Return the sampling rate after optional downsampling."""
    if target_fs is None:
        return float(source_fs)
    target = float(target_fs)
    if target <= 0:
        raise ValueError(f'target_fs must be positive, got {target_fs!r}')
    source = float(source_fs)
    if target >= source:
        return source
    return target


def center_crop_seconds(
    signal: np.ndarray,
    fs: float,
    window_seconds: float | None,
) -> np.ndarray:
    """Keep a centered window of ``window_seconds`` (None/<=0 keeps the full signal)."""
    values = np.asarray(signal, dtype=float).reshape(-1)
    if window_seconds is None or float(window_seconds) <= 0 or values.size == 0:
        return values
    n = int(round(float(window_seconds) * float(fs)))
    if n <= 0 or values.size <= n:
        return values
    start = (values.size - n) // 2
    return values[start : start + n]


def downsample_signal(
    signal: np.ndarray,
    source_fs: float,
    target_fs: float | None,
) -> np.ndarray:
    """Resample to target_fs when lower than source_fs; otherwise return unchanged."""
    values = np.asarray(signal, dtype=float).reshape(-1)
    fs_out = effective_sampling_rate(source_fs, target_fs)
    source = float(source_fs)
    if values.size == 0 or fs_out >= source:
        return values

    n_out = max(1, int(round(values.size * fs_out / source)))
    if n_out >= values.size:
        return values
    try:
        from scipy.signal import resample

        return resample(values, n_out).astype(float, copy=False)
    except Exception:
        indices = np.linspace(0, values.size - 1, n_out).astype(int)
        return values[indices]


def ppg_ssqi(signal: np.ndarray) -> float:
    """Skewness-based signal quality index (Elgendi / Chen et al. JBHI 2024)."""
    values = np.asarray(signal, dtype=float).reshape(-1)
    if values.size < 8:
        return 0.0
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-12:
        return 0.0
    centered = (values - mean) / std
    return float(np.mean(centered ** 3))


def ppg_kurtosis(signal: np.ndarray) -> float:
    """Pearson kurtosis (4th standardized moment). Higher often flags spikes/artifacts."""
    values = np.asarray(signal, dtype=float).reshape(-1)
    if values.size < 8:
        return 0.0
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-12:
        return 0.0
    centered = (values - mean) / std
    return float(np.mean(centered ** 4))


def ppg_ksqi(signal: np.ndarray) -> float:
    """Kurtosis SQI with higher=better (Elgendi KSQI, inverted for ranking).

    Empirically on IPS/VitalDB, clean windows sit near kurtosis ~2–3; heavy tails
    (spikes) push kurtosis up. Score = 1/(1+kurtosis) so high kurtosis ranks worse.
    """
    kurt = ppg_kurtosis(signal)
    if kurt <= 0.0:
        return 0.0
    return float(1.0 / (1.0 + kurt))


# Reject spike-like windows when gating skew / template with kurtosis.
PPG_KURTOSIS_MAX = 5.0


def ppg_skew_kurt_sqi(
    signal: np.ndarray,
    kurtosis_max: float = PPG_KURTOSIS_MAX,
) -> float:
    """Skewness SSQI, zeroed when kurtosis exceeds the spike ceiling."""
    skew = ppg_ssqi(signal)
    if ppg_kurtosis(signal) > float(kurtosis_max):
        return 0.0
    return skew


def ppg_template_kurt_sqi(
    signal: np.ndarray,
    fs: float,
    kurtosis_max: float = PPG_KURTOSIS_MAX,
) -> float:
    """Template cosine SQI, zeroed when kurtosis exceeds the spike ceiling."""
    score = ppg_template_sqi(signal, fs)
    if ppg_kurtosis(signal) > float(kurtosis_max):
        return 0.0
    return score


def _ppg_peak_indices(signal: np.ndarray, fs: float) -> np.ndarray:
    """Systolic-peak sample indices on a lightly band-passed copy; empty on failure."""
    values = np.asarray(signal, dtype=float).reshape(-1)
    if values.size < max(16, int(round(2.0 * float(fs)))):
        return np.asarray([], dtype=int)
    try:
        from scipy.signal import butter, find_peaks, resample, sosfiltfilt
    except Exception:
        return np.asarray([], dtype=int)

    quality_fs = min(float(fs), 100.0)
    n_quality = max(16, int(round(values.size * quality_fs / float(fs))))
    quality_signal = (
        resample(values, n_quality) if n_quality < values.size else values.copy()
    )
    centered = quality_signal - np.mean(quality_signal)
    sos = butter(3, [0.5, 8.0], btype='bandpass', fs=quality_fs, output='sos')
    pulse_band = sosfiltfilt(sos, centered)
    peaks, _ = find_peaks(
        pulse_band,
        distance=max(3, int(round(0.3 * quality_fs))),
        prominence=max(0.15 * float(np.std(pulse_band)), 1e-9),
    )
    if peaks.size == 0 or quality_fs >= float(fs) * 0.999:
        return peaks.astype(int)
    # Map peak indices from the quality-rate grid back to native samples.
    scale = float(values.size) / float(quality_signal.size)
    return np.clip(np.round(peaks.astype(float) * scale).astype(int), 0, values.size - 1)


def ppg_template_sqi(signal: np.ndarray, fs: float) -> float:
    """Mean cosine similarity of beats to the mean-pulse template.

    Orphanidou / Sci Rep 2024-style morphology SQA. Each beat is mean-centered and
    L2-normalized before averaging into a template; the score is the mean cosine
    similarity of beats to that template. Returns 0 if fewer than 3 usable beats.
    """
    values = np.asarray(signal, dtype=float).reshape(-1)
    peaks = _ppg_peak_indices(values, fs)
    if peaks.size < 3:
        return 0.0

    beat_len = int(np.median(np.diff(peaks)))
    if beat_len < 4:
        return 0.0
    half = beat_len // 2

    beats: list[np.ndarray] = []
    for peak in peaks:
        start = int(peak) - half
        end = start + beat_len
        if start < 0 or end > values.size:
            continue
        beat = values[start:end].astype(float, copy=True)
        beat -= float(np.mean(beat))
        norm = float(np.linalg.norm(beat))
        if norm < 1e-12:
            continue
        beats.append(beat / norm)

    if len(beats) < 3:
        return 0.0

    stacked = np.stack(beats, axis=0)
    template = stacked.mean(axis=0)
    t_norm = float(np.linalg.norm(template))
    if t_norm < 1e-12:
        return 0.0
    template /= t_norm
    return float(np.mean(stacked @ template))


def ppg_quality_score(
    signal: np.ndarray,
    fs: float,
    method: str = 'template',
) -> float:
    """Dispatch SQA score.

    Methods:
      template      — cosine to mean-pulse template
      ssqi          — skewness
      ksqi          — Pearson kurtosis
      skew_kurt     — skewness gated by kurtosis >= 3
      template_kurt — template gated by kurtosis >= 3
    """
    name = str(method).strip().lower()
    if name in ('template', 'template_sqi', 'tsqi'):
        return ppg_template_sqi(signal, fs)
    if name in ('ssqi', 'skew', 'skewness'):
        return ppg_ssqi(signal)
    if name in ('ksqi', 'kurt', 'kurtosis'):
        return ppg_ksqi(signal)
    if name in ('skew_kurt', 'ssqi_kurt', 'skewness_kurtosis'):
        return ppg_skew_kurt_sqi(signal)
    if name in ('template_kurt', 'template_ksqi', 'tsqi_kurt'):
        return ppg_template_kurt_sqi(signal, fs)
    raise ValueError(
        f'Unknown PPG SQA method {method!r} '
        f'(expected template|ssqi|ksqi|skew_kurt|template_kurt)'
    )


def select_best_segment(
    signal: np.ndarray,
    fs: float,
    window_seconds: float | None,
    hop_seconds: float = PPG_BEST_SEGMENT_HOP_SECONDS,
    sqa_method: str = 'template',
) -> np.ndarray:
    """Select the overlapping full window with the highest SQA score.

    Multi-channel signals keep a shared time window (mean score across channels).
    """
    values = np.asarray(signal, dtype=float)
    if values.size == 0 or window_seconds is None:
        return values

    window_samples = int(round(float(window_seconds) * float(fs)))
    hop_samples = int(round(float(hop_seconds) * float(fs)))
    if window_samples < 1 or hop_samples < 1:
        raise ValueError('best-segment window and hop must yield at least one sample')

    if values.ndim == 1:
        if values.size <= window_samples:
            return values
        starts = range(0, values.size - window_samples + 1, hop_samples)
        best_start = max(
            starts,
            key=lambda start: ppg_quality_score(
                values[start : start + window_samples], fs, method=sqa_method
            ),
        )
        return values[best_start : best_start + window_samples]

    if values.ndim != 2:
        raise ValueError(f'PPG signal must be 1-D or (C, T), got shape {values.shape}')
    length = values.shape[1]
    if length <= window_samples:
        return values
    starts = range(0, length - window_samples + 1, hop_samples)

    def _score(start: int) -> float:
        return float(
            np.mean(
                [
                    ppg_quality_score(
                        channel[start : start + window_samples], fs, method=sqa_method
                    )
                    for channel in values
                ]
            )
        )

    best_start = max(starts, key=_score)
    return values[:, best_start : best_start + window_samples]


def select_sqa_windows(
    signal: np.ndarray,
    fs: float,
    threshold: float,
    window_seconds: float = PPG_SQA_WINDOW_SECONDS,
    sqa_method: str = 'template',
) -> list[np.ndarray]:
    """Keep non-overlapping windows with quality score > threshold."""
    return [
        window
        for window in segment_signal(signal, fs=fs, window_seconds=window_seconds)
        if ppg_quality_score(window, fs, method=sqa_method) > float(threshold)
    ]


def apply_sqa_mask(
    signal: np.ndarray,
    fs: float,
    threshold: float | None,
    window_seconds: float = PPG_SQA_WINDOW_SECONDS,
    sqa_method: str = 'template',
) -> np.ndarray | None:
    """Concatenate SQA-accepted windows. None if SQA rejects the whole signal.

    Multi-channel signals keep a window only when every channel passes.
    """
    values = np.asarray(signal, dtype=float)
    if threshold is None:
        return values
    if values.ndim == 1:
        good = select_sqa_windows(
            values,
            fs=fs,
            threshold=threshold,
            window_seconds=window_seconds,
            sqa_method=sqa_method,
        )
        if not good:
            return None
        return np.concatenate(good)
    if values.ndim != 2:
        raise ValueError(f'PPG signal must be 1-D or (C, T), got shape {values.shape}')
    kept = []
    for window in segment_signal(values, fs=fs, window_seconds=window_seconds):
        if all(
            ppg_quality_score(channel, fs, method=sqa_method) > float(threshold)
            for channel in window
        ):
            kept.append(window)
    if not kept:
        return None
    return np.concatenate(kept, axis=1)


def extract_ppg_bpm(row: pd.Series) -> dict:
    ppg_bpm = safe_float(row.get('PPG/BPM'))
    sppg_bpm = safe_float(row.get('SPPG/BPM'))
    if np.isnan(ppg_bpm) and np.isnan(sppg_bpm):
        return {}
    return {
        'ppg_bpm': ppg_bpm,
        'sppg_bpm': sppg_bpm,
    }


def csv_column_names(
    csv_column: str | Sequence[str] | None,
) -> tuple[str, ...] | None:
    value = normalize_ppg_csv_column(csv_column)
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return value


def csv_column_tag(name: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in str(name).strip()).strip('_')


def _apply_per_channel(signal: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        return np.asarray(fn(arr), dtype=float).reshape(-1)
    if arr.ndim != 2:
        raise ValueError(f'PPG signal must be 1-D or (C, T), got shape {arr.shape}')
    channels = [np.asarray(fn(channel), dtype=float).reshape(-1) for channel in arr]
    lengths = {channel.size for channel in channels}
    if len(lengths) != 1:
        raise ValueError(
            f'Wavelength channels diverged during preprocessing: {[c.size for c in channels]}'
        )
    return np.stack(channels, axis=0)


def read_signal_waveform(
    signal_path: Path,
    trim_fractions: tuple[float, float] = DEFAULT_PPG_TRIM_FRACTIONS,
    fs: float = DEFAULT_PPG_SAMPLING_RATE,
    filter_name: str = DEFAULT_PPG_FILTER,
    target_fs: float | None = DEFAULT_PPG_TARGET_FS,
    center_window_seconds: float | None = None,
    csv_column: str | Sequence[str] | None = None,
    normalize: str = DEFAULT_PPG_NORMALIZE,
) -> np.ndarray:
    """Return PPG samples (trim → filter → downsample → crop → optional normalize).

    One CSV column yields a 1-D array of length T. Several columns yield (C, T).
    """
    signal = _read_signal_values(signal_path, csv_column=csv_column)
    signal = _apply_per_channel(signal, lambda values: trim_signal(values, *trim_fractions))
    signal = _apply_per_channel(
        signal, lambda values: apply_ppg_filter(values, fs=fs, filter_name=filter_name)
    )
    signal = _apply_per_channel(
        signal,
        lambda values: downsample_signal(values, source_fs=fs, target_fs=target_fs),
    )
    fs_eff = effective_sampling_rate(fs, target_fs)
    signal = _apply_per_channel(
        signal,
        lambda values: center_crop_seconds(
            values, fs=fs_eff, window_seconds=center_window_seconds
        ),
    )
    return apply_ppg_normalize(signal, normalize=normalize)


def _csv_has_header(columns) -> bool:
    try:
        float(str(columns[0]).strip())
    except (TypeError, ValueError):
        return True
    return False


def _match_csv_column(columns, csv_column: str) -> str:
    if csv_column in columns:
        return csv_column
    want = str(csv_column).lower().replace(' ', '')
    for name in columns:
        if str(name).lower().replace(' ', '') == want:
            return str(name)
    raise ValueError(
        f'PPG CSV column {csv_column!r} not in {list(columns)}'
    )


def _finite_signal_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    mat = np.column_stack(
        [
            pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
            for column in columns
        ]
    )
    finite = np.isfinite(mat).all(axis=1)
    mat = mat[finite]
    if mat.shape[1] == 1:
        return mat[:, 0]
    return mat.T


def _read_signal_values(
    signal_path: Path,
    csv_column: str | Sequence[str] | None = None,
) -> np.ndarray:
    names = csv_column_names(csv_column)
    if signal_path.suffix.lower() != '.csv':
        if names is not None and len(names) > 1:
            raise ValueError(
                f'Multi-wavelength PPG requires a CSV; got {signal_path.suffix}'
            )
        with open(signal_path, 'r') as f:
            return np.asarray([float(x) for x in f.read().split()], dtype=float)

    header = pd.read_csv(signal_path, nrows=0)
    if not _csv_has_header(header.columns):
        if names is not None and len(names) > 1:
            raise ValueError(
                f'Headerless CSV cannot select named wavelengths {list(names)}'
            )
        frame = pd.read_csv(signal_path, header=None)
        values = frame.iloc[:, 0]
        arr = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
        return arr[np.isfinite(arr)]

    if names is None:
        frame = pd.read_csv(signal_path)
        return _finite_signal_matrix(frame, [frame.columns[0]])

    try:
        resolved = [_match_csv_column(header.columns, name) for name in names]
        frame = pd.read_csv(signal_path, usecols=resolved)
    except ValueError:
        frame = pd.read_csv(signal_path)
        resolved = [_match_csv_column(frame.columns, name) for name in names]
    return _finite_signal_matrix(frame, resolved)


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
    mean_crossing_count = (
        float(np.sum(centered[:-1] * centered[1:] < 0)) if signal.size > 1 else 0.0
    )

    return {
        f'{prefix}ppg_length': float(signal.size),
        f'{prefix}ppg_mean': mean_value,
        f'{prefix}ppg_median': float(np.median(signal)),
        f'{prefix}ppg_std': std_value,
        f'{prefix}ppg_range': float(np.ptp(signal)), # peak-to-peak range
        f'{prefix}ppg_iqr': iqr, # interquartile range
        f'{prefix}ppg_cv': cv, # coefficient of variation
        f'{prefix}ppg_rms': float(np.sqrt(np.mean(signal ** 2))),
        f'{prefix}ppg_abs_mean': float(np.mean(np.abs(signal))),
        f'{prefix}ppg_skew': skewness,
        f'{prefix}ppg_kurtosis': kurtosis,
        f'{prefix}ppg_mean_crossing_count': mean_crossing_count,
    }


def _pulse_features(signal: np.ndarray, prefix: str) -> dict:
    baseline_window = max(5, signal.size // 20)
    baseline = _moving_average(signal, baseline_window) # direct current
    ac = signal - baseline # alternating current

    min_distance = max(3, signal.size // 120)
    peaks = _find_peaks(ac, min_distance=min_distance)

    if peaks.size < 2:
        return {
            f'{prefix}ppg_peak_count': float(peaks.size),
            f'{prefix}ppg_beat_density': 0.0,
            f'{prefix}ppg_ibi_mean': 0.0,
            f'{prefix}ppg_ibi_std': 0.0,
            f'{prefix}ppg_ibi_rmssd': 0.0,
            f'{prefix}ppg_peak_amp_mean': 0.0,
            f'{prefix}ppg_peak_amp_std': 0.0,
            f'{prefix}ppg_peak_amp_cv': 0.0,
            f'{prefix}ppg_max_slope': 0.0,
            f'{prefix}ppg_mean_rise_slope': 0.0,
            f'{prefix}ppg_perfusion_index': 0.0
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
        f'{prefix}ppg_peak_count': float(peaks.size),
        f'{prefix}ppg_beat_density': float(peaks.size / signal.size),
        f'{prefix}ppg_ibi_mean': float(np.mean(ibi)),
        f'{prefix}ppg_ibi_std': float(np.std(ibi)),
        f'{prefix}ppg_ibi_rmssd': _rmssd(ibi),
        f'{prefix}ppg_peak_amp_mean': amp_mean,
        f'{prefix}ppg_peak_amp_std': amp_std,
        f'{prefix}ppg_peak_amp_cv': amp_cv,
        f'{prefix}ppg_max_slope': float(np.max(np.abs(np.diff(ac)))) if ac.size > 1 else 0.0,
        f'{prefix}ppg_mean_rise_slope': float(np.mean(rise_slopes)) if rise_slopes else 0.0,
        f'{prefix}ppg_perfusion_index': perfusion_index
    }


def _emd_features(signal: np.ndarray, prefix: str, fs: float) -> dict:
    """EMD / IMF energy fractions and simple IMF1 stats (literature-style ablation)."""
    empty = {
        f'{prefix}ppg_emd_energy_imf1': 0.0,
        f'{prefix}ppg_emd_energy_imf2': 0.0,
        f'{prefix}ppg_emd_energy_imf3': 0.0,
        f'{prefix}ppg_emd_energy_resid': 0.0,
        f'{prefix}ppg_emd_energy_entropy': 0.0,
        f'{prefix}ppg_emd_imf1_std': 0.0,
    }
    values = np.asarray(signal, dtype=float).reshape(-1)
    if values.size < 32 or fs <= 0:
        return empty

    # Downsample / truncate so EMD stays tractable on high-fs MUST.
    target_fs = 100.0
    x = values - float(np.mean(values))
    fs_eff = float(fs)
    if fs_eff > target_fs * 1.5:
        step = max(1, int(round(fs_eff / target_fs)))
        x = x[::step]
        fs_eff = fs_eff / step
    max_samples = int(round(20.0 * fs_eff))  # ≤20 s
    if x.size > max_samples:
        start = (x.size - max_samples) // 2
        x = x[start : start + max_samples]
    if x.size < 32:
        return empty

    try:
        from PyEMD import EMD

        emd = EMD(max_imf=4)
        imfs = np.asarray(emd.emd(x, max_imf=4), dtype=float)
    except Exception:
        return empty
    if imfs.ndim != 2 or imfs.shape[0] < 1:
        return empty

    # Last row is often the residue when EMD returns IMFs+residue.
    energies = np.sum(imfs ** 2, axis=1)
    total = float(np.sum(energies))
    if total <= 1e-12:
        return empty
    frac = energies / total
    n = int(frac.size)
    # Treat final component as residue when multiple rows are returned.
    if n == 1:
        e1, e2, e3, eres = float(frac[0]), 0.0, 0.0, 0.0
    elif n == 2:
        e1, e2, e3, eres = float(frac[0]), 0.0, 0.0, float(frac[1])
    elif n == 3:
        e1, e2, e3, eres = float(frac[0]), float(frac[1]), 0.0, float(frac[2])
    else:
        e1, e2, e3, eres = float(frac[0]), float(frac[1]), float(frac[2]), float(np.sum(frac[3:]))

    pos = frac[frac > 1e-12]
    entropy = float(-np.sum(pos * np.log(pos)))
    imf1_std = float(np.std(imfs[0]))
    return {
        f'{prefix}ppg_emd_energy_imf1': e1,
        f'{prefix}ppg_emd_energy_imf2': e2,
        f'{prefix}ppg_emd_energy_imf3': e3,
        f'{prefix}ppg_emd_energy_resid': eres,
        f'{prefix}ppg_emd_energy_entropy': entropy,
        f'{prefix}ppg_emd_imf1_std': imf1_std,
    }


def _frequency_domain_features(signal: np.ndarray, prefix: str, fs: float) -> dict:
    if signal.size < 16:
        return {
            f'{prefix}ppg_dominant_frequency': 0.0,
            f'{prefix}ppg_spectral_entropy': 0.0,
            f'{prefix}ppg_band_power_low': 0.0,
            f'{prefix}ppg_band_power_mid': 0.0,
            f'{prefix}ppg_band_power_high': 0.0,
            f'{prefix}ppg_snr': 0.0,
        }

    centered = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / fs)
    total_power = float(np.sum(spectrum))
    if total_power <= 1e-12:
        return {
            f'{prefix}ppg_dominant_frequency': 0.0,
            f'{prefix}ppg_spectral_entropy': 0.0,
            f'{prefix}ppg_band_power_low': 0.0,
            f'{prefix}ppg_band_power_mid': 0.0,
            f'{prefix}ppg_band_power_high': 0.0,
            f'{prefix}ppg_snr': 0.0,
        }

    norm_spectrum = spectrum / total_power
    dominant_frequency = float(freqs[int(np.argmax(spectrum[1:])) + 1]) if spectrum.size > 1 else 0.0
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
        f'{prefix}ppg_dominant_frequency': dominant_frequency,
        f'{prefix}ppg_spectral_entropy': entropy,
        f'{prefix}ppg_band_power_low': low,
        f'{prefix}ppg_band_power_mid': mid,
        f'{prefix}ppg_band_power_high': high,
        f'{prefix}ppg_snr': snr,
    }


def _periodic_features(signal: np.ndarray, prefix: str, fs: float) -> dict:
    if signal.size < 8:
        return {
            f'{prefix}ppg_autocorr_peak_lag': 0.0,
            f'{prefix}ppg_autocorr_peak_val': 0.0,
            f'{prefix}ppg_heart_rate_bpm': 0.0,
        }

    centered = signal - np.mean(signal)
    autocorr = np.correlate(centered, centered, mode='full')
    autocorr = autocorr[autocorr.size // 2 :]
    if autocorr[0] <= 0:
        return {
            f'{prefix}ppg_autocorr_peak_lag': 0.0,
            f'{prefix}ppg_autocorr_peak_val': 0.0,
            f'{prefix}ppg_heart_rate_bpm': 0.0,
        }

    autocorr = autocorr / autocorr[0]
    min_lag = max(3, signal.size // 200)
    max_lag = max(min_lag + 1, signal.size // 2)
    search = autocorr[min_lag:max_lag]
    if search.size == 0:
        return {
            f'{prefix}ppg_autocorr_peak_lag': 0.0,
            f'{prefix}ppg_autocorr_peak_val': 0.0,
            f'{prefix}ppg_heart_rate_bpm': 0.0,
        }

    peak_lag = float(min_lag + int(np.argmax(search)))
    peak_val = float(autocorr[int(peak_lag)])
    heart_rate_bpm = float(60.0 * fs / peak_lag) if peak_lag > 0 else 0.0
    return {
        f'{prefix}ppg_autocorr_peak_lag': peak_lag,
        f'{prefix}ppg_autocorr_peak_val': peak_val,
        f'{prefix}ppg_heart_rate_bpm': heart_rate_bpm,
    }


def _derivative_features(signal: np.ndarray, prefix: str) -> dict:
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1) if diff1.size > 1 else np.empty(0, dtype=float)
    return {
        f'{prefix}ppg_diff1_mean': float(np.mean(np.abs(diff1))) if diff1.size else 0.0,
        f'{prefix}ppg_diff1_std': float(np.std(diff1)) if diff1.size else 0.0,
        f'{prefix}ppg_diff2_mean': float(np.mean(np.abs(diff2))) if diff2.size else 0.0,
        f'{prefix}ppg_diff2_std': float(np.std(diff2)) if diff2.size else 0.0,
    }


def _segment_features(signal: np.ndarray, prefix: str) -> dict:
    midpoint = signal.size // 2
    if midpoint == 0:
        return {
            f'{prefix}ppg_half_mean_diff': 0.0,
            f'{prefix}ppg_half_std_ratio': 0.0,
            f'{prefix}ppg_edge_mean_diff': 0.0,
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
        f'{prefix}ppg_half_mean_diff': float(np.mean(second_half) - np.mean(first_half)),
        f'{prefix}ppg_half_std_ratio': half_std_ratio,
        f'{prefix}ppg_edge_mean_diff': float(end_mean - start_mean),
    }


def _quality_features(signal: np.ndarray, prefix: str, fs: float) -> dict:
    """Quality indicators used as model features without discarding signal data."""
    values = np.asarray(signal, dtype=float).reshape(-1)

    # Five non-overlapping 2 s checks for a typical 10 s MUST recording.
    window_samples = max(8, int(round(2.0 * float(fs))))
    window_scores = [
        ppg_ssqi(values[start : start + window_samples])
        for start in range(0, values.size - window_samples + 1, window_samples)
    ]
    accepted_fraction = (
        float(np.mean(np.asarray(window_scores) > 0.3)) if window_scores else 0.0
    )
    template_window_scores = [
        ppg_template_sqi(values[start : start + window_samples], fs)
        for start in range(0, values.size - window_samples + 1, window_samples)
    ]
    template_accepted_fraction = (
        float(np.mean(np.asarray(template_window_scores) > 0.85))
        if template_window_scores
        else 0.0
    )

    peak_regularity = 0.0
    noise_ratio = 0.0
    try:
        from scipy.signal import butter, find_peaks, resample, sosfiltfilt

        # Quality scoring does not need the native 2175 Hz resolution.
        quality_fs = min(float(fs), 100.0)
        n_quality = max(16, int(round(values.size * quality_fs / float(fs))))
        quality_signal = (
            resample(values, n_quality) if n_quality < values.size else values.copy()
        )
        centered = quality_signal - np.mean(quality_signal)
        sos = butter(3, [0.5, 8.0], btype='bandpass', fs=quality_fs, output='sos')
        pulse_band = sosfiltfilt(sos, centered)
        peaks, _ = find_peaks(
            pulse_band,
            distance=max(3, int(round(0.3 * quality_fs))),
            prominence=max(0.15 * float(np.std(pulse_band)), 1e-9),
        )
        if peaks.size >= 3:
            ibi = np.diff(peaks).astype(float) / quality_fs
            ibi_mean = float(np.mean(ibi))
            if ibi_mean > 1e-9:
                peak_regularity = float(1.0 / (1.0 + np.std(ibi) / ibi_mean))

        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        freqs = np.fft.rfftfreq(centered.size, d=1.0 / quality_fs)
        pulse_power = float(np.sum(spectrum[(freqs >= 0.5) & (freqs < 8.0)]))
        noise_power = float(np.sum(spectrum[(freqs >= 8.0) & (freqs <= 20.0)]))
        relevant_power = pulse_power + noise_power
        if relevant_power > 1e-12:
            noise_ratio = noise_power / relevant_power
    except Exception:
        # Keep feature extraction usable without scipy.
        smooth = _moving_average(values, max(3, int(round(0.05 * float(fs)))))
        total_var = float(np.var(values))
        if total_var > 1e-12:
            noise_ratio = float(np.var(values - smooth) / total_var)

    return {
        f'{prefix}ppg_ssqi': ppg_ssqi(values),
        f'{prefix}ppg_kurtosis': ppg_kurtosis(values),
        f'{prefix}ppg_ksqi': ppg_ksqi(values),
        f'{prefix}ppg_skew_kurt_sqi': ppg_skew_kurt_sqi(values),
        f'{prefix}ppg_template_sqi': ppg_template_sqi(values, fs),
        f'{prefix}ppg_template_kurt_sqi': ppg_template_kurt_sqi(values, fs),
        f'{prefix}ppg_peak_regularity': peak_regularity,
        f'{prefix}ppg_noise_ratio': noise_ratio,
        f'{prefix}ppg_accepted_window_fraction': accepted_fraction,
        f'{prefix}ppg_template_accepted_fraction': template_accepted_fraction,
    }


def extract_ppg_features(
    signal: np.ndarray,
    fs: float = DEFAULT_PPG_SAMPLING_RATE,
    feature_groups: tuple[str, ...] = DEFAULT_PPG_FEATURE_GROUPS,
) -> dict:
    """Extract hand-crafted PPG features from a 1-D waveform."""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size == 0:
        raise ValueError('empty signal')

    enabled = set(feature_groups)
    unknown = enabled.difference({'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8'})
    if unknown:
        raise ValueError(f'Unknown PPG feature groups: {sorted(unknown)}')

    features = {}
    if 'g1' in enabled:
        features.update(_distribution_features(signal, prefix='g1_'))
    if 'g2' in enabled:
        features.update(_derivative_features(signal, prefix='g2_'))
    if 'g3' in enabled:
        features.update(_pulse_features(signal, prefix='g3_'))
    if 'g4' in enabled:
        features.update(_frequency_domain_features(signal, prefix='g4_', fs=fs))
    if 'g5' in enabled:
        features.update(_periodic_features(signal, prefix='g5_', fs=fs))
    if 'g6' in enabled:
        features.update(_segment_features(signal, prefix='g6_'))
    if 'g7' in enabled:
        features.update(_quality_features(signal, prefix='g7_', fs=fs))
    if 'g8' in enabled:
        features.update(_emd_features(signal, prefix='g8_', fs=fs))
    return features


def _prefix_wavelength_features(features: dict, tag: str) -> dict:
    renamed = {}
    for key, value in features.items():
        group, rest = key.split('_', 1) if '_' in key else (key, '')
        renamed[f'{group}_{tag}_{rest}' if rest else f'{tag}_{key}'] = value
    return renamed


def extract_channel_features(
    signal: np.ndarray,
    fs: float,
    feature_groups: tuple[str, ...],
    csv_column: str | Sequence[str] | None = None,
) -> dict:
    """Extract features from a 1-D or (C, T) waveform; prefix when C > 1."""
    values = np.asarray(signal, dtype=float)
    if values.ndim == 1:
        return extract_ppg_features(values, fs=fs, feature_groups=feature_groups)
    if values.ndim != 2:
        raise ValueError(f'PPG signal must be 1-D or (C, T), got shape {values.shape}')
    names = csv_column_names(csv_column) or tuple(
        f'ch{index}' for index in range(values.shape[0])
    )
    merged = {}
    for index, channel in enumerate(values):
        name = names[index] if index < len(names) else f'ch{index}'
        merged.update(
            _prefix_wavelength_features(
                extract_ppg_features(channel, fs=fs, feature_groups=feature_groups),
                csv_column_tag(name),
            )
        )
    return merged


def extract_raw_ppg(
    signal_path: str | Path,
    fs: float = DEFAULT_PPG_SAMPLING_RATE,
    trim_fractions: tuple[float, float] = DEFAULT_PPG_TRIM_FRACTIONS,
    filter_name: str = DEFAULT_PPG_FILTER,
    target_fs: float | None = DEFAULT_PPG_TARGET_FS,
    best_segment_seconds: float | None = DEFAULT_PPG_BEST_SEGMENT_SECONDS,
    feature_groups: tuple[str, ...] = DEFAULT_PPG_FEATURE_GROUPS,
    sqa_threshold: float | None = None,
    sqa_window_seconds: float = PPG_SQA_WINDOW_SECONDS,
    sqa_method: str = 'template',
    center_window_seconds: float | None = None,
    csv_column: str | Sequence[str] | None = None,
    normalize: str = DEFAULT_PPG_NORMALIZE,
) -> dict | None:
    signal = read_signal_waveform(
        Path(signal_path),
        trim_fractions=trim_fractions,
        fs=fs,
        filter_name=filter_name,
        target_fs=target_fs,
        center_window_seconds=center_window_seconds,
        csv_column=csv_column,
        normalize=normalize,
    )
    fs_eff = effective_sampling_rate(fs, target_fs)
    signal = select_best_segment(
        signal,
        fs=fs_eff,
        window_seconds=best_segment_seconds,
        sqa_method=sqa_method,
    )
    signal = apply_sqa_mask(
        signal,
        fs=fs_eff,
        threshold=sqa_threshold,
        window_seconds=sqa_window_seconds,
        sqa_method=sqa_method,
    )
    if signal is None or signal.size == 0:
        return None
    return extract_channel_features(
        signal, fs=fs_eff, feature_groups=feature_groups, csv_column=csv_column
    )


def segment_signal(signal: np.ndarray, fs: float, window_seconds: float) -> list[np.ndarray]:
    """Non-overlapping windows; trailing remainder shorter than one window is dropped."""
    values = np.asarray(signal, dtype=float)
    window_samples = int(round(float(window_seconds) * float(fs)))
    if window_samples < 1:
        raise ValueError(f'window_seconds={window_seconds} yields <1 sample at fs={fs}')
    if values.ndim == 1:
        values = values.reshape(-1)
        if values.size == 0:
            return []
        if values.size < window_samples:
            return [values]
        n_windows = values.size // window_samples
        return [
            values[i * window_samples : (i + 1) * window_samples]
            for i in range(n_windows)
        ]
    if values.ndim != 2:
        raise ValueError(f'PPG signal must be 1-D or (C, T), got shape {values.shape}')
    length = values.shape[1]
    if length == 0:
        return []
    if length < window_samples:
        return [values]
    n_windows = length // window_samples
    return [
        values[:, i * window_samples : (i + 1) * window_samples]
        for i in range(n_windows)
    ]


def extract_raw_ppg_windows(
    signal_path: str | Path,
    fs: float,
    window_seconds: float,
    trim_fractions: tuple[float, float] = DEFAULT_PPG_TRIM_FRACTIONS,
    filter_name: str = DEFAULT_PPG_FILTER,
    target_fs: float | None = DEFAULT_PPG_TARGET_FS,
    best_segment_seconds: float | None = DEFAULT_PPG_BEST_SEGMENT_SECONDS,
    feature_groups: tuple[str, ...] = DEFAULT_PPG_FEATURE_GROUPS,
    sqa_threshold: float | None = None,
    sqa_method: str = 'template',
    center_window_seconds: float | None = None,
    csv_column: str | Sequence[str] | None = None,
    normalize: str = DEFAULT_PPG_NORMALIZE,
) -> list[dict]:
    # Normalize per window (not the full recording) so each segment is unit-scale.
    signal = read_signal_waveform(
        Path(signal_path),
        trim_fractions=trim_fractions,
        fs=fs,
        filter_name=filter_name,
        target_fs=target_fs,
        center_window_seconds=center_window_seconds,
        csv_column=csv_column,
        normalize='none',
    )
    fs_eff = effective_sampling_rate(fs, target_fs)
    signal = select_best_segment(
        signal,
        fs=fs_eff,
        window_seconds=best_segment_seconds,
        sqa_method=sqa_method,
    )
    features = []
    for window in segment_signal(signal, fs=fs_eff, window_seconds=window_seconds):
        window = apply_ppg_normalize(window, normalize=normalize)
        if sqa_threshold is not None:
            if window.ndim == 1:
                if ppg_quality_score(window, fs_eff, method=sqa_method) <= float(
                    sqa_threshold
                ):
                    continue
            elif any(
                ppg_quality_score(channel, fs_eff, method=sqa_method)
                <= float(sqa_threshold)
                for channel in window
            ):
                continue
        features.append(
            extract_channel_features(
                window, fs=fs_eff, feature_groups=feature_groups, csv_column=csv_column
            )
        )
    return features
