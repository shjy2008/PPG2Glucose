from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from data_ppg import apply_ppg_filter, read_signal_waveform

# MUST RawData is 10-bit ADC counts; the published Figures use volts.
FS = 2175.0
ADC_TO_VOLT = 5.0 / 1024.0

path = Path('dataset/MUST/RawData/signal_01_0001.csv')
raw_adc = read_signal_waveform(path, filter_name='none')
mswf_ac = apply_ppg_filter(raw_adc, fs=FS, filter_name='mswf')

# Stage 1 only: Savitzky–Golay (same 0.05 s window as mswf_filter). Fair overlay
# for "does smoothing cut through the vibrating baseline?"
sw = max(5, int(round(0.05 * FS)))
if sw % 2 == 0:
    sw += 1
sw = min(sw, raw_adc.size if raw_adc.size % 2 == 1 else raw_adc.size - 1)
sg = savgol_filter(raw_adc, window_length=sw, polyorder=2, mode='interp')

t = np.arange(raw_adc.size) / FS
raw_v = raw_adc * ADC_TO_VOLT
sg_v = sg * ADC_TO_VOLT

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t, raw_v, color='C0', lw=0.8)
axes[0].set_ylabel('Voltage (V)')
axes[0].set_title('raw (same scale as MUST fig_01_0001)')
axes[0].set_ylim(2.4, 2.58)

axes[1].plot(t, raw_v, label='raw', alpha=0.5, lw=0.7)
axes[1].plot(t, sg_v, label='stage1 Savitzky–Golay (true smooth)', color='C1', lw=1.0)
axes[1].set_ylabel('Voltage (V)')
axes[1].set_title('fair overlay: smoothing should cut through the noise band')
axes[1].set_ylim(2.4, 2.58)
axes[1].legend(loc='upper right')

axes[2].plot(t, mswf_ac * ADC_TO_VOLT, color='C1', lw=0.9)
axes[2].set_xlabel('Time (sec)')
axes[2].set_ylabel('Voltage (V)')
axes[2].set_title('stage2 MSWF output (baseline removed → AC near 0; used by the pipeline)')
axes[2].axhline(0.0, color='k', lw=0.5, alpha=0.4)

fig.tight_layout()
out = Path('mswf_compare.png')
fig.savefig(out, dpi=150)
print(f'Saved {out.resolve()}')
