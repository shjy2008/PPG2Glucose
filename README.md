# Data Leakage in PPG-based Non-invasive Blood Glucose Prediction

Code for the paper **Data Leakage in PPG-based Non-invasive Blood Glucose Prediction**.

This repository trains three PPG-to-glucose models under a shared pipeline and compares five evaluation protocols, from leaky window-level splits to subject-independent and cross-dataset tests.

| Model | Role | Entry point |
| --- | --- | --- |
| **TinyML** | 1D ResNet-34 on raw PPG ([Zeynali et al., 2025](https://doi.org/10.1038/s41598-024-84265-8)) | `train_resnet.py` |
| **CBHFF** | Multi-view attention + cascaded BiLSTM ([Ali et al., 2025](https://ieeexplore.ieee.org/document/10684156)) | `train_cbhff.py` |
| **CatBoost** | Handcrafted PPG features (+ demographics) | `train_catboost.py` |

Glucose targets are in **mmol/L**. MUST and VitalDB labels stored in mg/dL are converted with \(18\,\mathrm{mg/dL} = 1\,\mathrm{mmol/L}\).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python **3.10+** is required. Training defaults to CPU (`USE_GPU = False` in `config.py`). For CUDA, set `USE_GPU = True` in that file. The paper reports CPU results; deep-learning runs are much faster on GPU and may differ slightly from the tables because of non-determinism.

Place datasets under `dataset/` as described below. Feature and waveform caches go to `feature_cache/` and `waveform_cache/` (gitignored). Change preprocessing in `config.py` and pass `--refresh-cache` once so caches are rebuilt.

## Datasets

The CLI name `southwest` is the **IPS-PPG** cohort in the paper.

| Paper name | CLI `--dataset` | Access | Sampling rate | Notes |
| --- | --- | --- | --- | --- |
| VitalDB | `vitaldb` | Public ([vitaldb.net](https://vitaldb.net), [Lee et al., 2022](https://doi.org/10.1038/s41597-022-01411-5)) | 500 Hz | Download with `download_vitaldb.py` |
| MUST | `MUST` | Public ([Mendeley Data](https://data.mendeley.com/datasets/37pm7jk7jn/3), [Kermani & Esmaeili, 2023](https://doi.org/10.17632/37pm7jk7jn.3)) | 2175 Hz | 23 subjects, 67 sessions |
| IPS-PPG | `southwest` | Private clinical cohort | 200 Hz | Fasting PPG + glucose; ethical approval No. 2022-SCI-007 |

IPS-PPG should not be redistributed unless your ethics approval allows it. Public-dataset experiments (MUST, VitalDB, and MUST-involved OOD) do not require it.

### MUST

Download the archive and extract so the tree is:

```text
dataset/MUST/
  Labels/label_XX_XXXX.csv
  RawData/signal_XX_XXXX.csv
```

This loader looks for lowercase `label_*.csv` and `signal_*.csv`. If the release uses `Label_*.csv`, rename the prefix. Keep the CSV files; `.mat` copies are unused.

### VitalDB

Each intraoperative lab glucose (`gluc`) is paired with a 16-minute PPG window (`SNUADC/PLETH`) centered on the draw. Training then **center-crops 60 s** by default (`--ppg-seconds 60`).

```bash
python download_vitaldb.py --output-dir dataset/vitaldb_16min --window-seconds 960
```

The download is large and resumable (Ctrl+C is safe). A typical extract is about 7,300 paired recordings from about 3,100 cases; the exact count depends on which tracks are available from the API. Override the directory with `VITALDB_16MIN_DIR` if you store the files elsewhere.

### IPS-PPG (`southwest`)

Expected layout (one fasting recording per participant):

```text
dataset/southwest/
  bg_re.xlsx          # row 2 = header; columns include subject_ID, demographics, Diabetes (mmol/L)
  ppg/{subject_ID}.txt
```

The first 10% of each PPG file is trimmed (`config.PPG_TRIM_FRACTIONS`). Two metadata rows have no matching waveform, so training uses 258 recordings.

## Evaluation protocols

All within-dataset runs use **5-fold × 5 seeds** (`42, 52, 62, 72, 82` in `config.DEFAULT_SEEDS`). Metrics are mean ± std over the 25 folds, plus Clarke Error Grid plots (`clarke_error_grid.png`, `seg_error_grid.png`).

| Paper protocol | What it tests | Flags (add to the train command) |
| --- | --- | --- |
| **RTTS** | Random split of PPG *windows* (same recording can appear in train and test) | `--sample-wise --window-seconds W` |
| **STTS** | Same windows, but all windows of a subject stay on one side | `--window-seconds W` (CatBoost: also `--subject-wise`) |
| **PAS** | Hold out a later *session* from subjects with 2+ recordings; single-session subjects stay in train | `--pas` (MUST and VitalDB only; not defined for IPS-PPG) |
| **LSPO** | Subject-independent CV, one sample per recording (no extra windowing) | default; CatBoost: `--subject-wise` |
| **OOD** | Train on one dataset, test on another | `train_*_cross.py` |

Window length **W**:

- MUST: `1` (native recordings are ~10 s)
- VitalDB and IPS-PPG: `10` (VitalDB is first cropped to 60 s, then split into 10 s windows)

PAS, LSPO, and OOD do **not** add `--window-seconds`.

TinyML (`train_resnet.py`) and CBHFF default to **PPG only**. CatBoost uses the demographic keys in `config.BASIC_FEATURE_KEYS` (MUST: age, weight; VitalDB and IPS-PPG: age, height, weight, BMI).

On MUST and VitalDB, CatBoost **auto-enables subject-wise CV** whenever a subject has more than one row. Use `--sample-wise` when you intend RTTS.

## Reproduce Table 5 (within-dataset)

Replace `MODEL` with `train_catboost.py`, `train_resnet.py`, or `train_cbhff.py`.

### MUST

```bash
# RTTS
python MODEL --dataset MUST --sample-wise --window-seconds 1

# STTS
python train_catboost.py --dataset MUST --subject-wise --window-seconds 1
python train_resnet.py   --dataset MUST --window-seconds 1
python train_cbhff.py    --dataset MUST --window-seconds 1

# PAS
python MODEL --dataset MUST --pas

# LSPO
python train_catboost.py --dataset MUST --subject-wise
python train_resnet.py   --dataset MUST
python train_cbhff.py    --dataset MUST
```

### VitalDB

```bash
# RTTS
python MODEL --dataset vitaldb --sample-wise --window-seconds 10

# STTS
python train_catboost.py --dataset vitaldb --subject-wise --window-seconds 10
python train_resnet.py   --dataset vitaldb --window-seconds 10
python train_cbhff.py    --dataset vitaldb --window-seconds 10

# PAS
python MODEL --dataset vitaldb --pas

# LSPO
python train_catboost.py --dataset vitaldb --subject-wise
python train_resnet.py   --dataset vitaldb
python train_cbhff.py    --dataset vitaldb
```

### IPS-PPG (`southwest`)

PAS is not applicable (one recording per participant).

```bash
# RTTS
python MODEL --dataset southwest --sample-wise --window-seconds 10

# STTS
python train_catboost.py --dataset southwest --subject-wise --window-seconds 10
python train_resnet.py   --dataset southwest --window-seconds 10
python train_cbhff.py    --dataset southwest --window-seconds 10

# LSPO
python train_catboost.py --dataset southwest --subject-wise
python train_resnet.py   --dataset southwest
python train_cbhff.py    --dataset southwest
```

Deep-learning smoke test (CPU, tiny subset):

```bash
python train_resnet.py --dataset MUST --quick --max-samples 16
```

## Reproduce Table 6 (OOD)

No extra windowing. Defaults train on VitalDB and test on MUST.

```bash
python train_catboost_cross.py --train-dataset vitaldb --test-dataset MUST
python train_resnet_cross.py   --train-dataset vitaldb --test-dataset MUST
python train_cbhff_cross.py    --train-dataset vitaldb --test-dataset MUST

python train_catboost_cross.py --train-dataset southwest --test-dataset MUST
python train_resnet_cross.py   --train-dataset southwest --test-dataset MUST
python train_cbhff_cross.py    --train-dataset southwest --test-dataset MUST

python train_catboost_cross.py --train-dataset vitaldb --test-dataset southwest
python train_resnet_cross.py   --train-dataset vitaldb --test-dataset southwest
python train_cbhff_cross.py    --train-dataset vitaldb --test-dataset southwest

python train_catboost_cross.py --train-dataset southwest --test-dataset vitaldb
python train_resnet_cross.py   --train-dataset southwest --test-dataset vitaldb
python train_cbhff_cross.py    --train-dataset southwest --test-dataset vitaldb
```

MUST is not used as an OOD *source* (too few subjects). Predictions from the five CatBoost seeds (or five DL seeds) are averaged on the target set.

## Preprocessing and CatBoost features

No band-pass filter, downsampling, or signal-quality rejection is applied in the paper configuration (`config.py`).

CatBoost features (39 PPG descriptors; see paper Table 4):

| Group | Contents |
| --- | --- |
| g1 distribution | length, mean, median, std, range, IQR, CV, RMS, abs mean, skew, kurtosis, mean-crossing count |
| g2 derivatives | first- and second-difference mean/std |
| g3 pulse | peaks, IBI statistics, amplitudes, slopes, perfusion index |
| g4 frequency | dominant frequency, spectral entropy, band powers, SNR |
| g5 periodicity | autocorrelation peak lag/value, heart rate |
| g6 segments | half-signal and edge statistics |

MUST CatBoost uses **g3–g6** plus age and weight (ablation in `config.py`). VitalDB and IPS-PPG use **g1–g6** plus age, height, weight, and BMI.

Default CatBoost hyperparameters are in `train_catboost.py` (`iterations=500`, `learning_rate=0.025`, `depth=3`, with `depth=4` on MUST and IPS-PPG). Optional Optuna search: `python tune_catboost.py --dataset MUST`.

TinyML: 25 epochs, batch 32, AdamW, early stopping patience 5, waveforms resampled to length 2048. CBHFF: 30 epochs, batch 32, Adam, same patience and length. Inner 60/20/20 train/val/test is carved inside each outer fold.

## Metrics

Reported in the training log: MAE, RMSE, \(R^2\), threshold accuracy, MARD, Pearson \(r\), Clarke zones A/B/A+B.

Accuracy (paper Table 2): absolute error \(< 0.83\) mmol/L if reference \(\le 5.55\) mmol/L, otherwise relative error \(\le 15\%\).

## Repository layout

```text
config.py                 # sampling rates, feature groups, demographics, seeds, GPU
data_loader.py            # dataset layouts, caches, splits
data_ppg.py               # handcrafted PPG features
download_vitaldb.py       # VitalDB PPG + glucose export
evaluation.py             # regression metrics, Clarke / SEG plots
utils.py                  # RTTS / STTS / PAS / LSPO splitters, repeated CV
train_catboost.py         # CatBoost within-dataset
train_resnet.py           # TinyML 1D ResNet-34
train_cbhff.py            # CBHFF
train_*_cross.py          # OOD trainers
CBHFF/                    # Ali et al. architecture
TinyML_code/resnet.py     # Zeynali et al. 1D ResNet-34
evaluate_models.py        # optional tabular ML comparison (XGBoost, RF, …)
```

## Third-party implementations

- TinyML 1D ResNet-34: [Zeynali et al., Sci. Rep. 2025](https://doi.org/10.1038/s41598-024-84265-8), code at [mahdinet1/BGL-Estmiation-from-PPG-Signal](https://github.com/mahdinet1/BGL-Estmiation-from-PPG-Signal)
- CBHFF: [Ali et al., IEEE JBHI 2025](https://ieeexplore.ieee.org/document/10684156)

## Citation

```bibtex
@inproceedings{ppg_bgl_leakage_2026,
  title     = {Data Leakage in PPG-based Non-invasive Blood Glucose Prediction},
  author    = {Anonymous},
  booktitle = {Submitted to NeurIPS},
  year      = {2026}
}
```

Update the BibTeX entry after the paper is published.
