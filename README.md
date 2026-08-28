# Data Leakage in PPG-based Non-invasive Blood Glucose Prediction

Code for the paper. Three models, five evaluation protocols, three datasets.

| Model | Script | Input |
| --- | --- | --- |
| TinyML ([Zeynali et al., 2025](https://doi.org/10.1038/s41598-024-84265-8)) | `train_resnet.py` | raw PPG |
| CBHFF ([Ali et al., 2025](https://ieeexplore.ieee.org/document/10684156)) | `train_cbhff.py` | raw PPG |
| CatBoost | `train_catboost.py` | handcrafted PPG + demographics |

Targets are **mmol/L** (MUST / VitalDB mg/dL labels are divided by 18).

```bash
pip install -r requirements.txt   # Python 3.10+
```

GPU: set `USE_GPU = True` in `config.py` (default CPU). Table 5 is 5-fold × 5 seeds (`42, 52, 62, 72, 82`). After changing `config.py`, pass `--refresh-cache`.

## Data

IPS-PPG is `--dataset IPS` (private). MUST and VitalDB cover the public experiments.

| Paper | `--dataset` | Get it |
| --- | --- | --- |
| MUST | `MUST` | [Mendeley](https://data.mendeley.com/datasets/37pm7jk7jn/3) → `dataset/MUST/{Labels,RawData}/` (`label_*.csv`, `signal_*.csv`) |
| VitalDB | `vitaldb` | `python download_vitaldb.py --output-dir dataset/vitaldb_16min --window-seconds 960` (resumable; train-time crop is 60 s) |
| IPS-PPG | `IPS` | `dataset/IPS/bg_re.xlsx` + `ppg/{id}.txt` |

## Protocols

| Paper | Flags |
| --- | --- |
| RTTS | `--sample-wise --window-seconds W` |
| STTS | `--window-seconds W` (CatBoost: also `--subject-wise`) |
| PAS | `--pas` (MUST and VitalDB only) |
| LSPO | CatBoost: `--subject-wise`; TinyML / CBHFF: default |
| OOD | `train_*_cross.py --train-dataset A --test-dataset B` |

`W=1` on MUST, `W=10` on VitalDB and IPS-PPG. PAS / LSPO / OOD do not window.

## Table 5 (within-dataset)

Each block is TinyML, CBHFF, CatBoost — the same order as the paper.

### VitalDB

```bash
# RTTS
python train_resnet.py   --dataset vitaldb --sample-wise --window-seconds 10
python train_cbhff.py    --dataset vitaldb --sample-wise --window-seconds 10
python train_catboost.py --dataset vitaldb --sample-wise --window-seconds 10

# STTS
python train_resnet.py   --dataset vitaldb --window-seconds 10
python train_cbhff.py    --dataset vitaldb --window-seconds 10
python train_catboost.py --dataset vitaldb --subject-wise --window-seconds 10

# PAS
python train_resnet.py   --dataset vitaldb --pas
python train_cbhff.py    --dataset vitaldb --pas
python train_catboost.py --dataset vitaldb --pas

# LSPO
python train_resnet.py   --dataset vitaldb
python train_cbhff.py    --dataset vitaldb
python train_catboost.py --dataset vitaldb --subject-wise
```

### MUST

```bash
# RTTS
python train_resnet.py   --dataset MUST --sample-wise --window-seconds 1
python train_cbhff.py    --dataset MUST --sample-wise --window-seconds 1
python train_catboost.py --dataset MUST --sample-wise --window-seconds 1

# STTS
python train_resnet.py   --dataset MUST --window-seconds 1
python train_cbhff.py    --dataset MUST --window-seconds 1
python train_catboost.py --dataset MUST --subject-wise --window-seconds 1

# PAS
python train_resnet.py   --dataset MUST --pas
python train_cbhff.py    --dataset MUST --pas
python train_catboost.py --dataset MUST --pas

# LSPO
python train_resnet.py   --dataset MUST
python train_cbhff.py    --dataset MUST
python train_catboost.py --dataset MUST --subject-wise
```

### IPS-PPG

PAS is blank in the paper (one recording per person).

```bash
# RTTS
python train_resnet.py   --dataset IPS --sample-wise --window-seconds 10
python train_cbhff.py    --dataset IPS --sample-wise --window-seconds 10
python train_catboost.py --dataset IPS --sample-wise --window-seconds 10

# STTS
python train_resnet.py   --dataset IPS --window-seconds 10
python train_cbhff.py    --dataset IPS --window-seconds 10
python train_catboost.py --dataset IPS --subject-wise --window-seconds 10

# LSPO
python train_resnet.py   --dataset IPS
python train_cbhff.py    --dataset IPS
python train_catboost.py --dataset IPS --subject-wise
```

## Table 6 (OOD)

MUST is not used as a training source. Order is TinyML, CBHFF, CatBoost.

```bash
# VitalDB → MUST
python train_resnet_cross.py   --train-dataset vitaldb --test-dataset MUST
python train_cbhff_cross.py    --train-dataset vitaldb --test-dataset MUST
python train_catboost_cross.py --train-dataset vitaldb --test-dataset MUST

# IPS-PPG → MUST
python train_resnet_cross.py   --train-dataset IPS --test-dataset MUST
python train_cbhff_cross.py    --train-dataset IPS --test-dataset MUST
python train_catboost_cross.py --train-dataset IPS --test-dataset MUST

# VitalDB → IPS-PPG
python train_resnet_cross.py   --train-dataset vitaldb --test-dataset IPS
python train_cbhff_cross.py    --train-dataset vitaldb --test-dataset IPS
python train_catboost_cross.py --train-dataset vitaldb --test-dataset IPS

# IPS-PPG → VitalDB
python train_resnet_cross.py   --train-dataset IPS --test-dataset vitaldb
python train_cbhff_cross.py    --train-dataset IPS --test-dataset vitaldb
python train_catboost_cross.py --train-dataset IPS --test-dataset vitaldb
```

Smoke test: `python train_resnet.py --dataset MUST --quick --max-samples 16`
