# Data Leakage in PPG-based Non-invasive Blood Glucose Prediction

Code for the paper. Three models, five evaluation protocols, three datasets.


| Model                                                                         | Script              | Input                          |
| ----------------------------------------------------------------------------- | ------------------- | ------------------------------ |
| ResNet34 ([Zeynali et al., 2025](https://doi.org/10.1038/s41598-024-84265-8)) | `train_resnet.py`   | raw PPG                        |
| CBHFF ([Ali et al., 2025](https://ieeexplore.ieee.org/document/10684156))     | `train_cbhff.py`    | raw PPG                        |
| CatBoost                                                                      | `train_catboost.py` | handcrafted PPG + demographics |


```bash
pip install -r requirements.txt   # Python 3.10+
```

CPU by default. For GPU, set `USE_GPU = True` in `config.py`.

## Data

- **VitalDB** — `python download_vitaldb.py --output-dir dataset/vitaldb --window-seconds 60`
- **MUST** — [Mendeley](https://data.mendeley.com/datasets/37pm7jk7jn/3) into `dataset/MUST/`
- **IPS-PPG** — private; `--dataset IPS`

## Table 5 (within-dataset)

Same order as the paper: dataset → ResNet34, CBHFF, CatBoost → RTTS, RTTS (subject-wise), PAS, LSPO.

### VitalDB

```bash
# ResNet34
python train_resnet.py   --dataset vitaldb --sample-wise --window-seconds 10   # RTTS
python train_resnet.py   --dataset vitaldb --window-seconds 10                # STTS
python train_resnet.py   --dataset vitaldb --pas                              # PAS
python train_resnet.py   --dataset vitaldb                                    # LSPO

# CBHFF
python train_cbhff.py    --dataset vitaldb --sample-wise --window-seconds 10   # RTTS
python train_cbhff.py    --dataset vitaldb --window-seconds 10                # STTS
python train_cbhff.py    --dataset vitaldb --pas                              # PAS
python train_cbhff.py    --dataset vitaldb                                    # LSPO

# CatBoost
python train_catboost.py --dataset vitaldb --sample-wise --window-seconds 10   # RTTS
python train_catboost.py --dataset vitaldb --subject-wise --window-seconds 10 # STTS
python train_catboost.py --dataset vitaldb --pas                              # PAS
python train_catboost.py --dataset vitaldb --subject-wise                     # LSPO
```

### MUST

```bash
# ResNet34
python train_resnet.py   --dataset MUST --sample-wise --window-seconds 1   # RTTS
python train_resnet.py   --dataset MUST --window-seconds 1                # STTS
python train_resnet.py   --dataset MUST --pas                             # PAS
python train_resnet.py   --dataset MUST                                   # LSPO

# CBHFF
python train_cbhff.py    --dataset MUST --sample-wise --window-seconds 1   # RTTS
python train_cbhff.py    --dataset MUST --window-seconds 1                # STTS
python train_cbhff.py    --dataset MUST --pas                             # PAS
python train_cbhff.py    --dataset MUST                                   # LSPO

# CatBoost
python train_catboost.py --dataset MUST --sample-wise --window-seconds 1   # RTTS
python train_catboost.py --dataset MUST --subject-wise --window-seconds 1 # STTS
python train_catboost.py --dataset MUST --pas                             # PAS
python train_catboost.py --dataset MUST --subject-wise                    # LSPO
```

### IPS-PPG

PAS is `/` in the paper (one recording per person).

```bash
# ResNet34
python train_resnet.py   --dataset IPS --sample-wise --window-seconds 10   # RTTS
python train_resnet.py   --dataset IPS --window-seconds 10                # STTS
python train_resnet.py   --dataset IPS                                    # LSPO

# CBHFF
python train_cbhff.py    --dataset IPS --sample-wise --window-seconds 10   # RTTS
python train_cbhff.py    --dataset IPS --window-seconds 10                # STTS
python train_cbhff.py    --dataset IPS                                    # LSPO

# CatBoost
python train_catboost.py --dataset IPS --sample-wise --window-seconds 10   # RTTS
python train_catboost.py --dataset IPS --subject-wise --window-seconds 10 # STTS
python train_catboost.py --dataset IPS --subject-wise                     # LSPO
```

## Table 6 (OOD)

Order is ResNet34, CBHFF, CatBoost.

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

# MUST → IPS-PPG
python train_resnet_cross.py   --train-dataset MUST --test-dataset IPS
python train_cbhff_cross.py    --train-dataset MUST --test-dataset IPS
python train_catboost_cross.py --train-dataset MUST --test-dataset IPS

# MUST → VitalDB
python train_resnet_cross.py   --train-dataset MUST --test-dataset vitaldb
python train_cbhff_cross.py    --train-dataset MUST --test-dataset vitaldb
python train_catboost_cross.py --train-dataset MUST --test-dataset vitaldb
```

Smoke test: `python train_resnet.py --dataset MUST --quick --max-samples 16`