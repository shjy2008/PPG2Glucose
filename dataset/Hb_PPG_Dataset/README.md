# A Four-Wavelength Photoplethysmogram dataset for non-invasive hemoglobin assessment

## Overview

Welcome to the Hb-PPG dataset repository. This dataset is designed to study physiological correlations of PPG signals within different wavelengths, and is eligible for in-depth analysis of relationships between PPG signals and Hb. Hb-PPG also supports conducting joint analysis of multiple physiological parameters including body blood glucose, blood pressure and improving reliability of non-invasive Hb measuring devices. 

## Contents

- [A Four-Wavelength Photoplethysmogram dataset for non-invasive hemoglobin assessment](#a-four-wavelength-photoplethysmogram-dataset-for-non-invasive-hemoglobin-assessment)
  - [Overview](#overview)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Dataset Description](#dataset-description)
  - [Data Structure](#data-structure)
  - [Suggested Usage](#suggested-usage)
  - [Citation](#citation)
  - [Contact](#contact)

## Introduction

Hemoglobin (Hb) is the major protein in blood erythrocytes and is responsible for transporting oxygen from the lungs to tissues and organs throughout the body. Photoplethysmography (PPG) stands out as a research hotspot in the field of non-invasive Hb detection due to its easy operation, low cost, stable performance. Multi-wavelength PPG offers chances to capture more hemodynamic changes by using different light wavelengths penetrating tissues at different depths, reflecting more comprehensively changes in blood components.

The dataset includes four-wavelength (660 nm, 730 nm, 850 nm and 940 nm) fingertip PPG signal collected from 252 adult subjects, aged 21–90 years, with females accounted for 56.7%. Hb and blood glucose was acquired from venous blood samples. The dataset also records brachial artery systolic and diastolic blood pressure values and some basic physiological information of subjects including height and weight.

## Dataset Description

The Hb-PPG dataset comprises four-wavelength (660 nm, 730 nm, 850 nm and 940 nm) PPG signal collected from 252 participants. Data was recorded across regulated sessions. The dataset includes:

- **Four-Wavelength PPG Signals**: PPG signals (660 nm, 730 nm, 850 nm, 940 nm) recorded from the left index fingertip by the devised sensor synchronously. Sample frequency: 200 Hz, recording length: 60 seconds.
- **Hemoglobin Concentration**: Hemoglobin value acquired from venous blood samples by HemoCue Hb 201+ hemoglobin analyzer (HemoCue AB, Ängelholm, Sweden).
- **Blood Glucose Concentration**: Blood glucose collected from venous blood samples by auto hematology analyzer BC-6100 (Mindray, Co., Ltd., Shenzhen, China).
- **Blood Pressure**: Systolic (SBP) and diastolic (DBP) blood pressure recorded from brachial artery of participants. Measured by Omron U724J (Omron Corp., Kyoto, Japan) electronic upper arm sphygmomanometer.
- **Age, Height and Weight**: Age, height and weight of some participants.

## Data Structure

Each subject signal was co-deposited as .mat and .csv format for ease of access, respectively. The .mat and .csv files were named by *subject number* (ID), for example, subject ID1 was named as *1.mat* or *1.csv*.

1) .mat file structure: Each .mat file was named by *subject ID*, 
representing one subject PPG data, it contains a structure named "PPGdata" which includes four fields: nm_660, nm_730, nm_850, and nm_940. Each field is a column vector, these fields represent 660 nm, 730 nm, 850 nm and 940 nm wavelength PPG signal, respectively.; 

2) .csv file structure: Each .csv file was named by *subject ID*, and the 660 nm, 730 nm, 850 nm and 940 nm wavelength PPG were stored in the first to fourth column, respectively.

The Hemoglobin, blood glucose concentration, blood pressure and gender, age, height and weight of subjects were saved in "*subject information.xlsx*" table. Subject ID corresponds to the names of *.mat* and *.csv* files. Due to finger movement during recording, some PPG segments were unable to discern cycle beats and were removed, thus these PPG length were less than 60 seconds. "Signal length" column represents the four-wavelength PPG length in seconds (e.g., 60 s, 50 s, 45 s).

The dataset file structure is as follows:

```
Hb-PPG-Dataset/
│
├── <data_mat>/
│ ├── <Subject ID>.mat
│ │ └── <PPGdata>
│ │   ├── <nm_660>_field
│ │   ├── <nm_730>_field
│ │   ├── <nm_850>_field
│ │   └── <nm_940>_field
│ └── ...
│
├── <data_csv>/
│ ├── <Subject ID>.csv
│ │ ├── <660nm>_column
│ │ ├── <730nm>_column
│ │ ├── <850nm>_column
│ │ └── <940nm>_column
│ └── ...
│
└── README.md
└── subject information.xlsx
```

## Suggested Usage
Hb-PPG dataset can be used to extracted multiple types of PPG features, including morphological features (e.g., time-, frequency-domain, etc.) and statistical features (e.g., mean, standard deviation, skewness, quartiles and other metrics) to analyze correlations among four-wavelength PPG signals. Based on 
machine learning and deep learning techniques, establish classification, estimation models for hemoglobin, blood glucose or blood pressure or perform joint research of multiple physiological parameters. It provides opportunities to devise methods to help improve the accuracy and reliability of non-invasive hemoglobin testing and health monitoring devices.

The dataset was devised to help develop algorithms for cardiovascular disease monitoring and may also eligible to investigate brain-heart associations, especially four-wavelength PPG and multi-frequency bands. These may also valuable for studies in cardiovascular and neuroscience fields.

## Citation
If you use this repository or any of its components and/or our paper as part of your research, please cite the publication as follows:

Chen, L. Q., Li, S. Y., Liang, Y. B., Chen, Z. C. & Elgendi, M. Figshare, https://doi.org/10.6084/m9.figshare.22256143.v5 (2025).

## Contact
For any questions or suggestions, please reach out to:

Yongbo Liang liangyongbo001@gmail.com
Mohamed Elgendi moe.elgendi@gmail.com