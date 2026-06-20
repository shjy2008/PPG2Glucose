import numpy as np
import pandas as pd

from utils import safe_float


def _first_present(row: pd.Series, keys: list[str]):
    for key in keys:
        value = row.get(key)
        if value is not None and not pd.isna(value):
            return value
    return np.nan

def parse_blood_pressure(row: pd.Series):
    bp_str = row.get('Diastolic Blood Pressure(mmHg)')
    if isinstance(bp_str, str) and '/' in bp_str:
        parts = bp_str.split('/')
        if len(parts) >= 2:
            return (safe_float(parts[0]), safe_float(parts[1]))
    return np.nan, np.nan

def _subject_ref(row: pd.Series):
    return _first_present(row, ['subject_ID', 'ID'])


def extract_basic_info(row: pd.Series) -> dict:
    sex_raw = _first_present(row, ['Sex(M/F)', 'Gender'])
    sex_value = str(sex_raw).strip().upper()
    sex = np.nan
    if sex_value in {'M', 'MALE'}:
        sex = 1.0
    elif sex_value in {'F', 'FEMALE'}:
        sex = 0.0

    valid = True
    subject_ref = _subject_ref(row)

    age = safe_float(_first_present(row, ['Age(year)', 'Age']))
    if np.isnan(age):
        print("Invalid age value, subject_ID:", subject_ref)
        valid = False

    height = safe_float(_first_present(row, ['Height(cm)', 'Height']))
    weight = safe_float(_first_present(row, ['Weight(kg)', 'Weight']))
    if np.isnan(height) or np.isnan(weight):
        print("Invalid height/weight value, subject_ID:", subject_ref)
        valid = False
    bmi = weight / ((height / 100.0) ** 2)

    # systolic, diastolic = parse_blood_pressure(row)
    # if np.isnan(systolic) or np.isnan(diastolic):
    #     print("Invalid blood pressure value, subject_ID:", row.get('subject_ID'))
    #     valid = False

    # heart_rate = safe_float(row.get('Heart Rate(b/m)'))
    # if np.isnan(heart_rate):
    #     print("Invalid heart rate value, subject_ID:", row.get('subject_ID'))
    #     valid = False

    if not valid:
        return {}

    return {
        # 'sex': sex,
        'age': age,
        'height': height,
        'weight': weight,
        'bmi': bmi,
        # 'blood_pressure_systolic': systolic,
        # 'blood_pressure_diastolic': diastolic,
        # 'heart_rate': heart_rate,
    }
