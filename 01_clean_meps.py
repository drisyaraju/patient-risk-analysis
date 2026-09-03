"""
STEP 1: PERFECT MEPS CLEANING (FIXED - Correct BMI Column)
============================================================
"""

import pandas as pd
import numpy as np
import os

SAVE_DIR = "meps_data"

print("=" * 70)
print("STEP 1: PERFECT MEPS CLEANING (FIXED)")
print("=" * 70)

# Load raw data
df_raw = pd.read_stata(os.path.join(SAVE_DIR, "h224.dta"), convert_categoricals=False)
print(f"Raw shape: {df_raw.shape}")

# Find correct columns
diabetes_cols = [c for c in df_raw.columns if 'DIAB' in c.upper()]
print(f"Diabetes columns: {diabetes_cols}")

bmi_cols = [c for c in df_raw.columns if 'BMI' in c.upper() or 'BMX' in c.upper()]
print(f"BMI columns: {bmi_cols}")

smoking_cols = [c for c in df_raw.columns if 'SMOK' in c.upper()]
print(f"Smoking columns: {smoking_cols}")

activity_cols = [c for c in df_raw.columns if 'PAL' in c.upper() or 'ACT' in c.upper()]
print(f"Activity columns: {activity_cols}")

# Use correct column names
DIABETES_COL = diabetes_cols[0] if diabetes_cols else None
BMI_COL = bmi_cols[0] if bmi_cols else None  # Use first found BMI column
SMOKING_COL = smoking_cols[0] if smoking_cols else None
ACTIVITY_COL = activity_cols[0] if activity_cols else None

print(f"\nUsing:")
print(f"  Diabetes: {DIABETES_COL}")
print(f"  BMI: {BMI_COL}")
print(f"  Smoking: {SMOKING_COL}")
print(f"  Activity: {ACTIVITY_COL}")

# Select columns
COLUMNS_TO_KEEP = {
    'DUPERSID': 'ID',
    'AGE31X': 'Age',
    'SEX': 'Gender',
    'RACEV1X': 'Race',
    'EDUYRDG': 'Education_Years',
    'FAMINC20': 'Family_Income',
    'INSCOV20': 'Insurance_Coverage',
    DIABETES_COL: 'Diabetes',
    'HIBPDX': 'Hypertension',
    'CHDDX': 'Heart_Disease_CHD',
    'MIDX': 'Heart_Attack',
    'STRKDX': 'Stroke',
    SMOKING_COL: 'Smoking',
    ACTIVITY_COL: 'Physical_Activity_Limit',
    BMI_COL: 'BMI',
    'RTHLTH31': 'Self_Reported_Health',
    'MNHLTH31': 'Mental_Health_Status'
}

# Keep only existing columns
available = {k: v for k, v in COLUMNS_TO_KEEP.items() if k and k in df_raw.columns}
print(f"\nSelected {len(available)} columns:")
for old, new in available.items():
    print(f"  {old} → {new}")

df = df_raw[list(available.keys())].copy()
df.rename(columns=available, inplace=True)

print(f"\nSelected data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Clean targets (same as before)
def clean_target(series, name):
    print(f"\n{name}:")
    print(f"  Before: {series.value_counts(dropna=False).to_dict()}")
    
    s = series.astype(str)
    missing_codes = ['-1 INAPPLICABLE', '-7 REFUSED', '-8 DK', '-9 NOT ASCERTAINED',
                     '-15', 'nan', 'NaN', 'None', '-1', '-7', '-8', '-9']
    s = s.replace(missing_codes, np.nan)
    s = s.replace({'1 YES': '1', '2 NO': '2', '1': '1', '2': '2'})
    s = pd.to_numeric(s, errors='coerce')
    s = s.map({1: 1, 2: 0})
    
    print(f"  After: {s.value_counts(dropna=False).to_dict()}")
    return s

for col in ['Diabetes', 'Hypertension', 'Heart_Disease_CHD', 'Heart_Attack', 'Stroke']:
    if col in df.columns:
        df[col] = clean_target(df[col], col)

# Create composite Heart Disease
heart_cols = ['Heart_Disease_CHD', 'Heart_Attack', 'Stroke']
available_heart = [c for c in heart_cols if c in df.columns]
if available_heart:
    df['Heart_Disease'] = df[available_heart].fillna(0).max(axis=1)
    all_nan = df[available_heart].isnull().all(axis=1)
    df.loc[all_nan, 'Heart_Disease'] = np.nan
    
    print(f"\nHeart_Disease: {df['Heart_Disease'].value_counts(dropna=False).to_dict()}")

# Save
output_path = os.path.join(SAVE_DIR, "meps_PERFECTLY_CLEANED_v2.csv")
df.to_csv(output_path, index=False)
print(f"\n✅ SAVED: {output_path}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")