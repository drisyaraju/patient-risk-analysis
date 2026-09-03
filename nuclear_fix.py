import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("NUCLEAR OPTION: FULL PIPELINE WITH BMI")
print("=" * 70)

# Find everything automatically
root = os.getcwd()
print(f"Working dir: {root}")

# Find h224.dta
dta_files = glob.glob(os.path.join(root, "**", "h224.dta"), recursive=True)
if not dta_files:
    print("❌ h224.dta not found!")
    exit()
dta_path = dta_files[0]
print(f"📁 Found raw .dta: {dta_path}")

# Find cleaned CSV
csv_pattern = os.path.join(root, "**", "meps_PERFECTLY_CLEANED_v2*.csv")
csv_files = glob.glob(csv_pattern, recursive=True)
csv_files = [f for f in csv_files if 'WITH_BMI' not in f and 'WITH_ADULT' not in f]
if not csv_files:
    print("❌ Cleaned CSV not found!")
    exit()
cleaned_path = csv_files[0]
print(f"📁 Found cleaned CSV: {cleaned_path}")

# Step 1: Extract BMI
print("\n📦 Extracting BMI from raw .dta...")
raw = pd.read_stata(dta_path, convert_categoricals=False)
bmi = pd.to_numeric(raw['ADBMI42'], errors='coerce')
bmi = bmi.replace([-1, -7, -8, -9, -15, -2], np.nan)
print(f"   Real BMI values: {bmi.notna().sum():,}")

# Step 2: Load cleaned, add BMI
print("\n📦 Adding BMI to cleaned data...")
df = pd.read_csv(cleaned_path)
df['BMI'] = bmi.values
print(f"   Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")

# Save
save_dir = os.path.dirname(cleaned_path)
bmi_path = os.path.join(save_dir, "meps_PERFECTLY_CLEANED_v2_WITH_BMI.csv")
df.to_csv(bmi_path, index=False)
print(f"   💾 Saved: {bmi_path}")

# Step 3: KNN Pipeline
print("\n📦 Running KNN Imputation...")

ID_COL = 'ID'
TARGETS = ['Diabetes', 'Hypertension', 'Heart_Disease']
NUMERICAL = ['Age', 'Family_Income', 'BMI']
CATEGORICAL = ['Gender', 'Race', 'Insurance_Coverage', 'Self_Reported_Health', 'Mental_Health_Status']

available_num = [c for c in NUMERICAL if c in df.columns]
available_cat = [c for c in CATEGORICAL if c in df.columns]
available_targets = [c for c in TARGETS if c in df.columns]

print(f"   Numerical: {available_num}")
print(f"   Categorical: {available_cat}")
print(f"   Targets: {available_targets}")

for target in available_targets:
    print(f"\n{'='*60}")
    print(f"🎯 {target}")
    print(f"{'='*60}")
    
    cohort = df.dropna(subset=[target]).copy()
    features = available_num + available_cat
    X = cohort[features].copy()
    y = cohort[target].copy()
    ids = cohort[ID_COL].copy()
    
    print(f"   Cohort: {len(cohort):,}")
    print(f"   Features: {features}")
    
    # Train-test split
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42, stratify=y
    )
    
    # Encode categoricals
    encoders = {}
    for col in available_cat:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
    
    # Scale numericals
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[available_num])
    X_test_num = scaler.transform(X_test[available_num])
    X_train[available_num] = X_train_num
    X_test[available_num] = X_test_num
    
    # KNN Impute
    knn = KNNImputer(n_neighbors=5, weights='distance')
    X_train_imp = knn.fit_transform(X_train)
    X_test_imp = knn.transform(X_test)
    
    X_train_imp = pd.DataFrame(X_train_imp, columns=features, index=X_train.index)
    X_test_imp = pd.DataFrame(X_test_imp, columns=features, index=X_test.index)
    
    # Inverse transform
    X_train_imp[available_num] = scaler.inverse_transform(X_train_imp[available_num])
    X_test_imp[available_num] = scaler.inverse_transform(X_test_imp[available_num])
    
    for col in available_cat:
        le = encoders[col]
        X_train_imp[col] = X_train_imp[col].round().astype(int).clip(0, len(le.classes_)-1)
        X_test_imp[col] = X_test_imp[col].round().astype(int).clip(0, len(le.classes_)-1)
        X_train_imp[col] = le.inverse_transform(X_train_imp[col])
        X_test_imp[col] = le.inverse_transform(X_test_imp[col])
    
    # Add back ID and target
    train_df = X_train_imp.copy()
    train_df[ID_COL] = id_train.values
    train_df[target] = y_train.values
    
    test_df = X_test_imp.copy()
    test_df[ID_COL] = id_test.values
    test_df[target] = y_test.values
    
    # Save
    train_path = os.path.join(save_dir, f"meps_{target.lower()}_train.csv")
    test_path = os.path.join(save_dir, f"meps_{target.lower()}_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"   ✅ Train: {len(train_df):,} | BMI? {'BMI' in train_df.columns}")
    print(f"   ✅ Test: {len(test_df):,}")
    print(f"   💾 Saved to {save_dir}")

print("\n" + "=" * 70)
print("🎉 DONE! All train/test files now include BMI.")
print("=" * 70)