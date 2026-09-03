"""
STEP 2: PERFECT KNN IMPUTATION (UPDATED)
========================================
- Uses all 10 features: Age, Family_Income, BMI, Gender, Race, 
  Insurance_Coverage, Smoking, Physical_Activity_Limit, 
  Self_Reported_Health, Mental_Health_Status
- Train/test split BEFORE imputation
- Scale ONLY numericals, encode categoricals
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

SAVE_DIR = "meps_data"
ARTIFACTS_DIR = "meps_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("=" * 70)
print("STEP 2: PERFECT KNN IMPUTATION (ALL FEATURES)")
print("=" * 70)

# ========== LOAD CLEANED DATA ==========
df = pd.read_csv(os.path.join(SAVE_DIR, "meps_PERFECTLY_CLEANED.csv"))
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# ========== DEFINE COLUMN TYPES (UPDATED) ==========
ID_COL = 'ID'
TARGETS = ['Diabetes', 'Hypertension', 'Heart_Disease']

# UPDATED: Include BMI, Smoking, Physical_Activity_Limit
NUMERICAL_FEATURES = ['Age', 'Family_Income', 'BMI']
CATEGORICAL_FEATURES = ['Gender', 'Race', 'Insurance_Coverage', 'Smoking',
                        'Physical_Activity_Limit', 'Self_Reported_Health',
                        'Mental_Health_Status']

available_numerical = [c for c in NUMERICAL_FEATURES if c in df.columns]
available_categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
available_targets = [c for c in TARGETS if c in df.columns]

print(f"\nNumerical: {available_numerical}")
print(f"Categorical: {available_categorical}")
print(f"Targets: {available_targets}")

# ========== CREATE COHORTS ==========
print("\n" + "=" * 70)
print("CREATING COHORTS")
print("=" * 70)

cohorts = {}
for target in available_targets:
    df_target = df.dropna(subset=[target]).copy()
    cohorts[target] = df_target
    print(f"\n{target}: {len(df_target)} rows")
    print(f"  Distribution: {df_target[target].value_counts().to_dict()}")

# ========== PROCESS EACH COHORT ==========
print("\n" + "=" * 70)
print("KNN IMPUTATION")
print("=" * 70)

def process_cohort(df_cohort, target_name, test_size=0.2, random_state=42):
    print(f"\n{'='*60}")
    print(f"Processing: {target_name}")
    print(f"{'='*60}")
    
    feature_cols = available_numerical + available_categorical
    X = df_cohort[feature_cols].copy()
    y = df_cohort[target_name].copy()
    ids = df_cohort[ID_COL].copy()
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Missing BEFORE imputation:")
    print(X.isnull().sum()[X.isnull().sum() > 0])
    
    # TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    
    # Separate numerical and categorical
    X_train_num = X_train[available_numerical].copy()
    X_train_cat = X_train[available_categorical].copy()
    X_test_num = X_test[available_numerical].copy()
    X_test_cat = X_test[available_categorical].copy()
    
    # Encode categoricals
    label_encoders = {}
    for col in available_categorical:
        le = LabelEncoder()
        combined = pd.concat([X_train_cat[col], X_test_cat[col]], axis=0).astype(str)
        le.fit(combined)
        X_train_cat[col] = le.transform(X_train_cat[col].astype(str))
        X_test_cat[col] = le.transform(X_test_cat[col].astype(str))
        label_encoders[col] = le
        print(f"  Encoded {col}: {len(le.classes_)} categories")
    
    # Scale ONLY numericals
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)
    
    X_train_num_scaled = pd.DataFrame(X_train_num_scaled, columns=available_numerical, index=X_train.index)
    X_test_num_scaled = pd.DataFrame(X_test_num_scaled, columns=available_numerical, index=X_test.index)
    
    print(f"\n  Numericals scaled ✅")
    print(f"  Categoricals NOT scaled ✅")
    
    # Combine
    all_features = available_numerical + available_categorical
    X_train_combined = pd.concat([X_train_num_scaled, X_train_cat], axis=1)[all_features]
    X_test_combined = pd.concat([X_test_num_scaled, X_test_cat], axis=1)[all_features]
    
    # KNN Impute
    print(f"\n  Running KNN (k=5)...")
    knn = KNNImputer(n_neighbors=5, weights='distance')
    X_train_imp = knn.fit_transform(X_train_combined)
    X_test_imp = knn.transform(X_test_combined)
    
    X_train_imp = pd.DataFrame(X_train_imp, columns=all_features, index=X_train.index)
    X_test_imp = pd.DataFrame(X_test_imp, columns=all_features, index=X_test.index)
    
    # Inverse transform
    X_train_num_final = pd.DataFrame(
        scaler.inverse_transform(X_train_imp[available_numerical]),
        columns=available_numerical, index=X_train.index
    )
    X_test_num_final = pd.DataFrame(
        scaler.inverse_transform(X_test_imp[available_numerical]),
        columns=available_numerical, index=X_test.index
    )
    
    X_train_cat_final = X_train_imp[available_categorical].round().astype(int)
    X_test_cat_final = X_test_imp[available_categorical].round().astype(int)
    
    for col in available_categorical:
        le = label_encoders[col]
        n_classes = len(le.classes_)
        X_train_cat_final[col] = X_train_cat_final[col].clip(0, n_classes - 1)
        X_test_cat_final[col] = X_test_cat_final[col].clip(0, n_classes - 1)
        X_train_cat_final[col] = le.inverse_transform(X_train_cat_final[col])
        X_test_cat_final[col] = le.inverse_transform(X_test_cat_final[col])
    
    # Combine final
    X_train_final = pd.concat([X_train_num_final, X_train_cat_final], axis=1)
    X_test_final = pd.concat([X_test_num_final, X_test_cat_final], axis=1)
    
    # Add ID and target
    train_df = X_train_final.copy()
    train_df[ID_COL] = id_train.values
    train_df[target_name] = y_train.values
    
    test_df = X_test_final.copy()
    test_df[ID_COL] = id_test.values
    test_df[target_name] = y_test.values
    
    print(f"\n  ✅ Train missing: {train_df.isnull().sum().sum()}")
    print(f"  ✅ Test missing: {test_df.isnull().sum().sum()}")
    
    # Save artifacts
    artifacts = {
        'label_encoders': label_encoders,
        'scaler': scaler,
        'knn_imputer': knn,
        'feature_cols': all_features
    }
    joblib.dump(artifacts, os.path.join(ARTIFACTS_DIR, f"{target_name.lower()}_artifacts.pkl"))
    
    return train_df, test_df

# Process all cohorts
results = {}
for target in available_targets:
    train_df, test_df = process_cohort(cohorts[target], target)
    results[target] = {'train': train_df, 'test': test_df}
    
    train_path = os.path.join(SAVE_DIR, f"meps_{target.lower()}_train.csv")
    test_path = os.path.join(SAVE_DIR, f"meps_{target.lower()}_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\n  💾 Saved train ({len(train_df)}) and test ({len(test_df)})")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
for target in available_targets:
    print(f"{target}: Train={len(results[target]['train'])}, Test={len(results[target]['test'])}")