"""
08_predict.py — Batch Inference for New Patients
Usage:
    python 08_predict.py --input new_patients.csv --target Diabetes --output predictions.csv

Supports: Diabetes, Hypertension, Heart_Disease
Outputs: Risk probability, risk tier, screening flag
"""

import argparse
import pickle
import os
import numpy as np
import pandas as pd

DATA_DIR = 'meps_data'
ARTIFACT_DIR = os.path.join(DATA_DIR, 'results', 'models_v2')
DROP_COLS = ['ID']

CATEGORICAL_COLS = [
    'Gender', 'Race', 'Insurance_Coverage', 'Smoking',
    'Self_Reported_Health', 'Age_Group', 'BMI_Category',
    'Income_Quartile', 'Risk_Category', 'Insurance_Gender'
]


def add_new_features(df):
    """Must match Phase 6.5 feature engineering exactly."""
    df = df.copy()
    if 'BMI' in df.columns and 'Age' in df.columns:
        df['BMI_x_Age'] = (df['BMI'] * df['Age']).round(1)
    if 'Comorbidity_Count' in df.columns:
        df['Poly_Risk'] = (df['Comorbidity_Count'] >= 2).astype(int)
    if 'Age' in df.columns and 'Self_Reported_Health' in df.columns:
        health_map = {'Excellent': 0, 'Very Good': 1, 'Good': 2, 'Fair': 3, 'Poor': 4}
        sr_numeric = df['Self_Reported_Health'].map(health_map).fillna(2)
        df['Age_x_Health'] = (df['Age'] * sr_numeric).round(0)
    if 'BMI_Category' in df.columns and 'Smoking' in df.columns:
        df['Obese_Smoker'] = ((df['BMI_Category'] == 'Obese') &
                              (df['Smoking'] == 'Current')).astype(int)
    if 'Insurance_Coverage' in df.columns and 'Income_Quartile' in df.columns:
        df['Uninsured_LowIncome'] = ((df['Insurance_Coverage'] == 'Uninsured') &
                                     (df['Income_Quartile'] == 'Q1_Low')).astype(int)
    if 'Age' in df.columns and 'BMI' in df.columns:
        df['Age_per_BMI'] = (df['Age'] / df['BMI'].clip(lower=1)).round(2)
    if 'Family_Income' in df.columns:
        median_income = df['Family_Income'].median()
        df['Income_Gap'] = (df['Family_Income'] - median_income).round(0)
    return df


def load_artifact(target):
    path = os.path.join(ARTIFACT_DIR, f'{target.lower()}_best_model.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def preprocess(df, artifact, target_col):
    """Apply saved preprocessing to new data."""
    df = add_new_features(df)

    feature_cols = [c for c in df.columns if c not in [target_col] + DROP_COLS]
    X = df[feature_cols].copy()

    cat_cols = artifact['cat_cols']
    num_cols = artifact['num_cols']
    encoders = artifact['encoders']
    scaler = artifact['scaler']

    # Handle infinities
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaNs
    for col in X.columns:
        if X[col].isna().sum() > 0:
            if col in cat_cols:
                mode_vals = X[col].dropna().mode()
                fill_val = mode_vals[0] if len(mode_vals) > 0 else 'Unknown'
                X[col] = X[col].fillna(fill_val)
            else:
                fill_val = X[col].median()
                X[col] = X[col].fillna(fill_val)

    # Encode categoricals
    for col in cat_cols:
        if col in encoders and col in X.columns:
            X[col] = X[col].astype(str)
            le = encoders[col]

            def safe_encode(x):
                if x in le.classes_:
                    return le.transform([x])[0]
                # Unknown category -> map to most frequent class (index 0 if sorted)
                return 0

            X[col] = X[col].apply(safe_encode)

    # Scale numericals
    if len(num_cols) > 0:
        existing_num = [c for c in num_cols if c in X.columns]
        if existing_num:
            X[existing_num] = scaler.transform(X[existing_num])

    # Subset to selected features
    selected_features = artifact['feature_names']
    missing = [c for c in selected_features if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required features in input: {missing}")
    X = X[selected_features]

    # Final NaN sweep
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(0)

    assert not np.isnan(X.values).any(), "NaNs remain after preprocessing!"
    return X


def predict(df, target):
    """Score new patients and return risk predictions."""
    artifact = load_artifact(target)
    model = artifact['model']
    threshold = artifact['threshold']

    X = preprocess(df, artifact, target)

    prob = model.predict_proba(X)[:, 1]

    # Risk tiers
    try:
        tier = pd.qcut(prob, q=[0, 0.7, 0.9, 1.0],
                       labels=['Low', 'Medium', 'High'])
    except ValueError:
        tier = pd.cut(prob, bins=[0, 0.3, 0.6, 1.0],
                      labels=['Low', 'Medium', 'High'], include_lowest=True)

    result = pd.DataFrame({
        'Patient_ID': df['ID'] if 'ID' in df.columns else range(1, len(df) + 1),
        'Target_Disease': target,
        'Risk_Probability': prob.round(4),
        'Risk_Tier': tier,
        'Flagged_For_Screening': (prob >= threshold).astype(int),
        'Screening_Threshold_Used': threshold
    })

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Predict disease risk for new patients using trained MEPS models.'
    )
    parser.add_argument('--input', required=True,
                        help='Path to input CSV with patient features')
    parser.add_argument('--target', required=True,
                        choices=['Diabetes', 'Hypertension', 'Heart_Disease'],
                        help='Disease to predict')
    parser.add_argument('--output', default='predictions.csv',
                        help='Output CSV path (default: predictions.csv)')
    parser.add_argument('--explain', action='store_true',
                        help='Print top risk drivers for each patient (requires SHAP)')
    args = parser.parse_args()

    print("=" * 60)
    print("MEPS 2020 — Patient Risk Prediction")
    print("=" * 60)
    print(f"Target: {args.target}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    # Load data
    df = pd.read_csv(args.input)
    print(f"\nLoaded {len(df)} patients × {df.shape[1]} features")

    # Predict
    result = predict(df, args.target)

    # Save
    result.to_csv(args.output, index=False)
    print(f"\nSaved predictions to: {args.output}")

    # Summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Total patients:      {len(result)}")
    print(f"Flagged for screening: {result['Flagged_For_Screening'].sum()} ({result['Flagged_For_Screening'].mean():.1%})")
    print("\nRisk Tier Distribution:")
    print(result['Risk_Tier'].value_counts().to_string())
    print("\nTop 10 Highest Risk Patients:")
    top10 = result.nlargest(10, 'Risk_Probability')[['Patient_ID', 'Risk_Probability', 'Risk_Tier']]
    print(top10.to_string(index=False))

    # Optional SHAP explanations
    if args.explain:
        try:
            import shap
            artifact = load_artifact(args.target)
            X = preprocess(df, artifact, args.target)
            model = artifact['model']

            print("\n--- SHAP Explanations (Top 5 Patients) ---")
            explainer = shap.Explainer(model, X.iloc[:50])
            shap_vals = explainer(X.iloc[:5])

            for i in range(5):
                print(f"\nPatient {result.iloc[i]['Patient_ID']} "
                      f"(Prob={result.iloc[i]['Risk_Probability']:.3f}):")
                sv = shap_vals[i]
                top_features = np.argsort(-np.abs(sv.values))[:5]
                for j in top_features:
                    direction = "INCREASES" if sv.values[j] > 0 else "DECREASES"
                    print(f"  {sv.feature_names[j]:20s}: {sv.values[j]:+7.3f}  ({direction} risk)")
        except Exception as e:
            print(f"\nSHAP explanation failed: {e}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == '__main__':
    main()