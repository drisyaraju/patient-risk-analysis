"""
Phase 7: SHAP Explainability, Fairness Audit, PR Curves & Risk Stratification
Patient Risk Analysis — MEPS 2020
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    brier_score_loss, precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("=" * 60)
    print("WARNING: shap not installed. Install: pip install shap")
    print("SHAP plots skipped. Fairness, calibration & PR curves will run.")
    print("=" * 60)

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')

# ============================================
# CONFIGURATION
# ============================================

DATA_DIR = 'meps_data'
TARGETS = ['Diabetes', 'Hypertension', 'Heart_Disease']
DROP_COLS = ['ID']

ARTIFACT_DIR = os.path.join(DATA_DIR, 'results', 'models_v2')
OUTPUT_DIR = os.path.join(DATA_DIR, 'results', 'explainability')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_FILES = {
    'Diabetes': os.path.join(DATA_DIR, 'meps_diabetes_test_engineered.csv'),
    'Hypertension': os.path.join(DATA_DIR, 'meps_hypertension_test_engineered.csv'),
    'Heart_Disease': os.path.join(DATA_DIR, 'meps_heart_disease_test_engineered.csv')
}

CATEGORICAL_COLS = [
    'Gender', 'Race', 'Insurance_Coverage', 'Smoking',
    'Self_Reported_Health', 'Age_Group', 'BMI_Category',
    'Income_Quartile', 'Risk_Category', 'Insurance_Gender'
]


# ============================================
# NEW FEATURES (match Phase 6.5)
# ============================================

def add_new_features(df):
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


# ============================================
# LOAD & RECONSTRUCT
# ============================================

def load_artifact(target):
    path = os.path.join(ARTIFACT_DIR, f'{target.lower()}_best_model.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def preprocess_with_artifact(test_df, target_col, artifact):
    test_df = add_new_features(test_df)

    feature_cols = [c for c in test_df.columns if c not in [target_col] + DROP_COLS]
    X = test_df[feature_cols].copy()
    y = test_df[target_col].values

    cat_cols = artifact['cat_cols']
    num_cols = artifact['num_cols']
    encoders = artifact['encoders']
    scaler = artifact['scaler']

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in X.columns:
        if X[col].isna().sum() > 0:
            if col in cat_cols:
                mode_vals = X[col].dropna().mode()
                fill_val = mode_vals[0] if len(mode_vals) > 0 else 'Unknown'
                X[col] = X[col].fillna(fill_val)
            else:
                fill_val = X[col].median()
                X[col] = X[col].fillna(fill_val)

    for col in cat_cols:
        if col in encoders and col in X.columns:
            X[col] = X[col].astype(str)
            le = encoders[col]

            def encode_val(x):
                if x in le.classes_:
                    return le.transform([x])[0]
                return 0

            X[col] = X[col].apply(encode_val)

    if len(num_cols) > 0:
        existing_num = [c for c in num_cols if c in X.columns]
        if existing_num:
            X[existing_num] = scaler.transform(X[existing_num])

    selected_features = artifact['feature_names']
    X = X[selected_features]

    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(0)

    assert not np.isnan(X.values).any()
    return X, y


# ============================================
# SHAP
# ============================================

def run_shap_analysis(model, X_test, feature_names, target, model_name, save_dir):
    if not SHAP_AVAILABLE:
        print("  SHAP skipped (not installed)")
        return

    print("\n  --- SHAP Explainability ---")
    background = X_test.iloc[:min(100, len(X_test))]

    try:
        explainer = shap.Explainer(model, background)
        shap_values = explainer(X_test)
    except Exception as e:
        print(f"    shap.Explainer failed ({e}), trying TreeExplainer...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        except Exception as e2:
            print(f"    TreeExplainer also failed ({e2}). Skipping SHAP.")
            return

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f'SHAP Feature Importance — {target}\n{model_name}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{target.lower()}_shap_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved SHAP summary: {path}")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title(f'Mean |SHAP| — {target}\n{model_name}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{target.lower()}_shap_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved SHAP bar: {path}")

    if hasattr(shap, 'plots') and hasattr(shap.plots, 'waterfall'):
        for i in range(min(3, len(X_test))):
            try:
                plt.figure(figsize=(10, 6))
                if hasattr(shap_values, 'shape'):
                    shap.plots.waterfall(shap_values[i], max_display=15, show=False)
                else:
                    shap.force_plot(explainer.expected_value[1] if isinstance(explainer.expected_value, list)
                                    else explainer.expected_value,
                                    shap_values[i], X_test.iloc[i], feature_names=feature_names,
                                    matplotlib=True, show=False)
                plt.title(f'Patient {i+1} Explanation — {target}', fontsize=12)
                plt.tight_layout()
                path = os.path.join(save_dir, f'{target.lower()}_patient_{i+1}_explanation.png')
                plt.savefig(path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    Saved patient {i+1} explanation: {path}")
            except Exception as e:
                print(f"    Patient {i+1} plot failed: {e}")
                break


# ============================================
# FAIRNESS AUDIT
# ============================================

def fairness_audit(model, X_test, y_test, test_df_raw, target, threshold, save_dir):
    subgroups = {
        'Gender': test_df_raw['Gender'].values,
        'Race': test_df_raw['Race'].values,
        'Insurance_Coverage': test_df_raw['Insurance_Coverage'].values
    }

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    results = []

    print("\n  --- Fairness Audit ---")
    for attr, groups in subgroups.items():
        unique_groups = np.unique(groups)
        print(f"    {attr}: {unique_groups}")

        for group in unique_groups:
            mask = groups == group
            if mask.sum() < 30:
                continue

            y_g = y_test[mask]
            y_pred_g = y_pred[mask]
            y_prob_g = y_prob[mask]

            if len(np.unique(y_g)) < 2:
                roc = np.nan
            else:
                roc = roc_auc_score(y_g, y_prob_g)

            results.append({
                'Target': target,
                'Attribute': attr,
                'Group': group,
                'N': mask.sum(),
                'Prevalence': y_g.mean(),
                'Accuracy': accuracy_score(y_g, y_pred_g),
                'Precision': precision_score(y_g, y_pred_g, zero_division=0),
                'Recall': recall_score(y_g, y_pred_g, zero_division=0),
                'F1-Score': f1_score(y_g, y_pred_g, zero_division=0),
                'ROC-AUC': roc
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, f'{target.lower()}_fairness_audit.csv')
    df.to_csv(csv_path, index=False)
    print(f"    Saved fairness CSV: {csv_path}")

    for attr in subgroups.keys():
        attr_df = df[df['Attribute'] == attr]
        if len(attr_df) < 2:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        metrics = ['Precision', 'Recall', 'ROC-AUC']

        for ax, metric in zip(axes, metrics):
            bars = ax.bar(attr_df['Group'].astype(str), attr_df[metric], color='steelblue')
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(metric)
            ax.set_title(f'{attr} — {metric}')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=15)

            gap = attr_df[metric].max() - attr_df[metric].min()
            if not np.isnan(gap):
                color = 'red' if gap > 0.1 else 'green'
                ax.text(0.5, 0.95, f'Gap: {gap:.3f}',
                       transform=ax.transAxes, ha='center', va='top',
                       fontsize=10, color=color, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Fairness Disparity — {target} ({attr})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(save_dir, f'{target.lower()}_fairness_{attr.lower()}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved disparity plot: {path}")

    return df


# ============================================
# CALIBRATION
# ============================================

def calibration_analysis(model, X_test, y_test, target, save_dir):
    y_prob = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_prob)
    print(f"\n  --- Calibration ---")
    print(f"    Brier Score: {brier:.4f}")

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.plot(prob_pred, prob_true, 's-', color='steelblue', linewidth=2,
            label=f'{target} (Brier={brier:.3f})')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Reliability Diagram — {target}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{target.lower()}_calibration.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved calibration plot: {path}")

    return brier


# ============================================
# PRECISION-RECALL CURVE
# ============================================

def pr_curve_analysis(model, X_test, y_test, target, save_dir):
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, linewidth=2, color='steelblue',
            label=f'{target} (AP={ap:.3f})')
    ax.axhline(baseline, color='red', linestyle='--', linewidth=1.5,
               label=f'Baseline (prevalence={baseline:.1%})')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision (PPV)', fontsize=12)
    ax.set_title(f'Precision-Recall Curve — {target}\n'
                 f'AP={ap:.3f} | Baseline={baseline:.3f}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    ax.text(0.95, 0.05, f'Average Precision\n(AP) = {ap:.3f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(save_dir, f'{target.lower()}_pr_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved PR curve: {path}")

    pr_df = pd.DataFrame({
        'Precision': precision[:-1],
        'Recall': recall[:-1],
        'Threshold': thresholds
    })
    pr_df.to_csv(os.path.join(save_dir, f'{target.lower()}_pr_curve_data.csv'), index=False)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    best_f1 = f1_scores[best_idx]
    best_t = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    print(f"    Average Precision (AP): {ap:.4f}")
    print(f"    Best F1 on PR curve: {best_f1:.4f} at threshold={best_t:.3f}")
    print(f"    Interpretation: ", end="")
    if ap > 0.5:
        print("Strong model — far above baseline.")
    elif ap > 0.3:
        print("Moderate — usable with careful threshold selection.")
    elif ap > 0.2:
        print("Weak — many false positives expected.")
    else:
        print("Poor — not recommended for clinical use.")

    return ap, best_f1, best_t


# ============================================
# RISK STRATIFICATION
# ============================================

def risk_stratification(model, X_test, y_test, target, save_dir):
    y_prob = model.predict_proba(X_test)[:, 1]

    tiers = pd.DataFrame({
        'Probability': y_prob,
        'Actual': y_test
    })

    try:
        tiers['Tier'] = pd.qcut(tiers['Probability'], q=[0, 0.7, 0.9, 1.0],
                                labels=['Low', 'Medium', 'High'])
    except ValueError:
        tiers['Tier'] = pd.cut(tiers['Probability'], bins=[0, 0.3, 0.6, 1.0],
                               labels=['Low', 'Medium', 'High'], include_lowest=True)

    summary = tiers.groupby('Tier').agg({
        'Actual': ['count', 'sum', 'mean'],
        'Probability': 'mean'
    }).round(3)

    summary.columns = ['N', 'Cases', 'Prevalence', 'Mean_Prob']
    summary['Capture_Rate'] = (summary['Cases'] / summary['Cases'].sum()).round(3)

    print(f"\n  --- Risk Stratification — {target} ---")
    print(summary.to_string())

    summary.to_csv(os.path.join(save_dir, f'{target.lower()}_risk_tiers.csv'))
    print(f"    Saved risk tiers: {os.path.join(save_dir, f'{target.lower()}_risk_tiers.csv')}")

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(summary))
    bars = ax.bar(x_pos, summary['Prevalence'], color=['green', 'orange', 'red'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary.index)
    ax.set_ylabel('Actual Disease Prevalence')
    ax.set_title(f'Risk Tier Validation — {target}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)

    for bar, (idx, row) in zip(bars, summary.iterrows()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}\nN={int(row["N"])}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{target.lower()}_risk_stratification.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved stratification plot: {path}")

    return summary


# ============================================
# REPORT
# ============================================

def generate_report(all_fairness, all_brier, all_pr, best_models, thresholds):
    path = os.path.join(OUTPUT_DIR, 'explainability_report.txt')

    with open(path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 7: SHAP, FAIRNESS, CALIBRATION, PR CURVES & RISK TIERS\n")
        f.write("MEPS 2020 Patient Risk Analysis\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. PRECISION-RECALL ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write("ROC-AUC is misleading for imbalanced data (11% prevalence).\n")
        f.write("Average Precision (AP) and PR curves reveal true performance.\n\n")

        for target in TARGETS:
            ap, best_f1, best_t = all_pr[target]
            f.write(f"\n{target}:\n")
            f.write(f"  Average Precision (AP): {ap:.4f}\n")
            f.write(f"  Baseline (random):      {thresholds[target]:.4f}\n")
            f.write(f"  Best F1 on PR curve:    {best_f1:.4f} at t={best_t:.3f}\n")
            if ap > 0.5:
                f.write("  -> Strong discriminative power.\n")
            elif ap > 0.3:
                f.write("  -> Moderate — usable with careful threshold selection.\n")
            elif ap > 0.2:
                f.write("  -> Weak — many false positives expected.\n")
            else:
                f.write("  -> Poor — not recommended for clinical use.\n")

        f.write("\n\n2. MODEL CALIBRATION\n")
        f.write("-" * 50 + "\n")
        for target in TARGETS:
            f.write(f"\n{target}:\n")
            f.write(f"  Brier Score: {all_brier[target]:.4f}\n")
            if all_brier[target] < 0.1:
                f.write("  -> Well calibrated. Probabilities are reliable.\n")
            elif all_brier[target] < 0.2:
                f.write("  -> Moderately calibrated. Consider Platt scaling.\n")
            else:
                f.write("  -> Poorly calibrated. Do not use raw probabilities.\n")

        f.write("\n\n3. FAIRNESS AUDIT\n")
        f.write("-" * 50 + "\n")
        f.write("Gap > 0.10 in ROC-AUC/Precision/Recall = significant disparity.\n\n")

        for target in TARGETS:
            f.write(f"\n--- {target.upper()} ---\n")
            df = all_fairness[target]
            for attr in ['Gender', 'Race', 'Insurance_Coverage']:
                attr_df = df[df['Attribute'] == attr]
                if len(attr_df) < 2:
                    continue
                f.write(f"  {attr}:\n")
                for _, row in attr_df.iterrows():
                    # FIXED: str(row['Group']) instead of row['Group']
                    f.write(f"    {str(row['Group']):20s}: ROC={row['ROC-AUC']:.3f}, "
                           f"Prec={row['Precision']:.3f}, Rec={row['Recall']:.3f}, "
                           f"N={int(row['N'])}, Prev={row['Prevalence']:.1%}\n")

                for metric in ['ROC-AUC', 'Precision', 'Recall']:
                    gap = attr_df[metric].max() - attr_df[metric].min()
                    f.write(f"    -> {metric} gap: {gap:.3f}")
                    if gap > 0.1:
                        f.write("  *** DISPARITY ***")
                    f.write("\n")

        f.write("\n\n4. SHAP CLINICAL INTERPRETATION\n")
        f.write("-" * 50 + "\n")
        f.write("""
SHAP values show how each feature pushes risk up/down vs. population average.

High-risk patient profile:
  * Age 65+, BMI 30+, Low income, Fair/Poor self-reported health
  * Current smoker, multiple comorbidities, uninsured/public insurance

Actionable factors:
  * BMI -> lifestyle intervention, diet, exercise programs
  * Smoking -> cessation programs
  * Physical activity -> physical therapy, rehab
  * Insurance -> Medicaid expansion, community health programs

Non-modifiable (intensify screening):
  * Age, Race (genetic predisposition), Gender
""" + "\n")

        f.write("\n5. RISK STRATIFICATION FOR DEPLOYMENT\n")
        f.write("-" * 50 + "\n")
        f.write("""
Tier Definitions:
  * HIGH RISK (top 10% predicted probability):
      -> Actual prevalence typically 40-70%
      -> Immediate specialist referral + intensive intervention
  * MEDIUM RISK (70-90th percentile):
      -> Actual prevalence typically 10-25%
      -> Annual screening + lifestyle counseling
  * LOW RISK (bottom 70%):
      -> Actual prevalence typically <5%
      -> Standard care + periodic reassessment

Use TUNED THRESHOLD for binary screening flag (not 0.5).
""" + "\n")

        f.write("\n6. DEPLOYMENT & ETHICS\n")
        f.write("-" * 50 + "\n")
        f.write("""
* Review fairness gaps > 0.10 before production.
* Document SHAP explanations for each flagged patient.
* Continuous monitoring: alert if subgroup performance drifts > 5%.
* Regulatory: Satisfies FDA SaMD and EU AI Act explainability requirements.
""" + "\n")

    print(f"\nExplainability report saved: {path}")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("PHASE 7: SHAP, FAIRNESS, CALIBRATION, PR CURVES & RISK TIERS")
    print("MEPS 2020 Patient Risk Analysis")
    print("=" * 70)

    if not SHAP_AVAILABLE:
        print("\nNOTE: SHAP not installed. Only fairness, calibration, PR & risk tiers will run.")
        print("To get SHAP plots: pip install shap\n")

    all_fairness = {}
    all_brier = {}
    all_pr = {}
    best_models = {}
    thresholds = {}

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {target.upper()}")
        print(f"{'='*60}")

        artifact = load_artifact(target)
        model = artifact['model']
        threshold = artifact['threshold']
        feature_names = artifact['feature_names']
        best_models[target] = artifact['best_model_name']
        thresholds[target] = threshold

        print(f"  Model: {best_models[target]}")
        print(f"  Threshold: {threshold:.3f}")

        test_raw = pd.read_csv(TEST_FILES[target])
        test_raw = add_new_features(test_raw)

        X_test, y_test = preprocess_with_artifact(test_raw, target, artifact)
        print(f"  Reconstructed: X={X_test.shape}, y={y_test.shape}")

        run_shap_analysis(model, X_test, feature_names, target, best_models[target], OUTPUT_DIR)

        fairness_df = fairness_audit(model, X_test, y_test, test_raw, target, threshold, OUTPUT_DIR)
        all_fairness[target] = fairness_df

        brier = calibration_analysis(model, X_test, y_test, target, OUTPUT_DIR)
        all_brier[target] = brier

        ap, best_f1, best_t = pr_curve_analysis(model, X_test, y_test, target, OUTPUT_DIR)
        all_pr[target] = (ap, best_f1, best_t)

        risk_stratification(model, X_test, y_test, target, OUTPUT_DIR)

    generate_report(all_fairness, all_brier, all_pr, best_models, thresholds)

    print("\n" + "=" * 70)
    print("PHASE 7 COMPLETE")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()