"""
Phase 6: Model Training & Evaluation
Patient Risk Analysis — MEPS 2020
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Using GradientBoosting as alternative.")
    print("Install with: pip install xgboost")

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================
# CONFIGURATION
# ============================================

DATA_DIR = 'meps_data'
TARGETS = ['Diabetes', 'Hypertension', 'Heart_Disease']

CATEGORICAL_COLS = [
    'Gender', 'Race', 'Insurance_Coverage', 'Smoking',
    'Self_Reported_Health', 'Age_Group', 'BMI_Category',
    'Income_Quartile', 'Risk_Category', 'Insurance_Gender'
]

OUTLIER_COLS = ['Family_Income', 'BMI', 'Age_x_Income', 'Income_per_Age']


def find_file(base_name):
    paths_to_try = [
        os.path.join(DATA_DIR, base_name),
        os.path.join(DATA_DIR, base_name + '.csv'),
    ]
    for p in paths_to_try:
        if os.path.exists(p):
            return p
    return os.path.join(DATA_DIR, base_name + '.csv')


TRAIN_FILES = {
    'Diabetes': find_file('meps_diabetes_train_engineered'),
    'Hypertension': find_file('meps_hypertension_train_engineered'),
    'Heart_Disease': find_file('meps_heart_disease_train_engineered')
}
TEST_FILES = {
    'Diabetes': find_file('meps_diabetes_test_engineered'),
    'Hypertension': find_file('meps_hypertension_test_engineered'),
    'Heart_Disease': find_file('meps_heart_disease_test_engineered')
}

OUTPUT_DIR = os.path.join(DATA_DIR, 'results', 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5

# ============================================
# OUTLIER DETECTION (IQR Method)
# ============================================

def detect_outliers_iqr(df, cols, k=1.5):
    outlier_stats = {}
    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # SKIP if IQR is zero (happens when too many identical values, e.g. median-imputed)
        if IQR < 1e-9:
            print(f"    Skipping {col}: IQR=0 (too many identical values)")
            continue
            
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_stats[col] = (lower, upper, n_out)
    return outlier_stats


def clip_outliers(df, outlier_stats):
    df = df.copy()
    for col, (lower, upper, n_out) in outlier_stats.items():
        if n_out > 0 and col in df.columns:
            df[col] = df[col].clip(lower, upper)
            print(f"    Clipped {n_out} outliers in {col} -> [{lower:.1f}, {upper:.1f}]")
    return df


# ============================================
# 1. LOAD DATA
# ============================================

def load_data(target):
    train = pd.read_csv(TRAIN_FILES[target])
    test = pd.read_csv(TEST_FILES[target])
    print(f"\n[{target}] Loaded: Train={train.shape}, Test={test.shape}")
    print(f"  Class distribution (train): {train[target].value_counts().to_dict()}")
    return train, test


# ============================================
# 2. PREPROCESSING (Bulletproof)
# ============================================

def preprocess(train_df, test_df, target_col):
    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Determine column types
    detected_cat = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = list(set(CATEGORICAL_COLS + detected_cat).intersection(feature_cols))
    num_cols = [c for c in feature_cols if c not in cat_cols]

    print(f"  Features: {len(feature_cols)} total ({len(cat_cols)} cat, {len(num_cols)} num)")
    print(f"  Categorical: {cat_cols}")
    print(f"  Numerical: {num_cols}")

    # ===== STEP 1: Replace infinities with NaN =====
    for df_name, df in [('train', X_train), ('test', X_test)]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ===== STEP 2: Fill ALL NaNs (using assignment, NOT inplace=True) =====
    for df_name, df in [('train', X_train), ('test', X_test)]:
        for col in df.columns:
            n_nans = df[col].isna().sum()
            if n_nans > 0:
                if col in cat_cols:
                    mode_vals = df[col].dropna().mode()
                    fill_val = mode_vals[0] if len(mode_vals) > 0 else 'Unknown'
                    df[col] = df[col].fillna(fill_val)
                    print(f"  Filled {df_name}.{col} (cat, {n_nans} NaNs) with mode={fill_val}")
                else:
                    fill_val = df[col].median()
                    df[col] = df[col].fillna(fill_val)
                    print(f"  Filled {df_name}.{col} (num, {n_nans} NaNs) with median={fill_val:.2f}")

    # ===== STEP 3: Outlier Detection (IQR) =====
    print(f"\n  --- Outlier Detection (IQR k=1.5) ---")
    outlier_stats_train = detect_outliers_iqr(X_train, OUTLIER_COLS)
    X_train = clip_outliers(X_train, outlier_stats_train)
    X_test = clip_outliers(X_test, outlier_stats_train)

    # ===== STEP 4: Encode categoricals =====
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le

    # ===== STEP 5: Scale numericals =====
    scaler = StandardScaler()
    if len(num_cols) > 0:
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    # ===== STEP 6: FINAL NaN SWEEP =====
    print(f"\n  --- Final NaN Check ---")
    for df_name, df in [('train', X_train), ('test', X_test)]:
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            print(f"  Emergency NaNs in {df_name}: {nan_cols.to_dict()}")
            for col in nan_cols.index:
                fill_val = 0
                df[col] = df[col].fillna(fill_val)
                print(f"  Emergency filled {df_name}.{col} with {fill_val}")

    # Verify clean
    assert not np.isnan(X_train.values).any(), f"NaNs in X_train: {dict(X_train.isna().sum()[X_train.isna().sum() > 0])}"
    assert not np.isnan(X_test.values).any(), f"NaNs in X_test: {dict(X_test.isna().sum()[X_test.isna().sum() > 0])}"
    assert not np.isinf(X_train.values).any(), "Infs still in X_train!"
    assert not np.isinf(X_test.values).any(), "Infs still in X_test!"
    print(f"  Data clean: no NaNs or Infs remaining")

    return X_train, X_test, y_train, y_test, feature_cols, cat_cols, num_cols, encoders, scaler


# ============================================
# 3. MODEL DEFINITIONS
# ============================================

def get_models(y_train):
    """
    y_train is needed to compute scale_pos_weight for XGBoost.
    scale_pos_weight = count(negative) / count(positive)
    """
    # Compute class ratio for XGBoost
    neg, pos = np.bincount(y_train.astype(int))
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"  Class ratio (neg/pos): {scale_pos_weight:.2f}")

    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced', random_state=RANDOM_STATE,
            max_depth=12, min_samples_split=100, min_samples_leaf=50
        ),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced', n_estimators=200, max_depth=15,
            min_samples_split=50, min_samples_leaf=20,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_split=50, min_samples_leaf=20, random_state=RANDOM_STATE
        )
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,  # FIXED: numeric ratio, not 'balanced'
            random_state=RANDOM_STATE,
            n_jobs=-1, eval_metric='logloss'
        )

    return models


# ============================================
# 4. TRAINING & EVALUATION
# ============================================

def train_and_evaluate(target):
    print(f"\n{'='*60}")
    print(f"TRAINING MODELS FOR: {target.upper()}")
    print(f"{'='*60}")

    train_df, test_df = load_data(target)
    X_train, X_test, y_train, y_test, feature_names, cat_cols, num_cols, encoders, scaler = preprocess(train_df, test_df, target)

    models = get_models(y_train)
    results = []
    trained_models = {}

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\n  Training {name}...")

        cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            'Target': target,
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_prob),
            'CV_ROC_AUC_mean': cv_roc.mean(),
            'CV_ROC_AUC_std': cv_roc.std(),
            'CV_F1_mean': cv_f1.mean(),
            'CV_F1_std': cv_f1.std()
        }

        results.append(metrics)
        trained_models[name] = model

        print(f"    ROC-AUC: {metrics['ROC-AUC']:.4f} | F1: {metrics['F1-Score']:.4f} | CV-ROC: {cv_roc.mean():.4f}±{cv_roc.std():.4f}")

    results_df = pd.DataFrame(results)

    best_idx = results_df['ROC-AUC'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_model = trained_models[best_model_name]

    print(f"\n  >>> BEST MODEL: {best_model_name} (ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']:.4f})")

    artifact_path = os.path.join(OUTPUT_DIR, f'{target.lower()}_best_model.pkl')
    with open(artifact_path, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'encoders': encoders,
            'scaler': scaler,
            'feature_names': feature_names,
            'cat_cols': cat_cols,
            'num_cols': num_cols,
            'best_model_name': best_model_name,
            'metrics': results_df.loc[best_idx].to_dict()
        }, f)
    print(f"  Saved artifacts to: {artifact_path}")

    return results_df, trained_models, best_model_name, best_model, X_test, y_test, feature_names


# ============================================
# 5. VISUALIZATION
# ============================================

def plot_model_comparison(results_df, target, save_dir):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metrics))
    width = 0.15

    for i, (_, row) in enumerate(results_df.iterrows()):
        offset = (i - len(results_df)/2) * width + width/2
        ax.bar(x + offset, [row[m] for m in metrics], width, label=row['Model'])

    ax.set_ylabel('Score')
    ax.set_title(f'Model Comparison — {target}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{target.lower()}_model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison plot: {path}")


def plot_confusion_matrix(model, X_test, y_test, target, model_name, save_dir):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'], ax=ax)
    ax.set_title(f'Confusion Matrix — {target}\n{model_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    path = os.path.join(save_dir, f'{target.lower()}_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix: {path}")


def plot_roc_curves(trained_models, X_test, y_test, target, save_dir):
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves — {target}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{target.lower()}_roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved ROC curves: {path}")


def plot_feature_importance(model, feature_names, target, model_name, save_dir, top_n=15):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"  Skipping feature importance for {model_name}")
        return

    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feat_df)))
    ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances — {target}\n{model_name}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{target.lower()}_feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved feature importance: {path}")


# ============================================
# 6. HEALTHCARE REPORT
# ============================================

def generate_healthcare_report(all_results, best_models):
    report_path = os.path.join(OUTPUT_DIR, 'healthcare_relevance_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("HEALTHCARE RELEVANCE REPORT — Patient Risk Analysis (MEPS 2020)\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. WHY THESE METRICS MATTER IN HEALTHCARE\n")
        f.write("-" * 50 + "\n")
        f.write("""
Accuracy: Overall correctness, but MISLEADING with class imbalance.
          A model predicting 'no disease' for everyone achieves 89% accuracy
          for Diabetes/Heart Disease — yet catches ZERO patients.

Precision: Of predicted positives, how many are TRUE positives?
           High precision -> Fewer false alarms, less unnecessary testing/cost.

Recall (Sensitivity): Of actual positives, how many did we CATCH?
                      High recall -> Fewer missed diagnoses.

F1-Score: Harmonic mean of Precision & Recall. Best single metric for
          imbalanced medical data.

ROC-AUC: Ability to discriminate between diseased and healthy across ALL
         thresholds. AUC > 0.8 = clinically useful.
""" + "\n")

        f.write("\n2. MODEL PERFORMANCE SUMMARY\n")
        f.write("-" * 50 + "\n")

        for target in TARGETS:
            f.write(f"\n--- {target.upper()} ---\n")
            df = all_results[all_results['Target'] == target]
            f.write(df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))
            f.write(f"\n\nBest Model: {best_models[target]}\n")

            best_row = df[df['Model'] == best_models[target]].iloc[0]
            f.write(f"  ROC-AUC: {best_row['ROC-AUC']:.3f} -> ")
            if best_row['ROC-AUC'] >= 0.85:
                f.write("EXCELLENT discrimination.\n")
            elif best_row['ROC-AUC'] >= 0.75:
                f.write("GOOD discrimination. Useful for screening.\n")
            else:
                f.write("MODERATE discrimination.\n")

            f.write(f"  Recall: {best_row['Recall']:.3f} -> ")
            if best_row['Recall'] >= 0.70:
                f.write("Catches most at-risk patients.\n")
            else:
                f.write("May miss too many cases.\n")

        f.write("\n\n3. OUTLIER DETECTION METHODOLOGY\n")
        f.write("-" * 50 + "\n")
        f.write("""
Method: Interquartile Range (IQR) with k=1.5
Applied to: Family_Income, BMI, Age_x_Income, Income_per_Age

Why IQR for medical data:
  * Robust to skewed distributions (income is log-normal, BMI is right-skewed)
  * Unlike Z-score, IQR doesn't assume normality
  * Z-score would flag too many legitimate high-BMI obese patients as outliers
  * IQR captures only extreme values (e.g., BMI > 45, income > $200k)

Handling zero-IQR:
  * When >50% of values are identical (e.g., median-imputed NaNs), IQR=0
  * Skipping clipping prevents collapsing the feature to a single value
  * This preserves variance in heavily imputed columns like BMI

Clinical justification:
  * Extreme BMI values (>50) may be data entry errors or rare genetic conditions
  * Extreme income values could be top-coded MEPS values or errors
  * Outliers disproportionately influence tree-based models and linear models
  * Clipping preserves the patient record while preventing model distortion
""" + "\n")

        f.write("\n4. CLINICAL INSIGHTS\n")
        f.write("-" * 50 + "\n")
        f.write("""
Top Predictors:
  * Age / Age_Squared: Exponential risk increase
  * BMI / BMI_Category: Metabolic syndrome driver
  * Family_Income / Income_Quartile: Social determinants
  * Self_Reported_Health: Morbidity burden proxy
  * Hypertension: Comorbidity clustering
  * Smoking: Major modifiable risk factor
  * Comorbidity_Count: Disease clustering

Actionability:
  - HIGH-RISK (top decile) -> intensive counseling + specialist referral
  - MODERATE-RISK -> preventive care + annual screening
""" + "\n")

        f.write("\n5. DEPLOYMENT RECOMMENDATIONS\n")
        f.write("-" * 50 + "\n")
        f.write("""
* Threshold Tuning: Use precision-recall curves for optimal threshold.
* Fairness Audit: Check performance across Race/Gender/Insurance subgroups.
* Retraining: Retrain annually as population health trends shift.
* Explainability: Use SHAP values for individual patient explanations.
""" + "\n")

    print(f"\nHealthcare relevance report saved: {report_path}")


# ============================================
# 7. MAIN
# ============================================

def main():
    print("=" * 70)
    print("PHASE 6: MODEL TRAINING & EVALUATION")
    print("MEPS 2020 Patient Risk Analysis")
    print("=" * 70)

    print(f"\nData directory: {os.path.abspath(DATA_DIR)}")
    for target in TARGETS:
        t_ok = "OK" if os.path.exists(TRAIN_FILES[target]) else "MISSING"
        v_ok = "OK" if os.path.exists(TEST_FILES[target]) else "MISSING"
        print(f"  {target}: Train {t_ok} | Test {v_ok}")

    all_results = []
    best_models = {}

    for target in TARGETS:
        results_df, trained_models, best_name, best_model, X_test, y_test, feature_names = train_and_evaluate(target)
        all_results.append(results_df)
        best_models[target] = best_name

        plot_model_comparison(results_df, target, OUTPUT_DIR)
        plot_confusion_matrix(best_model, X_test, y_test, target, best_name, OUTPUT_DIR)
        plot_roc_curves(trained_models, X_test, y_test, target, OUTPUT_DIR)
        plot_feature_importance(best_model, feature_names, target, best_name, OUTPUT_DIR)

    final_results = pd.concat(all_results, ignore_index=True)
    results_csv = os.path.join(OUTPUT_DIR, 'all_model_results.csv')
    final_results.to_csv(results_csv, index=False)
    print(f"\nAll results saved: {results_csv}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY — BEST MODEL PER TARGET")
    print("=" * 70)
    summary = final_results.loc[final_results.groupby('Target')['ROC-AUC'].idxmax()]
    print(summary[['Target', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))

    generate_healthcare_report(final_results, best_models)

    print("\n" + "=" * 70)
    print("PHASE 6 COMPLETE")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()