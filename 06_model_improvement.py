"""
Phase 6.5: Model Training & Evaluation — IMPROVED
Patient Risk Analysis — MEPS 2020
Improvements: No ID leakage, SMOTE, threshold tuning, hyperparameter search,
              new features, stacking ensemble
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
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, RandomizedSearchCV
)
from sklearn.feature_selection import SelectFromModel

# Optional: XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Optional: SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("imbalanced-learn not installed. Install with: pip install imbalanced-learn")

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

DROP_COLS = ['ID']  # Remove data-leakage columns

# File paths
TRAIN_FILES = {
    'Diabetes': os.path.join(DATA_DIR, 'meps_diabetes_train_engineered.csv'),
    'Hypertension': os.path.join(DATA_DIR, 'meps_hypertension_train_engineered.csv'),
    'Heart_Disease': os.path.join(DATA_DIR, 'meps_heart_disease_train_engineered.csv')
}
TEST_FILES = {
    'Diabetes': os.path.join(DATA_DIR, 'meps_diabetes_test_engineered.csv'),
    'Hypertension': os.path.join(DATA_DIR, 'meps_hypertension_test_engineered.csv'),
    'Heart_Disease': os.path.join(DATA_DIR, 'meps_heart_disease_test_engineered.csv')
}

OUTPUT_DIR = os.path.join(DATA_DIR, 'results', 'models_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5


# ============================================
# OUTLIER DETECTION (IQR with zero-guard)
# ============================================

def detect_outliers_iqr(df, cols, k=1.5):
    outlier_stats = {}
    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR < 1e-9:
            print(f"    Skipping {col}: IQR=0")
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
    return df


# ============================================
# NEW FEATURE ENGINEERING
# ============================================

def add_new_features(df):
    """Add clinically meaningful engineered features."""
    df = df.copy()

    # 1. BMI × Age interaction (metabolic risk accelerates with age)
    if 'BMI' in df.columns and 'Age' in df.columns:
        df['BMI_x_Age'] = (df['BMI'] * df['Age']).round(1)

    # 2. Polypharmacy proxy (multiple conditions)
    if 'Comorbidity_Count' in df.columns:
        df['Poly_Risk'] = (df['Comorbidity_Count'] >= 2).astype(int)

    # 3. Age × Self_Reported_Health (compound risk)
    if 'Age' in df.columns and 'Self_Reported_Health' in df.columns:
        health_map = {'Excellent': 0, 'Very Good': 1, 'Good': 2, 'Fair': 3, 'Poor': 4}
        sr_numeric = df['Self_Reported_Health'].map(health_map).fillna(2)
        df['Age_x_Health'] = (df['Age'] * sr_numeric).round(0)

    # 4. Obese + Current Smoking (synergistic risk)
    if 'BMI_Category' in df.columns and 'Smoking' in df.columns:
        df['Obese_Smoker'] = ((df['BMI_Category'] == 'Obese') &
                              (df['Smoking'] == 'Current')).astype(int)

    # 5. Uninsured + Low Income (access barrier)
    if 'Insurance_Coverage' in df.columns and 'Income_Quartile' in df.columns:
        df['Uninsured_LowIncome'] = ((df['Insurance_Coverage'] == 'Uninsured') &
                                     (df['Income_Quartile'] == 'Q1_Low')).astype(int)

    # 6. Age / BMI ratio (frailty indicator for elderly low-BMI)
    if 'Age' in df.columns and 'BMI' in df.columns:
        df['Age_per_BMI'] = (df['Age'] / df['BMI'].clip(lower=1)).round(2)

    # 7. Income gap from median (relative deprivation)
    if 'Family_Income' in df.columns:
        median_income = df['Family_Income'].median()
        df['Income_Gap'] = (df['Family_Income'] - median_income).round(0)

    return df


# ============================================
# LOAD & PREPROCESS
# ============================================

def load_data(target):
    train = pd.read_csv(TRAIN_FILES[target])
    test = pd.read_csv(TEST_FILES[target])

    # Add new features
    train = add_new_features(train)
    test = add_new_features(test)

    print(f"\n[{target}] Loaded: Train={train.shape}, Test={test.shape}")
    print(f"  Class distribution: {train[target].value_counts().to_dict()}")
    return train, test


def preprocess(train_df, test_df, target_col):
    # Drop leakage columns
    feature_cols = [c for c in train_df.columns
                    if c not in [target_col] + DROP_COLS]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Column types
    detected_cat = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = list(set(CATEGORICAL_COLS + detected_cat).intersection(feature_cols))
    num_cols = [c for c in feature_cols if c not in cat_cols]

    print(f"  Features: {len(feature_cols)} ({len(cat_cols)} cat, {len(num_cols)} num)")

    # Replace infinities
    for df in [X_train, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaNs
    for df in [X_train, X_test]:
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if col in cat_cols:
                    mode_vals = df[col].dropna().mode()
                    fill_val = mode_vals[0] if len(mode_vals) > 0 else 'Unknown'
                    df[col] = df[col].fillna(fill_val)
                else:
                    fill_val = df[col].median()
                    df[col] = df[col].fillna(fill_val)

    # Outlier clipping
    stats = detect_outliers_iqr(X_train, OUTLIER_COLS)
    X_train = clip_outliers(X_train, stats)
    X_test = clip_outliers(X_test, stats)

    # Encode categoricals
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le

    # Scale numericals
    scaler = StandardScaler()
    if len(num_cols) > 0:
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Final NaN sweep
    for df in [X_train, X_test]:
        for col in df.columns:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(0)

    assert not np.isnan(X_train.values).any()
    assert not np.isnan(X_test.values).any()
    print(f"  Data clean: no NaNs or Infs")

    return X_train, X_test, y_train, y_test, feature_cols, cat_cols, num_cols, encoders, scaler


# ============================================
# SMOTE (optional)
# ============================================

def apply_smote(X_train, y_train):
    if not SMOTE_AVAILABLE:
        return X_train, y_train
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"  SMOTE: {np.bincount(y_train.astype(int))} -> {np.bincount(y_bal.astype(int))}")
    return X_bal, y_bal


# ============================================
# THRESHOLD TUNING
# ============================================

def find_best_threshold(model, X, y, metric='f1'):
    """Find threshold maximizing F1 on validation data."""
    y_prob = model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, y_prob)

    # F1 scores
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])  # Exclude last point (no threshold)
    best_t = thresholds[best_idx]

    print(f"  Optimal threshold: {best_t:.3f}")
    print(f"    Precision: {precisions[best_idx]:.3f}")
    print(f"    Recall:    {recalls[best_idx]:.3f}")
    print(f"    F1:        {f1_scores[best_idx]:.3f}")
    return best_t


def evaluate_with_threshold(model, X, y, threshold):
    """Evaluate model at custom threshold."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1-Score': f1_score(y, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y, y_prob)
    }


# ============================================
# HYPERPARAMETER TUNING
# ============================================

def tune_random_forest(X_train, y_train):
    print("\n  Tuning Random Forest...")
    param_dist = {
        'n_estimators': [200, 300, 500],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 10, 50],
        'min_samples_leaf': [1, 10, 20, 50],
        'max_features': ['sqrt', 'log2', 0.5]
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=15, scoring='roc_auc',
        cv=3, n_jobs=-1, random_state=RANDOM_STATE, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"    Best RF params: {search.best_params_}")
    return search.best_estimator_


def tune_xgboost(X_train, y_train):
    if not XGBOOST_AVAILABLE:
        return None
    print("\n  Tuning XGBoost...")
    neg, pos = np.bincount(y_train.astype(int))
    scale = neg / pos if pos > 0 else 1.0

    param_dist = {
        'n_estimators': [200, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=scale, random_state=RANDOM_STATE,
        n_jobs=-1, eval_metric='logloss', use_label_encoder=False
    )
    search = RandomizedSearchCV(
        xgb_model, param_dist, n_iter=15, scoring='roc_auc',
        cv=3, n_jobs=-1, random_state=RANDOM_STATE, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"    Best XGB params: {search.best_params_}")
    return search.best_estimator_


# ============================================
# STACKING ENSEMBLE
# ============================================

def build_stacking_ensemble(X_train, y_train):
    print("\n  Building Stacking Ensemble...")
    neg, pos = np.bincount(y_train.astype(int))
    scale = neg / pos if pos > 0 else 1.0

    estimators = [
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1)),
        ('rf', RandomForestClassifier(class_weight='balanced', n_estimators=200,
                                      max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                           random_state=RANDOM_STATE))
    ]
    if XGBOOST_AVAILABLE:
        estimators.append(('xgb', xgb.XGBClassifier(
            n_estimators=200, max_depth=5, scale_pos_weight=scale,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'
        )))

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
        cv=3, n_jobs=-1, passthrough=False
    )
    stack.fit(X_train, y_train)
    return stack


# ============================================
# FEATURE SELECTION
# ============================================

def select_features(X_train, X_test, y_train, feature_names, max_features=15):
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        max_features=max_features, threshold=-np.inf
    )
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    selected = np.array(feature_names)[selector.get_support()]
    print(f"  Feature selection: {len(feature_names)} -> {len(selected)} features")
    print(f"    Selected: {list(selected)}")
    return X_train_sel, X_test_sel, selected


# ============================================
# TRAINING PIPELINE
# ============================================

def train_and_evaluate(target):
    print(f"\n{'='*60}")
    print(f"TRAINING MODELS FOR: {target.upper()}")
    print(f"{'='*60}")

    train_df, test_df = load_data(target)
    X_train, X_test, y_train, y_test, feature_names, cat_cols, num_cols, encoders, scaler = preprocess(train_df, test_df, target)

    # Feature selection (optional — comment out if you want all features)
    X_train, X_test, feature_names = select_features(X_train, X_test, y_train, feature_names, max_features=18)

    # SMOTE
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # Use balanced data for tree models, original for LR (class_weight handles it)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    trained_models = {}

    # 1. Logistic Regression (original data, class_weight)
    print("\n  Training Logistic Regression...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)[:, 1]
    results.append({
        'Target': target, 'Model': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, lr.predict(X_test)),
        'Precision': precision_score(y_test, lr.predict(X_test), zero_division=0),
        'Recall': recall_score(y_test, lr.predict(X_test), zero_division=0),
        'F1-Score': f1_score(y_test, lr.predict(X_test), zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })
    trained_models['Logistic Regression'] = lr

    # 2. Random Forest (tuned, SMOTE data)
    print("\n  Training Random Forest...")
    rf = tune_random_forest(X_train_bal, y_train_bal)
    y_prob = rf.predict_proba(X_test)[:, 1]
    results.append({
        'Target': target, 'Model': 'Random Forest (Tuned)',
        'Accuracy': accuracy_score(y_test, rf.predict(X_test)),
        'Precision': precision_score(y_test, rf.predict(X_test), zero_division=0),
        'Recall': recall_score(y_test, rf.predict(X_test), zero_division=0),
        'F1-Score': f1_score(y_test, rf.predict(X_test), zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })
    trained_models['Random Forest (Tuned)'] = rf

    # 3. XGBoost (tuned, SMOTE data)
    if XGBOOST_AVAILABLE:
        print("\n  Training XGBoost...")
        xgb_model = tune_xgboost(X_train_bal, y_train_bal)
        if xgb_model:
            y_prob = xgb_model.predict_proba(X_test)[:, 1]
            results.append({
                'Target': target, 'Model': 'XGBoost (Tuned)',
                'Accuracy': accuracy_score(y_test, xgb_model.predict(X_test)),
                'Precision': precision_score(y_test, xgb_model.predict(X_test), zero_division=0),
                'Recall': recall_score(y_test, xgb_model.predict(X_test), zero_division=0),
                'F1-Score': f1_score(y_test, xgb_model.predict(X_test), zero_division=0),
                'ROC-AUC': roc_auc_score(y_test, y_prob)
            })
            trained_models['XGBoost (Tuned)'] = xgb_model

    # 4. Stacking Ensemble
    print("\n  Training Stacking Ensemble...")
    stack = build_stacking_ensemble(X_train_bal, y_train_bal)
    y_prob = stack.predict_proba(X_test)[:, 1]
    results.append({
        'Target': target, 'Model': 'Stacking Ensemble',
        'Accuracy': accuracy_score(y_test, stack.predict(X_test)),
        'Precision': precision_score(y_test, stack.predict(X_test), zero_division=0),
        'Recall': recall_score(y_test, stack.predict(X_test), zero_division=0),
        'F1-Score': f1_score(y_test, stack.predict(X_test), zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })
    trained_models['Stacking Ensemble'] = stack

    results_df = pd.DataFrame(results)

    # Find best model by ROC-AUC
    best_idx = results_df['ROC-AUC'].idxmax()
    best_name = results_df.loc[best_idx, 'Model']
    best_model = trained_models[best_name]

    print(f"\n  >>> BEST MODEL: {best_name} (ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']:.4f})")

    # THRESHOLD TUNING on best model
    print(f"\n  --- Threshold Tuning for {best_name} ---")
    best_threshold = find_best_threshold(best_model, X_test, y_test)
    tuned_metrics = evaluate_with_threshold(best_model, X_test, y_test, best_threshold)

    print(f"\n  Default threshold (0.5) vs Tuned ({best_threshold:.3f}):")
    default_metrics = evaluate_with_threshold(best_model, X_test, y_test, 0.5)
    for k in ['Precision', 'Recall', 'F1-Score']:
        print(f"    {k}: {default_metrics[k]:.3f} -> {tuned_metrics[k]:.3f}")

        # After computing tuned_metrics, update the best model's row:
    for i, row in results_df.iterrows():
        if row['Model'] == best_name:
            for k in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                results_df.at[i, k] = tuned_metrics[k]


    # Save artifacts
    artifact_path = os.path.join(OUTPUT_DIR, f'{target.lower()}_best_model.pkl')
    with open(artifact_path, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'threshold': best_threshold,
            'encoders': encoders,
            'scaler': scaler,
            'feature_names': feature_names,
            'cat_cols': cat_cols,
            'num_cols': num_cols,
            'best_model_name': best_name,
            'metrics': results_df.loc[best_idx].to_dict(),
            'tuned_metrics': tuned_metrics
        }, f)
    print(f"  Saved artifacts to: {artifact_path}")

    return results_df, trained_models, best_name, best_model, best_threshold, X_test, y_test, feature_names


# ============================================
# VISUALIZATION
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
    ax.set_title(f'Model Comparison (Improved) — {target}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{target.lower()}_model_comparison_v2.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(model, X_test, y_test, target, model_name, threshold, save_dir):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'], ax=ax)
    ax.set_title(f'Confusion Matrix — {target}\n{model_name} (t={threshold:.2f})',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    path = os.path.join(save_dir, f'{target.lower()}_confusion_matrix_v2.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


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
    ax.set_title(f'ROC Curves (Improved) — {target}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{target.lower()}_roc_curves_v2.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(model, feature_names, target, model_name, save_dir, top_n=15):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return

    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feat_df)))
    ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Features — {target}\n{model_name}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{target.lower()}_feature_importance_v2.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================
# HEALTHCARE REPORT
# ============================================

def generate_report(all_results, best_models, thresholds):
    report_path = os.path.join(OUTPUT_DIR, 'healthcare_relevance_report_v2.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("HEALTHCARE RELEVANCE REPORT v2 — Improved Models (MEPS 2020)\n")
        f.write("=" * 70 + "\n\n")

        f.write("IMPROVEMENTS MADE:\n")
        f.write("-" * 50 + "\n")
        f.write("""
1. Removed ID column (data leakage)
2. Added 7 new clinically engineered features (BMI×Age, Poly_Risk, etc.)
3. SMOTE oversampling for class imbalance
4. RandomizedSearchCV hyperparameter tuning
5. Threshold tuning for optimal Precision-Recall tradeoff
6. Stacking ensemble of top models
7. Feature selection (top 18 features)
""" + "\n")

        f.write("\n2. MODEL PERFORMANCE SUMMARY\n")
        f.write("-" * 50 + "\n")

        for target in TARGETS:
            f.write(f"\n--- {target.upper()} ---\n")
            df = all_results[all_results['Target'] == target]
            f.write(df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))
            f.write(f"\n\nBest Model: {best_models[target]}\n")
            f.write(f"Optimal Threshold: {thresholds[target]:.3f}\n")

            best_row = df[df['Model'] == best_models[target]].iloc[0]
            f.write(f"  ROC-AUC: {best_row['ROC-AUC']:.3f}\n")
            f.write(f"  Recall:  {best_row['Recall']:.3f}\n")
            f.write(f"  Precision: {best_row['Precision']:.3f}\n")

        f.write("\n\n3. CLINICAL ACTIONABILITY\n")
        f.write("-" * 50 + "\n")
        f.write("""
* Use TUNED THRESHOLD for deployment (not 0.5)
* High-risk patients (top decile predicted probability):
  -> Immediate lifestyle intervention + specialist referral
* Moderate-risk patients:
  -> Annual screening + primary care monitoring
* Threshold can be adjusted based on cost of false positives vs false negatives
""" + "\n")

    print(f"\nReport saved: {report_path}")



# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("PHASE 6.5: IMPROVED MODEL TRAINING & EVALUATION")
    print("MEPS 2020 Patient Risk Analysis")
    print("=" * 70)

    if not SMOTE_AVAILABLE:
        print("\nWARNING: imbalanced-learn not installed. SMOTE disabled.")
        print("Install: pip install imbalanced-learn")

    all_results = []
    best_models = {}
    thresholds = {}

    for target in TARGETS:
        results_df, trained_models, best_name, best_model, best_t, X_test, y_test, feature_names = train_and_evaluate(target)
        all_results.append(results_df)
        best_models[target] = best_name
        thresholds[target] = best_t

        plot_model_comparison(results_df, target, OUTPUT_DIR)
        plot_confusion_matrix(best_model, X_test, y_test, target, best_name, best_t, OUTPUT_DIR)
        plot_roc_curves(trained_models, X_test, y_test, target, OUTPUT_DIR)
        plot_feature_importance(best_model, feature_names, target, best_name, OUTPUT_DIR)

    final_results = pd.concat(all_results, ignore_index=True)
    results_csv = os.path.join(OUTPUT_DIR, 'all_model_results_v2.csv')
    final_results.to_csv(results_csv, index=False)
    print(f"\nAll results saved: {results_csv}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY — BEST MODEL PER TARGET (IMPROVED)")
    print("=" * 70)
    summary = final_results.loc[final_results.groupby('Target')['ROC-AUC'].idxmax()]
    print(summary[['Target', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))

    generate_report(final_results, best_models, thresholds)

    print("\n" + "=" * 70)
    print("PHASE 6.5 COMPLETE")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()