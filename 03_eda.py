"""
STEP 3: EXPLORATORY DATA ANALYSIS
=================================
Impressive visualizations with healthcare storytelling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import os

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

SAVE_DIR = "meps_data"
RESULTS_DIR = "results/eda"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ========== LOAD DATA ==========
cohorts = {
    'Diabetes': pd.concat([
        pd.read_csv(f"{SAVE_DIR}/meps_diabetes_train.csv"),
        pd.read_csv(f"{SAVE_DIR}/meps_diabetes_test.csv")
    ], ignore_index=True),
    'Hypertension': pd.concat([
        pd.read_csv(f"{SAVE_DIR}/meps_hypertension_train.csv"),
        pd.read_csv(f"{SAVE_DIR}/meps_hypertension_test.csv")
    ], ignore_index=True),
    'Heart_Disease': pd.concat([
        pd.read_csv(f"{SAVE_DIR}/meps_heart_disease_train.csv"),
        pd.read_csv(f"{SAVE_DIR}/meps_heart_disease_test.csv")
    ], ignore_index=True)
}

df = cohorts['Diabetes'].copy()
print(f"Loaded all cohorts. Primary cohort (Diabetes): {df.shape}")

# ========== 1. MISSING DATA PATTERNS ==========
print("\n1. Missing Data Patterns")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Load original cleaned data for missingness visualization
df_original = pd.read_csv(f"{SAVE_DIR}/meps_PERFECTLY_CLEANED.csv")
msno.matrix(df_original, ax=axes[0], sparkline=False)
axes[0].set_title('Missing Data Pattern (Before Imputation)', fontsize=14, fontweight='bold')

missing = df_original.isnull().sum()
missing_pct = (missing / len(df_original) * 100)
missing_df = pd.DataFrame({'Missing %': missing_pct[missing_pct > 0]}).sort_values('Missing %')
missing_df.plot(kind='barh', ax=axes[1], color='coral', legend=False)
axes[1].set_title('Missing Value Percentage', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Percentage Missing (%)')

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/01_missing_data.png", dpi=300, bbox_inches='tight')
plt.close()

# ========== 2. TARGET DISTRIBUTIONS ==========
print("\n2. Target Distributions")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
targets = ['Diabetes', 'Hypertension', 'Heart_Disease']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for idx, (target, color) in enumerate(zip(targets, colors)):
    data = cohorts[target][target]
    counts = data.value_counts().sort_index()
    percentages = counts / len(data) * 100
    
    bars = axes[idx].bar(['No ' + target, target], counts.values, 
                         color=[color, '#2C3E50'], edgecolor='black')
    axes[idx].set_title(f'{target}\n(n={len(data):,})', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('Count')
    
    for bar, pct in zip(bars, percentages.values):
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:,.0f}\n({pct:.1f}%)',
                      ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/02_target_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

# ========== 3. AGE BY DISEASE ==========
print("\n3. Age Analysis")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, target in enumerate(targets):
    data = cohorts[target]
    no_disease = data[data[target] == 0]['Age']
    has_disease = data[data[target] == 1]['Age']
    
    axes[idx].hist(no_disease, bins=30, alpha=0.7, label=f'No {target}', 
                   color='lightblue', edgecolor='black')
    axes[idx].hist(has_disease, bins=30, alpha=0.7, label=target, 
                   color='coral', edgecolor='black')
    
    axes[idx].axvline(no_disease.mean(), color='blue', linestyle='--', linewidth=2)
    axes[idx].axvline(has_disease.mean(), color='red', linestyle='--', linewidth=2)
    
    axes[idx].set_title(f'{target} by Age', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Age')
    axes[idx].set_ylabel('Count')
    axes[idx].legend()

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/03_age_by_disease.png", dpi=300, bbox_inches='tight')
plt.close()

# ========== 4. INCOME BY DISEASE ==========
print("\n4. Income Analysis")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, target in enumerate(targets):
    data = cohorts[target].copy()
    data['Income_Quartile'] = pd.qcut(data['Family_Income'], q=4, 
                                       labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'])
    
    quartile_rates = data.groupby('Income_Quartile')[target].mean() * 100
    
    bars = axes[idx].bar(range(len(quartile_rates)), quartile_rates.values,
                         color=['#E74C3C', '#F39C12', '#F1C40F', '#27AE60'],
                         edgecolor='black')
    axes[idx].set_title(f'{target} Rate by Income', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('Disease Rate (%)')
    axes[idx].set_xticks(range(len(quartile_rates)))
    axes[idx].set_xticklabels(quartile_rates.index, rotation=15)
    
    for bar, rate in zip(bars, quartile_rates.values):
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                      f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/04_income_by_disease.png", dpi=300, bbox_inches='tight')
plt.close()

# ========== 6. CORRELATION MATRIX ==========
print("\n6. Correlation Matrix")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'ID']

plt.figure(figsize=(12, 10))
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/06_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# ========== 7. CATEGORICAL FEATURES ==========
print("\n7. Categorical Features")

categorical_features = ['Gender', 'Race', 'Insurance_Coverage', 'Smoking',
                        'Physical_Activity_Limit', 'Self_Reported_Health']

fig, axes = plt.subplots(len(categorical_features), 3, figsize=(20, 24))

for row, feature in enumerate(categorical_features):
    for col, target in enumerate(targets):
        data = cohorts[target]
        
        if feature in data.columns:
            crosstab = pd.crosstab(data[feature], data[target], normalize='index') * 100
            
            if 1 in crosstab.columns:
                rates = crosstab[1].sort_values(ascending=False)
                bars = axes[row, col].bar(range(len(rates)), rates.values,
                                          color=plt.cm.viridis(np.linspace(0, 1, len(rates))),
                                          edgecolor='black')
                axes[row, col].set_title(f'{target} by {feature}', fontsize=12, fontweight='bold')
                axes[row, col].set_ylabel('Disease Rate (%)')
                axes[row, col].set_xticks(range(len(rates)))
                axes[row, col].set_xticklabels(rates.index, rotation=45, ha='right', fontsize=9)
                
                for bar, rate in zip(bars, rates.values):
                    height = bar.get_height()
                    axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                       f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/07_categorical_features.png", dpi=300, bbox_inches='tight')
plt.close()

# ========== 8. COMORBIDITY ==========
print("\n8. Comorbidity Analysis")

df_comorb = cohorts['Diabetes'].copy()
comorb_cols = [c for c in ['Hypertension', 'Heart_Disease'] if c in df_comorb.columns]

if len(comorb_cols) >= 1:
    df_comorb['Comorbidity'] = 'Diabetes Only'
    for col in comorb_cols:
        df_comorb.loc[df_comorb[col] == 1, 'Comorbidity'] += f' + {col}'
    
    comorb_counts = df_comorb['Comorbidity'].value_counts()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(comorb_counts)))
    bars = ax.barh(comorb_counts.index, comorb_counts.values, color=colors, edgecolor='black')
    ax.set_title('Diabetes Comorbidity Patterns', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Patients')
    
    for bar, count in zip(bars, comorb_counts.values):
        width = bar.get_width()
        pct = count / len(df_comorb) * 100
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f' {count:,} ({pct:.1f}%)', ha='left', va='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/08_comorbidity.png", dpi=300, bbox_inches='tight')
    plt.close()

# ========== SUMMARY ==========
print("\n" + "=" * 70)
print("EDA COMPLETE!")
print("=" * 70)
print(f"Saved to: {RESULTS_DIR}/")
print("Files: 01_missing_data.png through 08_comorbidity.png")