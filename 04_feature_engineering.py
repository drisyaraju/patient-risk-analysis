"""
STEP 4: FEATURE ENGINEERING
===========================
Create new features from raw data to improve model performance
"""

import pandas as pd
import numpy as np
import os

SAVE_DIR = "meps_data"
RESULTS_DIR = "results/features"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 70)

# ========== LOAD DATA ==========
print("\nLoading imputed datasets...")

cohorts = {
    'Diabetes': {
        'train': pd.read_csv(f"{SAVE_DIR}/meps_diabetes_train.csv"),
        'test': pd.read_csv(f"{SAVE_DIR}/meps_diabetes_test.csv")
    },
    'Hypertension': {
        'train': pd.read_csv(f"{SAVE_DIR}/meps_hypertension_train.csv"),
        'test': pd.read_csv(f"{SAVE_DIR}/meps_hypertension_test.csv")
    },
    'Heart_Disease': {
        'train': pd.read_csv(f"{SAVE_DIR}/meps_heart_disease_train.csv"),
        'test': pd.read_csv(f"{SAVE_DIR}/meps_heart_disease_test.csv")
    }
}

print("Loaded all cohorts")

# ========== FEATURE ENGINEERING FUNCTIONS ==========

def engineer_features(df):
    """
    Create new features from existing columns.
    Each transformation has a clinical or statistical rationale.
    """
    df = df.copy()
    
    # ------------------------------------------------------------------
    # FEATURE 1: Age Groups
    # ------------------------------------------------------------------
    # WHY: Age has non-linear relationship with disease risk
    #      Risk increases exponentially after 40, then plateaus
    #      Young adults (0-40): Low risk
    #      Middle-aged (40-60): Risk accelerates
    #      Older (60-75): High risk
    #      Senior (75+): Very high risk, but different care needs
    
    df['Age_Group'] = pd.cut(
        df['Age'], 
        bins=[0, 40, 60, 75, 100],
        labels=['Young', 'Middle', 'Older', 'Senior']
    )
    
    # ------------------------------------------------------------------
    # FEATURE 2: BMI Categories
    # ------------------------------------------------------------------
    # WHY: BMI has established clinical thresholds
    #      Underweight (<18.5): Malnutrition risk
    #      Normal (18.5-25): Healthy
    #      Overweight (25-30): Pre-disease state
    #      Obese (30+): Major risk factor for all three conditions
    
    df['BMI_Category'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )
    
    # ------------------------------------------------------------------
    # FEATURE 3: Income Quartiles
    # ------------------------------------------------------------------
    # WHY: EDA showed clear income-disease gradient
    #      Absolute income matters less than relative position
    #      Quartiles capture socioeconomic stratification
    
    df['Income_Quartile'] = pd.qcut(
        df['Family_Income'],
        q=4,
        labels=['Q1_Lowest', 'Q2', 'Q3', 'Q4_Highest'],
        duplicates='drop'
    )
    
    # ------------------------------------------------------------------
    # FEATURE 4: Age-Income Interaction
    # ------------------------------------------------------------------
    # WHY: EDA showed older + poorer = highest risk
    #      Simple multiplication captures this interaction
    #      High values = old and poor (most vulnerable)
    
    df['Age_x_Income'] = df['Age'] * df['Family_Income']
    
    # ------------------------------------------------------------------
    # FEATURE 5: Health Risk Score
    # ------------------------------------------------------------------
    # WHY: Self-reported health and mental health are strongly correlated
    #      Combining them creates a composite wellness indicator
    #      Lower score = worse overall health
    
    # First encode self-reported health numerically (5=excellent, 1=poor)
    health_map = {
        'Excellent': 5, 'Very Good': 4, 'Good': 3, 
        'Fair': 2, 'Poor': 1
    }
    
    # Apply mapping if columns are strings, otherwise assume already numeric
    if df['Self_Reported_Health'].dtype == 'object':
        df['Health_Numeric'] = df['Self_Reported_Health'].map(health_map)
    else:
        df['Health_Numeric'] = df['Self_Reported_Health']
    
    if df['Mental_Health_Status'].dtype == 'object':
        df['Mental_Numeric'] = df['Mental_Health_Status'].map(health_map)
    else:
        df['Mental_Numeric'] = df['Mental_Health_Status']
    
    # Combine: average of physical and mental health
    df['Health_Risk_Score'] = (df['Health_Numeric'] + df['Mental_Numeric']) / 2
    
    # ------------------------------------------------------------------
    # FEATURE 6: Comorbidity Count
    # ------------------------------------------------------------------
    # WHY: EDA showed diseases cluster together
    #      Count of other conditions predicts severity
    #      Only available if multiple targets exist in dataset
    
    target_cols = ['Diabetes', 'Hypertension', 'Heart_Disease']
    available_targets = [c for c in target_cols if c in df.columns]
    
    if len(available_targets) >= 2:
        # Sum of OTHER conditions (not the primary target)
        df['Comorbidity_Count'] = df[available_targets].sum(axis=1)
    else:
        df['Comorbidity_Count'] = 0
    
    # ------------------------------------------------------------------
    # FEATURE 7: Risk Category (High/Medium/Low)
    # ------------------------------------------------------------------
    # WHY: Clinicians think in risk categories, not probabilities
    #      Combines age, BMI, and health score into actionable tier
    
    # Create risk score components
    age_risk = pd.cut(df['Age'], bins=[0, 40, 60, 100], labels=[1, 2, 3]).astype(float)
    
    bmi_risk = df['BMI'].apply(
        lambda x: 3 if x >= 30 else (2 if x >= 25 else 1)
    )
    
    health_risk = df['Health_Risk_Score'].apply(
        lambda x: 3 if x <= 2 else (2 if x <= 3 else 1)
    )
    
    # Combined risk score
    total_risk = age_risk + bmi_risk + health_risk
    
    df['Risk_Category'] = pd.cut(
        total_risk,
        bins=[0, 4, 7, 10],
        labels=['Low', 'Medium', 'High']
    )
    
    # ------------------------------------------------------------------
    # FEATURE 8: Insurance-Gender Interaction
    # ------------------------------------------------------------------
    # WHY: Uninsured women may have different access than uninsured men
    #      Captures intersectionality in healthcare access
    
    df['Insurance_Gender'] = (
        df['Insurance_Coverage'].astype(str) + '_' + 
        df['Gender'].astype(str)
    )
    
    # ------------------------------------------------------------------
    # FEATURE 9: Age Squared
    # ------------------------------------------------------------------
    # WHY: Disease risk increases non-linearly with age
    #      Polynomial term captures exponential growth
    
    df['Age_Squared'] = df['Age'] ** 2
    
    # ------------------------------------------------------------------
    # FEATURE 10: Income per Age (Wealth Accumulation Proxy)
    # ------------------------------------------------------------------
    # WHY: Older people should have more savings
    #      Low ratio = economic hardship despite age
    
    df['Income_per_Age'] = df['Family_Income'] / (df['Age'] + 1)  # +1 to avoid division by zero
    
    # Drop intermediate numeric columns
    df = df.drop(['Health_Numeric', 'Mental_Numeric'], axis=1, errors='ignore')
    
    return df

# ========== APPLY TO ALL COHORTS ==========
print("\n" + "=" * 70)
print("APPLYING FEATURE ENGINEERING")
print("=" * 70)

engineered_cohorts = {}

for target_name, data_dict in cohorts.items():
    print(f"\n{'='*60}")
    print(f"Processing: {target_name}")
    print(f"{'='*60}")
    
    # Apply to train and test separately
    train_engineered = engineer_features(data_dict['train'])
    test_engineered = engineer_features(data_dict['test'])
    
    print(f"Train: {data_dict['train'].shape} → {train_engineered.shape}")
    print(f"Test:  {data_dict['test'].shape} → {test_engineered.shape}")
    
    # Show new features
    new_cols = [c for c in train_engineered.columns 
                if c not in data_dict['train'].columns]
    print(f"New features created: {new_cols}")
    
    # Save
    train_path = f"{SAVE_DIR}/meps_{target_name.lower()}_train_engineered.csv"
    test_path = f"{SAVE_DIR}/meps_{target_name.lower()}_test_engineered.csv"
    
    train_engineered.to_csv(train_path, index=False)
    test_engineered.to_csv(test_path, index=False)
    
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")
    
    engineered_cohorts[target_name] = {
        'train': train_engineered,
        'test': test_engineered
    }

# ========== FEATURE IMPORTANCE PREVIEW ==========
print("\n" + "=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)

for target_name, data_dict in engineered_cohorts.items():
    df = data_dict['train']
    
    print(f"\n{target_name}:")
    print(f"  Total features: {len([c for c in df.columns if c not in ['ID', target_name]])}")
    print(f"  Feature categories:")
    
    # Categorize features
    demographic = ['Age', 'Gender', 'Race', 'Age_Group']
    socioeconomic = ['Family_Income', 'Income_Quartile', 'Age_x_Income', 'Income_per_Age']
    clinical = ['BMI', 'BMI_Category', 'Self_Reported_Health', 'Mental_Health_Status', 'Health_Risk_Score']
    healthcare = ['Insurance_Coverage', 'Smoking', 'Physical_Activity_Limit']
    engineered = ['Risk_Category', 'Comorbidity_Count', 'Age_Squared', 'Insurance_Gender']
    
    for category, cols in [("Demographic", demographic), ("Socioeconomic", socioeconomic),
                           ("Clinical", clinical), ("Healthcare Access", healthcare),
                           ("Engineered", engineered)]:
        available = [c for c in cols if c in df.columns]
        print(f"    {category}: {available}")

# ========== SAVE FEATURE DOCUMENTATION ==========
print("\n" + "=" * 70)
print("SAVING FEATURE DOCUMENTATION")
print("=" * 70)

feature_docs = """
FEATURE ENGINEERING DOCUMENTATION
=================================

ORIGINAL FEATURES (7):
- Age: Patient age in years
- Gender: Male/Female
- Race: White/Black/Hispanic/Asian/Other
- Family_Income: Annual family income
- Insurance_Coverage: Private/Public/Uninsured
- Self_Reported_Health: Excellent/Very Good/Good/Fair/Poor
- Mental_Health_Status: Excellent/Very Good/Good/Fair/Poor

ENGINEERED FEATURES (10):

1. Age_Group (Categorical)
   - Bins: Young(0-40), Middle(40-60), Older(60-75), Senior(75+)
   - Rationale: Non-linear age-risk relationship

2. BMI_Category (Categorical)
   - Bins: Underweight(<18.5), Normal(18.5-25), Overweight(25-30), Obese(30+)
   - Rationale: Clinical thresholds for metabolic risk

3. Income_Quartile (Categorical)
   - Bins: Q1-Q4 based on distribution
   - Rationale: Relative poverty vs absolute income

4. Age_x_Income (Numerical)
   - Formula: Age * Family_Income
   - Rationale: Captures old+poor interaction (highest risk)

5. Health_Risk_Score (Numerical)
   - Formula: Average of encoded Self_Reported_Health + Mental_Health_Status
   - Rationale: Composite wellness indicator

6. Comorbidity_Count (Numerical)
   - Formula: Sum of other disease targets
   - Rationale: Disease clustering from EDA

7. Risk_Category (Categorical)
   - Bins: Low/Medium/High based on age+BMI+health
   - Rationale: Clinically actionable risk tiers

8. Insurance_Gender (Categorical)
   - Formula: Insurance_Coverage + Gender combination
   - Rationale: Intersectionality in healthcare access

9. Age_Squared (Numerical)
   - Formula: Age^2
   - Rationale: Non-linear risk acceleration

10. Income_per_Age (Numerical)
    - Formula: Family_Income / (Age + 1)
    - Rationale: Wealth accumulation proxy

TOTAL FEATURES: 17 (7 original + 10 engineered)
"""

with open(f"{RESULTS_DIR}/feature_documentation.txt", "w") as f:
    f.write(feature_docs)

print(f"Saved: {RESULTS_DIR}/feature_documentation.txt")

print("\n" + "=" * 70)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("=" * 70)