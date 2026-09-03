# Patient Risk Analysis — MEPS 2020

Predicts Diabetes, Hypertension, and Heart Disease risk using 
Medical Expenditure Panel Survey (MEPS) 2020 data.

## Pipeline
1. **Data Cleaning** — 27,805 patients, 1,451 raw features → 17 clean features
2. **KNN Imputation** — Target-specific cohorts, train/test split before imputation
3. **EDA** — 8 visualizations revealing class imbalance & socioeconomic gradients
4. **Feature Engineering** — 17 features including BMI×Age, Poly_Risk, Income_Gap
5. **Model Training** — LR, RF, XGB, Stacking with SMOTE + threshold tuning
6. **Explainability** — SHAP values + Fairness audit across Race/Gender/Insurance
7. **Deployment** — Batch prediction script with risk tier stratification

## Results
| Disease | Best Model | ROC-AUC | F1 (tuned) | Key Drivers |
|---------|-----------|---------|------------|-------------|
| Diabetes | Logistic Regression | 0.859 | 0.466 | Age, BMI, Income, Self-Reported Health |
| Hypertension | Logistic Regression | 0.839 | 0.715 | Age, BMI, Comorbidity |
| Heart Disease | Logistic Regression | 0.830 | 0.442 | Age, Hypertension, Smoking |


deployment link: https://patient-risk-analysis-rgz4xe5rftbqezgdwakxn7.streamlit.app/

## Run
```bash
pip install -r requirements.txt
python 05_model_training.py      # Baseline
python 06_model_improvement.py   # Improved
python 07_explainability.py      # SHAP + Fairness
python 08_predict.py --input new_patients.csv --target Diabetes --output out.csv
