import pandas as pd
import numpy as np
import os

print("=" * 60)
print("FIXING BMI IN TRAIN/TEST FILES")
print("=" * 60)

save_dir = "meps_data"

# Step 1: Extract BMI + ID from raw .dta
print("\n📦 Extracting BMI from h224.dta...")
raw = pd.read_stata(os.path.join(save_dir, "h224.dta"), convert_categoricals=False)
bmi_data = pd.DataFrame({
    'ID': raw['DUPERSID'].astype(str),
    'BMI': pd.to_numeric(raw['ADBMI42'], errors='coerce')
})
bmi_data['BMI'] = bmi_data['BMI'].replace([-1, -7, -8, -9, -15, -2], np.nan)
print(f"   Extracted {bmi_data['BMI'].notna().sum():,} real BMI values")

# Step 2: Fix each train/test file
targets = ['diabetes', 'hypertension', 'heart_disease']
for name in targets:
    for split in ['train', 'test']:
        path = os.path.join(save_dir, f"meps_{name}_{split}.csv")
        if not os.path.exists(path):
            print(f"   ❌ {path} not found, skipping")
            continue
        
        df = pd.read_csv(path)
        
        # Check if BMI already exists
        if 'BMI' in df.columns:
            print(f"   ✅ meps_{name}_{split}.csv already has BMI")
            continue
        
        # Merge BMI by ID
        df['ID'] = df['ID'].astype(str)
        df = df.merge(bmi_data, on='ID', how='left')
        
        # Save back
        df.to_csv(path, index=False)
        print(f"   ✅ Added BMI to meps_{name}_{split}.csv ({df['BMI'].notna().sum():,} real values)")

print("\n🎉 Done! Now run your feature engineering script.")