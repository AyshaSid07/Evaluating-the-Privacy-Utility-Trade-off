import pandas as pd

df_gen = pd.read_csv("suppression_only.csv")

# 1. Spatial Generalization (Quasi-Identifiers)
# Masks everything after MCC-MNC (e.g., 240-07-XXXX)
def mask_location(val):
    if pd.isna(val) or val == "": return val
    parts = str(val).split('-')
    return f"{parts[0]}-{parts[1]}-XXXX" if len(parts) >= 2 else val

for col in ['LAI', 'RAI', 'PLMN']:
    if col in df_gen.columns:
        df_gen[col] = df_gen[col].apply(mask_location)

# 2. Suppress micro-location columns (LAC, RAC, TAC, Cell_ID)
# These are too unique for k-anonymity; they must be removed to form groups
extra_ids = ['LAC', 'RAC', 'TAC', 'Cell_ID']
df_gen = df_gen.drop(columns=[c for c in extra_ids if c in df_gen.columns], errors='ignore')

# 3. Categorical Generalization
if 'Device_Type' in df_gen.columns:
    df_gen['Device_Type'] = df_gen['Device_Type'].replace(r'.*Phone.*', 'Smartphone', regex=True)

df_gen.to_csv("generalization.csv", index=False)
print("Generalization Complete.")