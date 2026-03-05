import pandas as pd
df = pd.read_csv("../../data/suppression_only.csv")
df_masking = df.copy()

def mask_postal_code(value):
    if pd.isna(value):
        return value
    value = str(value)
    if len(value) <= 2:
        return value + "*" * (5 - len(value))
    return value[:2] + "*" * (len(value) - 2)

if "Postal Code" in df_masking.columns:
    df_masking["Postal Code"] = df_masking["Postal Code"].apply(mask_postal_code)

df_masking.to_csv("masking.csv", index=False)