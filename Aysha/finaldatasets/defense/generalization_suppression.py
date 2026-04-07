Defense 5: Generalization + Attribute Suppression
Generalize PLMN to MCC, LAI/RAI to MCC only (country level - coarsest).
Then suppress TMSI, LMSI, TLLI, Timestamp.

import pandas as pd

df = pd.read_csv("../csvfiles/telecom_dataset.csv")
df_comb = df.copy()

# Generalize to MCC only 
df_comb["PLMN"] = df_comb["PLMN"].astype(str).str[:3]

# LAI → MCC only (e.g, 240-07-5012 → 240)
def generalize_lai_mcc(val):
    if pd.isna(val): return val
    parts = str(val).split('-')
    return parts[0] if len(parts) >= 1 else val

df_comb["LAI"] = df_comb["LAI"].apply(generalize_lai_mcc)
df_comb["RAI"] = df_comb["RAI"].apply(generalize_lai_mcc)

# Suppress columns
cols_to_drop = ["IMSI", "MSISDN", "IMEI", "IP_Address", "TMSI", "LMSI", "TLLI", "Timestamp"]
df_comb = df_comb.drop(columns=[c for c in cols_to_drop if c in df_comb.columns])

df_comb.to_csv("../defense/generalized_and_suppressed_telecom.csv", index=False)
print("Defense 5: Generalization + Suppression applied.")
print(f"  Rows: {len(df_comb)}, Unique LAI: {df_comb['LAI'].nunique()}")