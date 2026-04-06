"""
Defense 1: Generalization
Generalize LAI to MCC-MNC level, PLMN to MCC, RAI to MCC-MNC.
"""
import pandas as pd

df = pd.read_csv("../csvfiles/telecom_dataset.csv")
df_gen = df.copy()

# Remove direct identifiers
df_gen = df_gen.drop(columns=["IMSI", "MSISDN", "IMEI", "IP_Address"], errors='ignore')

# Generalize PLMN → MCC only
df_gen["PLMN"] = df_gen["PLMN"].astype(str).str[:3]

# Generalize LAI → MCC-MNC (e.g., 240-07-5012 → 240-07)
def generalize_lai_mcc_mnc(val):
    if pd.isna(val): return val
    parts = str(val).split('-')
    return f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else val

# Generalize RAI → MCC-MNC
def generalize_rai_mcc_mnc(val):
    if pd.isna(val): return val
    parts = str(val).split('-')
    return f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else val

df_gen["LAI"] = df_gen["LAI"].apply(generalize_lai_mcc_mnc)
df_gen["RAI"] = df_gen["RAI"].apply(generalize_rai_mcc_mnc)

df_gen.to_csv("../defense/generalized_telecom.csv", index=False)
print("Defense 1: Generalization applied.")
print(f"  Rows: {len(df_gen)}, Unique LAI: {df_gen['LAI'].nunique()}")