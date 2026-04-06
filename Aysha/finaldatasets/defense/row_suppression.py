Defense 6: Row Suppression for k-Anonymity (k=5)
Generalize LAI to MCC-MNC, then remove rows where group size < k.

import pandas as pd

df = pd.read_csv("../csvfiles/telecom_dataset.csv")
df_kanon = df.copy()

df_kanon = df_kanon.drop(columns=["IMSI", "MSISDN", "IMEI", "IP_Address"], errors='ignore')

# Generalize LAI/RAI to MCC-MNC
def generalize_to_mcc_mnc(val):
    if pd.isna(val): return val
    parts = str(val).split('-')
    return f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else val

df_kanon["LAI"] = df_kanon["LAI"].apply(generalize_to_mcc_mnc)
df_kanon["RAI"] = df_kanon["RAI"].apply(generalize_to_mcc_mnc)

quasi_identifiers = ['Device_Type', 'Network_Type', 'PLMN', 'LAI']
k = 5

group_sizes = df_kanon.groupby(quasi_identifiers).transform('size')
df_kanon = df_kanon[group_sizes >= k].reset_index(drop=True)

df_kanon.to_csv("../defense/row_suppressed_telecom.csv", index=False)
print("Defense 6: Row Suppression k-Anonymity applied.")
print(f"  Rows: {len(df_kanon)}, Data loss: {((3000-len(df_kanon))/3000)*100:.1f}%")