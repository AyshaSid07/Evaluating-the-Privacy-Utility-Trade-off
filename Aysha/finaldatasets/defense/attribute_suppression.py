"""
Defense 2: Attribute Suppression
Drop direct IDs + suppress LAI, RAI, TMSI, LMSI, TLLI, Timestamp entirely.
This forces attacks to use only Device_Type, Network_Type, PLMN as QIs.
"""
import pandas as pd

df = pd.read_csv("../csvfiles/telecom_dataset.csv")
df_sup = df.copy()

columns_to_drop = [
    "IMSI", "MSISDN", "IMEI", "IP_Address",  # direct identifiers
    "TMSI", "LMSI", "TLLI", "Timestamp",      # quasi-identifiers
    "LAI", "RAI"                                # suppress LAI and RAI entirely
]

df_sup = df_sup.drop(columns=[c for c in columns_to_drop if c in df_sup.columns])

df_sup.to_csv("../defense/attribute_suppressed_telecom.csv", index=False)
print("Defense 2: Attribute Suppression applied.")
print(f"  Rows: {len(df_sup)}, Columns: {list(df_sup.columns)}")