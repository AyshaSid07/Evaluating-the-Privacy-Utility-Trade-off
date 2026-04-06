"""
Defense 4: Data Swapping
Generalize LAI to MCC-MNC level first, then swap 40% of QI values.
Without generalization, swapping alone doesn't improve k-anonymity 
because swapped unique values are still unique.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.read_csv("../csvfiles/telecom_dataset.csv")
df_swap = df.copy()

df_swap = df_swap.drop(columns=["IMSI", "MSISDN", "IMEI", "IP_Address"], errors='ignore')

# Generalize LAI/RAI to MCC-MNC first
def generalize_to_mcc_mnc(val):
    if pd.isna(val): return val
    parts = str(val).split('-')
    return f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else val

df_swap["LAI"] = df_swap["LAI"].apply(generalize_to_mcc_mnc)
df_swap["RAI"] = df_swap["RAI"].apply(generalize_to_mcc_mnc)

# Swap 40% of values in QI columns
columns_to_swap = ["PLMN", "LAI", "RAI", "Network_Type", "Device_Type"]
fraction = 0.40

for col in columns_to_swap:
    if col not in df_swap.columns: continue
    swap_idx = df_swap.sample(frac=fraction, random_state=42).index
    df_swap.loc[swap_idx, col] = np.random.permutation(df_swap.loc[swap_idx, col].values)

df_swap.to_csv("../defense/swapped_telecom.csv", index=False)
print("Defense 4: Data Swapping applied.")
print(f"  Rows: {len(df_swap)}, Unique LAI: {df_swap['LAI'].nunique()}")