Defense 3: Cell Suppression
Generalize LAI to MCC-MNC level first, then suppress rare cell values (count < k=5).

import pandas as pd

df = pd.read_csv("../csvfiles/telecom_dataset.csv")
df_cell = df.copy()

df_cell = df_cell.drop(columns=["IMSI", "MSISDN", "IMEI", "IP_Address"], errors='ignore')

# Generalize LAI to MCC-MNC-partial_LAC (keep first 2 digits of LAC)
# e.g., 240-07-5012 → 240-07-50 (coarser than raw but finer than MCC-MNC)
def generalize_lai_partial(val):
    if pd.isna(val): return val
    parts = str(val).split('-')
    if len(parts) >= 3:
        return f"{parts[0]}-{parts[1]}-{parts[2][:2]}"
    return val

def generalize_rai_partial(val):
    if pd.isna(val): return val
    parts = str(val).split('-')
    if len(parts) >= 3:
        return f"{parts[0]}-{parts[1]}-{parts[2][:2]}"
    return val

df_cell["LAI"] = df_cell["LAI"].apply(generalize_lai_partial)
df_cell["RAI"] = df_cell["RAI"].apply(generalize_rai_partial)

# Cell suppression: replace rare values with SUPPRESSED
quasi_identifiers = ['Device_Type', 'Network_Type', 'PLMN', 'LAI']
k = 5

for qi in quasi_identifiers:
    if qi not in df_cell.columns: continue
    counts = df_cell[qi].value_counts()
    rare = counts[counts < k].index
    df_cell.loc[df_cell[qi].isin(rare), qi] = "SUPPRESSED"

df_cell.to_csv("../defense/cell_suppressed_telecom.csv", index=False)
print("Defense 3: Cell Suppression applied.")
print(f"  Rows: {len(df_cell)}, Unique LAI: {df_cell['LAI'].nunique()}")