""" import pandas as pd

df = pd.read_csv("raw_dataset.csv")

# Generalize LAI
df["LAI"] = df["LAI"].str.split("-").str[0] + "-" + df["LAI"].str.split("-").str[1]

# Generalize RAI
df["RAI"] = df["RAI"].str.split("-").str[0] + "-" + df["RAI"].str.split("-").str[1]

df.to_csv("generalized_dataset.csv", index=False)

print("Generalization completed") """

# generalization.py
import pandas as pd

# Load raw data
df = pd.read_csv("raw_dataset.csv")

# GENERALIZATION: reduce location to MCC-MNC (region-level)
df_gen = df.copy()

# Remove direct + temporary identifiers
for col in ["IMSI", "MSISDN", "IMEI", "IP_Address", "TMSI", "LMSI", "TLLI"]:
    if col in df_gen.columns:
        df_gen = df_gen.drop(columns=[col])

# Generalize LAI/RAI to MCC-MNC
def to_mcc_mnc(val):
    parts = str(val).split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return str(val)

if "LAI" in df_gen.columns:
    df_gen["LAI"] = df_gen["LAI"].apply(to_mcc_mnc)

if "RAI" in df_gen.columns:
    df_gen["RAI"] = df_gen["RAI"].apply(to_mcc_mnc)

# PLMN: keep MCC-MNC (first 5 chars)
if "PLMN" in df_gen.columns:
    df_gen["PLMN"] = df_gen["PLMN"].astype(str).str[:5]

# Save
df_gen.to_csv("generalized_dataset.csv", index=False)
print("generalized_dataset.csv written.")






