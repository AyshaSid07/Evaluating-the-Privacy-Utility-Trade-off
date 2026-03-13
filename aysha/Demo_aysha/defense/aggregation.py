# AGGREGATION
import pandas as pd

# Load raw data
df = pd.read_csv("raw_dataset.csv")

# AGGREGATION: collapse location to MCC (country-level)
df_agg = df.copy()

# Remove direct + temporary identifiers
for col in ["IMSI", "MSISDN", "IMEI", "IP_Address", "TMSI", "LMSI", "TLLI"]:
    if col in df_agg.columns:
        df_agg = df_agg.drop(columns=[col])

# Aggregate LAI/RAI to MCC only
if "LAI" in df_agg.columns:
    df_agg["LAI"] = df_agg["LAI"].astype(str).apply(lambda x: x.split("-")[0])

if "RAI" in df_agg.columns:
    df_agg["RAI"] = df_agg["RAI"].astype(str).apply(lambda x: x.split("-")[0])

# PLMN: keep MCC only (first 3 chars)
if "PLMN" in df_agg.columns:
    df_agg["PLMN"] = df_agg["PLMN"].astype(str).str[:3]

# Save
df_agg.to_csv("aggregated_dataset.csv", index=False)
print("aggregated_dataset.csv written.")
