""" import pandas as pd

# Load raw dataset

df = pd.read_csv("../defense/raw_dataset.csv")

# Direct identifiers to remove
direct_identifiers = ["IMSI", "MSISDN", "IMEI", "IP_Address"]

# Remove columns
df_suppressed = df.drop(columns=direct_identifiers)

# Save new dataset
df_suppressed.to_csv("suppressed_dataset.csv", index=False)

print("Direct identifiers removed successfully")
print("Remaining columns:")
print(df_suppressed.columns) """


# suppression.py
import pandas as pd

# Load raw data
df = pd.read_csv("raw_dataset.csv")

# SUPPRESSION: remove identifiers + fine-grained location (keep MCC-MNC-LAC)
df_supp = df.copy()

# Remove direct + temporary identifiers
for col in ["IMSI", "MSISDN", "IMEI", "IP_Address", "TMSI", "LMSI", "TLLI"]:
    if col in df_supp.columns:
        df_supp = df_supp.drop(columns=[col])

# Suppress fine-grained location (drop RAC)
if "LAI" in df_supp.columns:
    df_supp["LAI"] = df_supp["LAI"].astype(str).apply(
        lambda x: "-".join(x.split("-")[:3]) if "-" in x else x
    )

if "RAI" in df_supp.columns:
    df_supp["RAI"] = df_supp["RAI"].astype(str).apply(
        lambda x: "-".join(x.split("-")[:3]) if "-" in x else x
    )

# PLMN: keep MCC-MNC (first 5 chars if formatted that way)
if "PLMN" in df_supp.columns:
    df_supp["PLMN"] = df_supp["PLMN"].astype(str).str[:5]

# Save
df_supp.to_csv("suppressed_dataset.csv", index=False)
print("suppressed_dataset.csv written.")






