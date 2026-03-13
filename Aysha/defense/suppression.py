""" import pandas as pd
df = pd.read_csv("../datasets/telecom_dataset.csv")

df_suppression = df.copy()

if "Network_Type" in df_suppression.columns:
    df_suppression = df_suppression.drop(columns=["Network_Type"])

df_suppression.to_csv("../defense/suppressed_dataset.csv", index=False) """

import pandas as pd

# Load raw dataset
df = pd.read_csv("../datasets/telecom_dataset.csv")

df_suppression = df.copy()

# Columns to suppress (identifiers)
columns_to_suppress = ["IMSI", "MSISDN", "IMEI", "IP_Address"]

# Drop only these columns if they exist
df_suppression = df_suppression.drop(
    columns=[c for c in columns_to_suppress if c in df_suppression.columns]
)

# Save anonymized dataset
df_suppression.to_csv("../defense/suppressed_dataset.csv", index=False)

print("Suppressed dataset created successfully")