""" import pandas as pd
df = pd.read_csv("../datasets/telecom_dataset.csv")

df_suppression = df.copy()

if "Network_Type" in df_suppression.columns:
    df_suppression = df_suppression.drop(columns=["Network_Type"])

df_suppression.to_csv("../defense/suppressed_dataset.csv", index=False) """

import pandas as pd

# 1. Load the dataset (using 'telecomb' and correct relative path)
df = pd.read_csv("../datasets/telecomb_dataset.csv")

df_suppression = df.copy()

# 2. Columns to suppress (direct identifiers)
columns_to_suppress = ["IMSI", "MSISDN", "IMEI", "IP_Address"]

# 3. Drop only these columns if they exist
df_suppression = df_suppression.drop(
    columns=[c for c in columns_to_suppress if c in df_suppression.columns]
)

# 4. Save to the defense folder
df_suppression.to_csv("suppressed_dataset.csv", index=False)

print(f"Suppressed dataset created successfully. Removed: {columns_to_suppress}")