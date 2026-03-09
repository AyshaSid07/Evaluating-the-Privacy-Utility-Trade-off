import pandas as pd

# Load raw dataset
df = pd.read_csv("../raw_dataset.csv")

# Direct identifiers to remove
direct_identifiers = ["IMSI", "MSISDN", "IMEI", "IP_Address"]

# Remove columns
df_suppressed = df.drop(columns=direct_identifiers)

# Save new dataset
df_suppressed.to_csv("suppressed_dataset.csv", index=False)

print("Direct identifiers removed successfully")
print("Remaining columns:")
print(df_suppressed.columns)