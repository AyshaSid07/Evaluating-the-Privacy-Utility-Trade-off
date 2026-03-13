""" import pandas as pd

# Load generalized dataset
df = pd.read_csv("raw_dataset.csv")

# Mask temporary identifiers
df["TMSI"] = df["TMSI"].astype(str).str[:4] + "****"
df["LMSI"] = df["LMSI"].astype(str).str[:4] + "****"
df["TLLI"] = df["TLLI"].astype(str).str[:4] + "****"

# Save dataset
df.to_csv("masked_dataset.csv", index=False)

print("Masking applied successfully") """

# masking.py
import pandas as pd

# Load raw data
df = pd.read_csv("raw_dataset.csv")

# MASKING: remove direct identifiers, mask temporary ones
df_masked = df.copy()

# Remove direct identifiers
for col in ["IMSI", "MSISDN", "IMEI", "IP_Address"]:
    if col in df_masked.columns:
        df_masked = df_masked.drop(columns=[col])

# Mask temporary identifiers (keep partial structure)
for col in ["TMSI", "LMSI", "TLLI"]:
    if col in df_masked.columns:
        df_masked[col] = df_masked[col].astype(str).str[:4] + "****"

# Save
df_masked.to_csv("masked_dataset.csv", index=False)
print("masked_dataset.csv written.")


