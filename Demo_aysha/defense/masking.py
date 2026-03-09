import pandas as pd

# Load generalized dataset
df = pd.read_csv("generalized_dataset.csv")

# Mask temporary identifiers
df["TMSI"] = df["TMSI"].astype(str).str[:4] + "****"
df["LMSI"] = df["LMSI"].astype(str).str[:4] + "****"
df["TLLI"] = df["TLLI"].astype(str).str[:4] + "****"

# Save dataset
df.to_csv("masked_dataset.csv", index=False)

print("Masking applied successfully")