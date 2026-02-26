import pandas as pd
df = pd.read_csv("../../data/mendeley_data.csv")

df_suppression = df.copy()

# Remove direct identifier (IP Address)
if "IP Address" in df_suppression.columns:
    df_suppression = df_suppression.drop(columns=["IP Address"])

df_suppression.to_csv("suppression_only.csv", index=False)
