import pandas as pd
df = pd.read_csv("../datasets/mendeley_data.csv")

df_suppression = df.copy()

# Remove direct identifier (IP Address)
if "City" in df_suppression.columns:
    df_suppression = df_suppression.drop(columns=["City"])

df_suppression.to_csv("../datasets/de-identified-datasets/suppressed_city.csv", index=False)
