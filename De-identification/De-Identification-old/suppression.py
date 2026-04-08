import pandas as pd
df = pd.read_csv("../datasets/synthetic_telecom_data.csv")

df_suppression = df.copy()

if "Network_Type" in df_suppression.columns:
    df_suppression = df_suppression.drop(columns=["Network_Type"])

df_suppression.to_csv("../datasets/de-identified-datasets/suppressed_network_type_telecom.csv", index=False)
