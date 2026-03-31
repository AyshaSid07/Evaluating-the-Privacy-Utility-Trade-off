import pandas as pd

# We go back to the base suppression file and remove all linkage keys
df_max = pd.read_csv("suppression_only.csv")

# Delete the columns used to join with external data (Quasi-Identifiers)
bridge_cols = ['PLMN', 'LAI', 'RAI', 'Timestamp', 'MCC', 'MNC', 'LAC', 'RAC']
df_max = df_max.drop(columns=[c for c in bridge_cols if c in df_max.columns], errors='ignore')

# What remains is disconnected from any individual identity
df_max.to_csv("suppression_and_swapping.csv", index=False)
print("suppression and swapping complete.")