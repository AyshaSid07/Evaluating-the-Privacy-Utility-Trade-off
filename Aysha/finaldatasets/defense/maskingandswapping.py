import pandas as pd
import numpy as np

df_ms = pd.read_csv("swapping.csv")

# 2nd Layer of Shuffling (Increased to 50% for Very Strong tier)
# We shuffle Quasi-Identifiers to maximize uncertainty
quasi_cols = ['LAI', 'PLMN', 'Network_Type']

for col in quasi_cols:
    if col in df_ms.columns:
        n = int(len(df_ms) * 0.50)
        idx = np.random.choice(df_ms.index, size=n, replace=False)
        df_ms.loc[idx, col] = np.random.permutation(df_ms.loc[idx, col].values)

df_ms.to_csv("masking_and_swapping.csv", index=False)
print("Masking and Swapping Complete.")