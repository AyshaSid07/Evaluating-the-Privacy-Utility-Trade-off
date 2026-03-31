import pandas as pd
import numpy as np

df_swap = pd.read_csv("generalization.csv")

# 1. Perform 30% attribute swapping on sensitive columns
# This protects against Attribute Disclosure even if k-anonymity is low
cols_to_shuffle = ['RAI', 'Device_Type', 'Network_Type']

for col in cols_to_shuffle:
    if col in df_swap.columns:
        n_to_swap = int(len(df_swap) * 0.30)
        idx = np.random.choice(df_swap.index, size=n_to_swap, replace=False)
        df_swap.loc[idx, col] = np.random.permutation(df_swap.loc[idx, col].values)

df_swap.to_csv("swapping.csv", index=False)
print("Swapping Complete.")