"""
Data Swapping 
Swap values in a row with another row of the same quasi-identifier.
How many rows to swap can be controlled by the fraction parameter.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("../datasets/de_identified_datasets/suppressed_DI_telecom.csv", dtype={'MNC': str})

quasi_identifiers_to_swap = ["RAI", "LAI", "PLMN"] # we can choose which quasi-identifiers to swap
fraction = 0.30 # how much of the data to swap, e.g., 0.3 = 30% of the rows in the specified columns will be swapped

for col in quasi_identifiers_to_swap:
    if col in df.columns:
        swap_idx = df.sample(frac=fraction, random_state=42).index
        df.loc[swap_idx, col] = np.random.permutation(df.loc[swap_idx, col])

df.to_csv("../datasets/de_identified_datasets/swapped_dataset_telecom.csv", index=False)