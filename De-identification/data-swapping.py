import pandas as pd
import numpy as np

df = pd.read_csv("../datasets/mendeley_data.csv")

def apply_swapping(data, columns_to_swap):
    temp = data.copy()
    fraction = 0.30 # how much of the data to swap, e.g., 15% of the rows in the specified columns will be swapped
    
    for col in columns_to_swap:
        if col in temp.columns:
            swap_idx = temp.sample(frac=fraction, random_state=42).index
            temp.loc[swap_idx, col] = np.random.permutation(temp.loc[swap_idx, col])
            
    return temp

quasi_identifiers_to_swap = ["IP Address", "City", "ISP", "Latitude", "Longitude"] # we can choose which quasi-identifiers to swap
swapped_df = apply_swapping(df, quasi_identifiers_to_swap)

swapped_df.to_csv("../datasets/de-identified-datasets/swapped_data.csv", index=False)