import pandas as pd
import numpy as np

# 1. Load the specific dataset
# Path adjusted based on your terminal location
df = pd.read_csv("../datasets/telecomb_dataset.csv")
df_swapped = df.copy()

# 2. Settings
fraction = 0.40  
columns_to_swap = ["RAI", "LAI", "TMSI", "LMSI", "TLLI"]

# 3. Apply Swapping
for col in columns_to_swap:
    if col in df_swapped.columns:
        # Pick random indices
        swap_idx = df_swapped.sample(frac=fraction, random_state=42).index
        
        # .tolist() fixes the 'read-only' ValueError and the StringArray warning
        values = df_swapped.loc[swap_idx, col].tolist()
        
        # Shuffle the list
        np.random.seed(42) 
        np.random.shuffle(values)
        
        # Put the shuffled values back
        df_swapped.loc[swap_idx, col] = values

# 4. Save the result
df_swapped.to_csv("swapped_dataset.csv", index=False)

print(f"Success! Swapped {fraction*100}% of {columns_to_swap}")