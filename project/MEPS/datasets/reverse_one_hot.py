import pandas as pd
import numpy as np

input_file = "MEPS19_RAW.csv"  
df = pd.read_csv(input_file)

one_hot_cols = [col for col in df.columns if '=' in col]
prefixes = set([col.split('=')[0] for col in one_hot_cols])

for prefix in prefixes:
    cols = [col for col in df.columns if col.startswith(prefix + '=')]
    
    
    has_one = df[cols].max(axis=1) > 0
    category_values = df[cols].idxmax(axis=1).str.split('=').str[1]
    
    df[prefix] = np.where(has_one, category_values.astype(float), np.nan)
    
    df = df.drop(columns=cols)

output_file = "MEPS.csv"
df.to_csv(output_file, index=False)
