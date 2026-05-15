import pandas as pd
import numpy as np

# 1. Load your dataset (assuming it is loaded into a pandas DataFrame named 'df')
df = pd.read_csv('credit-card-clients.csv')

# For demonstration, here is a slice of your exact LIMIT_BAL data
# data = {'LIMIT_BAL': [20000, 120000, 90000, 3913, 29239, 250000]}
# df = pd.DataFrame(data)

# 2. Define the edges of your bins
# This creates brackets: 0-50k, 50k-100k, 100k-200k, and everything above 200k
bins = [0, 50000, 100000, 200000, np.inf]

# 3. Define the category names (labels) for each bin
labels = [0, 1, 2, 3] # 0= 0-50k, 1 = 50k-100k, 2 = 100k-200k, 3 = above 200k

# 4. Create the new binned column using pd.cut
# right=True means the bin includes the rightmost edge (e.g., exactly 50,000 goes to 'Low')
df['LIMIT_CATEGORY'] = pd.cut(df['LIMIT_BAL'], bins=bins, labels=labels, right=True)

# 5. Drop the original continuous column so it doesn't leak into ARX
df = df.drop(columns=['LIMIT_BAL'])

print(df)

df.to_csv('../datasets/credit_card_clients_binned.csv', index=False)