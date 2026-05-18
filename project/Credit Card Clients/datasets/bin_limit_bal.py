import pandas as pd
import numpy as np

df = pd.read_csv('credit-card-clients.csv')

bins = [0, 50000, 100000, 200000, np.inf]

labels = [0, 1, 2, 3] # 0= 0-50k, 1 = 50k-100k, 2 = 100k-200k, 3 = above 200k

df['LIMIT_CATEGORY'] = pd.cut(df['LIMIT_BAL'], bins=bins, labels=labels, right=True)

df = df.drop(columns=['LIMIT_BAL'])

print(df)

df.to_csv('../datasets/credit_card_clients_binned.csv', index=False)