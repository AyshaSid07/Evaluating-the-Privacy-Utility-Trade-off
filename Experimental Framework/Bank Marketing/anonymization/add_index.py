import pandas as pd

df_original = pd.read_csv('../datasets/bank-additional-full.csv', sep=';')

df_original.insert(0, 'Linkage_Index', df_original.index)

df_original.to_csv('../datasets/bank-additional-full.csv', sep=';', index=False)