import pandas as pd

df_original = pd.read_csv('../datasets/credit_card_clients_binned.csv')

df_original.insert(0, 'Linkage_Index', df_original.index)

df_original.to_csv('../datasets/credit_card_clients_binned.csv', index=False)