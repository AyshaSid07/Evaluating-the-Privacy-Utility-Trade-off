import pandas as pd

df_original = pd.read_csv('../datasets/folktables_income_RAW.csv')

df_original.insert(0, 'Linkage_Index', df_original.index)

df_original.to_csv('../datasets/folktables_income_RAW.csv', index=False)