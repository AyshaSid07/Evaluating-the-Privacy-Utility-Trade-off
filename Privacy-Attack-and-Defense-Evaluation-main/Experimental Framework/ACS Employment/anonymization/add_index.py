import pandas as pd

#load original folktable employment dataset
df_original = pd.read_csv('../datasets/folktables_employment_RAW.csv')

#add a unique linkage index column as the first column 
df_original.insert(0, 'Linkage_Index', df_original.index)

#save the updated dataset to CSV file
df_original.to_csv('../datasets/folktables_employment_RAW.csv', index=False)