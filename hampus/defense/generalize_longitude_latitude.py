import pandas as pd
df = pd.read_csv('../datasets/mendeley_data.csv')
df_protected = df.copy()
df_protected = df_protected.drop(columns=['IP Address']) 
df_protected['Latitude'] = df_protected['Latitude'].round(0)
df_protected['Longitude'] = df_protected['Longitude'].round(0)

df_protected.to_csv('../datasets/generalized_lat_long.csv', index=False)