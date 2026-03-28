import pandas as pd
import numpy as np

df = pd.read_csv("../finaldatasets/mendeley_dataset.csv")

cols_to_check = [
    'City', 'Region', 'Country', 'Postal Code', 'Latitude', 
    'Longitude', 'Timezone', 'ISP', 'Organization', 'Autonomous System'
]

df.replace('N/A', np.nan, inplace=True)

df = df.dropna(subset=cols_to_check, how='all')

df.replace('TELEF�NICA BRASIL S.A', 'TELEFONICA BRASIL S.A', inplace=True)
df.to_csv("../mendeley_dataset.csv", index=False)
