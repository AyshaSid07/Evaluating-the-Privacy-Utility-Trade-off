"""
Row Suppression for a k value in k-Anonymity
Remove rows where group size < k.
"""

import pandas as pd

df_k_anon = pd.read_csv("../datasets/de_identified_datasets/suppressed_DI_mendeley.csv")

quasi_identifiers = ['City', 'Region', 'Country', 'Postal Code', 'Latitude', 'Longitude', 'Timezone', 'ISP', 'Organization', 'Autonomous System']

k = 2 # change k as needed 

group_sizes = df_k_anon.groupby(quasi_identifiers).transform('size')
df_k_anon = df_k_anon[group_sizes >= k].reset_index(drop=True)

df_k_anon.to_csv("../datasets/de_identified_datasets/row_suppression_mendeley.csv", index=False)