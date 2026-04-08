"""
Mask values in the data
For the telecom data, mask LAI and RAI to only show MCC-MNC and replace the rest with stars.
"""
import pandas as pd

df = pd.read_csv("../datasets/de_identified_datasets/suppressed_DI_telecom.csv", dtype={'MNC': str})


# mask LAI
df['LAI'] = df['LAI'].astype(str).apply(
    lambda x: '-'.join(x.split('-')[:2]) + '-****' if pd.notnull(x) else x
)

# mask RAI
df['RAI'] = df['RAI'].astype(str).apply(
    lambda x: '-'.join(x.split('-')[:2]) + '-****-**' if pd.notnull(x) else x
)

df.to_csv("../datasets/de_identified_datasets/masking_telecom.csv", index=False)