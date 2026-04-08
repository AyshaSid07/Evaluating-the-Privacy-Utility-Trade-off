"""
Suppress Direct Identifiers
Drop direct identifiers (IP Address) from the mendeley dataset.
"""
import pandas as pd

df = pd.read_csv("../datasets/mendeley_dataset.csv")
df = df.drop(columns=["IP Address"], errors='ignore')
df.to_csv("../datasets/de_identified_datasets/suppressed_DI_mendeley.csv", index=False)