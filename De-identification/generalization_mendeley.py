"""
Generalization
Generalize the mendeley dataset by removing city and postal code to make it less specific.
"""
import pandas as pd

df = pd.read_csv("../datasets/de_identified_datasets/suppressed_DI_mendeley.csv")

# Remove city and postal code
df = df.drop(columns=["City", "Postal Code"], errors='ignore')
df.to_csv("../datasets/de_identified_datasets/generalization_mendeley.csv", index=False)