"""
Rounding
Round the latitude and longitude values to certain decimal
The decimal parameter controls how much to round, e.g., 1 = round to 1 decimal place, 2 = round to 2 decimal places, etc.
"""
import pandas as pd
df = pd.read_csv("../datasets/de_identified_datasets/suppressed_DI_mendeley.csv")

decimals = 1 # how many decimals to round 

df["Latitude"] = df["Latitude"].round(decimals)
df["Longitude"] = df["Longitude"].round(decimals)

df.to_csv("../datasets/de_identified_datasets/rounding_mendeley.csv", index=False)