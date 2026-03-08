import pandas as pd

df = pd.read_csv("suppressed_dataset.csv")

# Generalize LAI
df["LAI"] = df["LAI"].str.split("-").str[0] + "-" + df["LAI"].str.split("-").str[1]

# Generalize RAI
df["RAI"] = df["RAI"].str.split("-").str[0] + "-" + df["RAI"].str.split("-").str[1]

df.to_csv("generalized_dataset.csv", index=False)

print("Generalization completed")