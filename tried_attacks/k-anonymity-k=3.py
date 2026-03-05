import pandas as pd

# Load dataset
df = pd.read_csv("ip_only_safe.csv")

k = 3

# Generalize City to Region
df["City"] = df["Region"]

#Count group sizes once
counts = df.groupby(["City", "Organization"]).size().reset_index(name="count")

# Merge counts back to original dataframe
df = df.merge(counts, on=["City", "Organization"])

#Replace small groups
df.loc[df["count"] < k, "Organization"] = "Other"

# Recalculate group sizes
counts2 = df.groupby(["City", "Organization"]).size().reset_index(name="count2")

df = df.merge(counts2, on=["City", "Organization"])

# Keep only rows satisfying k
df_final = df[df["count2"] >= k]

# Drop helper columns
df_final = df_final.drop(columns=["count", "count2"])

# Save file
df_final.to_csv("defense1.csv", index=False)

print("Defense completed successfully")
print("Remaining records:", len(df_final))
