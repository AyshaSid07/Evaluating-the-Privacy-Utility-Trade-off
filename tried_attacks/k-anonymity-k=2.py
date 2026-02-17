import pandas as pd

# Load the masked dataset
df = pd.read_csv("ip_only_safe.csv")

# Set the value of k
k = 2

# Select the quasi-identifiers
# assuming these attributes an attacker might know
quasi_identifiers = ["Region", "IP_Masked", "ISP"]

# Counting how many times each combination appears
group_sizes = df.groupby(quasi_identifiers)["Region"].transform("size")

# Keep only the records where the group size is at least k
df_k = df[group_sizes >= k]

#Save the new anonymized dataset
df_k.to_csv("defended.csv", index=False)

print("K-anonymity defense applied successfully")
print("Number of remaining records:", len(df_k))


