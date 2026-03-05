import pandas as pd

# Load the same dataset used in the attack
df = pd.read_csv("ip_only_safe.csv")

# Using the same attributes claimed in the attack
city = "City of Westminster"
org = "Trustmarque Solutions Limited"

# Filter records matching the attack profile
matches = df[
    (df["City"] == city) &
    (df["Organization"] == org)
]

print("Number of matching records:", len(matches))
print(matches)
