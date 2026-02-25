import pandas as pd

# Load the raw dataset

df = pd.read_csv("../datasets/mendeley_data.csv")


# 1. Suppression: Remove direct identifiers

# IP Address is a direct identifier. Keeping it would allow
# an attacker to directly link a record to a device.
# So I remove it completely.
if "IP Address" in df.columns:
    df = df.drop(columns=["IP Address"])
if "Organization" in df.columns:
    df = df.drop(columns=["City"])

# 2. Masking: Hide part of the Postal Code

# Postal Code is a quasi-identifier. It reveals location,
# but I only need the broad region, not the exact address.
#
# I keep the first 2 characters and replace the rest with '*'.
# Example: 12345 → 12***
def mask_postal_code(value):
    if pd.isna(value):
        return value
    value = str(value)
    if len(value) <= 2:
        return value + "*" * (5 - len(value))
    return value[:2] + "*" * (len(value) - 2)

df = df.apply(lambda col: col.apply(mask_postal_code) if col.name in ["Region","Country","Postal Code","Latitude","Longitude","Timezone","ISP","Organization","Autonomous System"] else col)
# if "Postal Code" in df.columns:
#     df["Postal Code"] = df["Postal Code"].apply(mask_postal_code)


# 3. Generalization: Reduce precision of Latitude/Longitude

# Latitude/Longitude are highly identifying. Even without
# IP address, exact coordinates can pinpoint a person.
#
# 1 degree ≈ 111 km, so rounding reduces precision:
#   - 3 decimals (0.001°) ≈ 111 meters  → street-level area
#   - 2 decimals (0.01°)  ≈ 1.11 km     → neighborhood-level
#   - 1 decimal  (0.1°)   ≈ 11.1 km     → city-region level
#
# These three versions represent increasing anonymization:
#   v1 → light (still fairly precise)
#   v2 → medium (general neighborhood)
#   v3 → strong (only broad city region)
def apply_generalization(data, decimals):
    temp = data.copy()
    if "Latitude" in temp.columns:
        temp["Latitude"] = temp["Latitude"].round(decimals)
    if "Longitude" in temp.columns:
        temp["Longitude"] = temp["Longitude"].round(decimals)
    return temp

# Create the three anonymized versions
# v1 = apply_generalization(df, 3)   # Light anonymization (~111 m)
# v2 = apply_generalization(df, 2)   # Medium anonymization (~1.1 km)
# v3 = apply_generalization(df, 1)   # Strong anonymization (~11 km)

df.to_csv("../datasets/masked.csv", index=False)
# 4. Save the anonymized datasets

# v1.to_csv("../datasets/anonymized_v1.csv", index=False)
# v2.to_csv("../datasets/anonymized_v2.csv", index=False)
# v3.to_csv("../datasets/generalized.csv", index=False)

print("Anonymization completed.")
# print("Saved: anonymized_v1.csv, anonymized_v2.csv, anonymized_v3.csv")
