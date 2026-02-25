import pandas as pd

# Load the raw dataset
df = pd.read_csv("../../data/mendeley_data.csv")

# 1. SUPPRESSION ONLY

df_suppression = df.copy()

# Remove direct identifier (IP Address)
if "IP Address" in df_suppression.columns:
    df_suppression = df_suppression.drop(columns=["IP Address"])

df_suppression.to_csv("suppression_only.csv", index=False)


# 2. MASKING ONLY

df_masking = df.copy()

def mask_postal_code(value):
    if pd.isna(value):
        return value
    value = str(value)
    if len(value) <= 2:
        return value + "*" * (5 - len(value))
    return value[:2] + "*" * (len(value) - 2)

if "Postal Code" in df_masking.columns:
    df_masking["Postal Code"] = df_masking["Postal Code"].apply(mask_postal_code)

df_masking.to_csv("masking_only.csv", index=False)


# 3. GENERALIZATION ONLY

def apply_generalization(data, decimals):
    temp = data.copy()
    if "Latitude" in temp.columns:
        temp["Latitude"] = temp["Latitude"].round(decimals)
    if "Longitude" in temp.columns:
        temp["Longitude"] = temp["Longitude"].round(decimals)
    return temp

generalization_v1 = apply_generalization(df, 3)
generalization_v2 = apply_generalization(df, 2)
generalization_v3 = apply_generalization(df, 1)

generalization_v1.to_csv("generalization_v1.csv", index=False)
generalization_v2.to_csv("generalization_v2.csv", index=False)
generalization_v3.to_csv("generalization_v3.csv", index=False)


# 4. GENERALIZATION + SUPPRESSION TOGETHER

# First apply suppression
df_gen_sup = df.copy()

if "IP Address" in df_gen_sup.columns:
    df_gen_sup = df_gen_sup.drop(columns=["IP Address"])

# apply generalization at 3 levels
gen_sup_v1 = apply_generalization(df_gen_sup, 3)
gen_sup_v2 = apply_generalization(df_gen_sup, 2)
gen_sup_v3 = apply_generalization(df_gen_sup, 1)

gen_sup_v1.to_csv("generalization_suppression_v1.csv", index=False)
gen_sup_v2.to_csv("generalization_suppression_v2.csv", index=False)
gen_sup_v3.to_csv("generalization_suppression_v3.csv", index=False)


print("Anonymization completed.")
print("Files saved:")
print("- suppression_only.csv")
print("- masking_only.csv")
print("- generalization_v1.csv")
print("- generalization_v2.csv")
print("- generalization_v3.csv")
print("- generalization_suppression_v1.csv")
print("- generalization_suppression_v2.csv")
print("- generalization_suppression_v3.csv")
