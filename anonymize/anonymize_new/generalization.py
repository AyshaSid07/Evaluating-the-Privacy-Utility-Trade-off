import pandas as pd
df = pd.read_csv("../../data/mendeley.csv")


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