import pandas as pd
df = pd.read_csv("../datasets/de-identified-datasets/masking.csv")


def apply_generalization(data, decimals):
    temp = data.copy()
    if "Latitude" in temp.columns:
        temp["Latitude"] = temp["Latitude"].round(decimals)
    if "Longitude" in temp.columns:
        temp["Longitude"] = temp["Longitude"].round(decimals)
    return temp

generalization = apply_generalization(df, 1)

generalization.to_csv("../datasets/de-identified-datasets/generalization_and_masking.csv", index=False)