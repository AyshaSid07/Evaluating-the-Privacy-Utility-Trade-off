import pandas as pd

# Load dataset and keep MCC and MNC as strings (to preserve leading zeros)
df = pd.read_csv("newtelecom_dataset.csv", dtype={"MCC": str, "MNC": str})

# Fix formatting (important for correct matching)
df["MNC"] = df["MNC"].str.zfill(2)
df["MCC"] = df["MCC"].astype(str)

print("Dependency Checks\n")

# 1. Check how many operators (MNC) exist per country (MCC)
print("MCC to MNC mapping")
print(df.groupby("MCC")["MNC"].nunique())

# 2. Check relationship between MCC and Network_Type
print("\nMCC vs Network_Type")
print(pd.crosstab(df["MCC"], df["Network_Type"]))

# 3. Check relationship between MCC and Device_Type
print("\nMCC vs Device_Type")
print(pd.crosstab(df["MCC"], df["Device_Type"]))

# 4. Check if PLMN is correctly formed as MCC + MNC
df["PLMN_check"] = df["MCC"] + df["MNC"]
mismatch = (df["PLMN"].astype(str) != df["PLMN_check"]).sum()

print("\nPLMN check")
print("Number of mismatches:", mismatch)

# 5. Check if LAI starts with MCC-MNC
def check_lai(row):
    expected = row["MCC"] + "-" + row["MNC"]
    return str(row["LAI"]).startswith(expected)

lai_errors = (~df.apply(check_lai, axis=1)).sum()

print("\nLAI check")
print("Number of errors:", lai_errors)

# 6. Check if RAI starts with LAI
def check_rai(row):
    return str(row["RAI"]).startswith(str(row["LAI"]))

rai_errors = (~df.apply(check_rai, axis=1)).sum()

print("\nRAI check")
print("Number of errors:", rai_errors)

# Final summary
print("\nSummary")

if mismatch == 0 and lai_errors == 0 and rai_errors == 0:
    print("All structural dependencies are correct")
else:
    print("Some dependency issues found")

print("Network and Device dependencies can be seen in the tables above")