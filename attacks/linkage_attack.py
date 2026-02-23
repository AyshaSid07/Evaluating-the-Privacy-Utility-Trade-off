import pandas as pd

# Quasi-identifiers used in the linkage attack
QI_COLS = [
    "Region",
    "Country",
    "ISP",
    "Organization",
    "Autonomous System",
    "Postal Code",
    "Latitude",
    "Longitude"
]

files = {
    "v1 (3 decimals)": "../anonymize/anonymized_v1.csv",
    "v2 (2 decimals)": "../anonymize/anonymized_v2.csv",
    "v3 (1 decimal)": "../anonymize/anonymized_v3.csv"
}

for label, filename in files.items():
    print(f"\nLinkage Attack Results for {label}")

    df = pd.read_csv(filename)

    # Step 1: Select quasi-identifiers (Sweeney 2002)
    qi_df = df[QI_COLS].dropna()

    # Step 2: Group by QI combinations (Samarati 2001)
    group_sizes = qi_df.groupby(QI_COLS).size()

    total_records = len(qi_df)

    # Step 3: Identify unique QI combinations (k=1) where k is the group size
    unique_classes = (group_sizes == 1).sum()
    records_in_unique = group_sizes[group_sizes == 1].sum()

    # Step 4: Identify small groups (k <= 5)
    classes_le_5 = (group_sizes <= 5).sum()
    records_in_le_5 = group_sizes[group_sizes <= 5].sum()

    # Step 5: Identify strictly high-risk (2 <= k <= 5)
    classes_2_5 = ((group_sizes > 1) & (group_sizes <= 5)).sum()
    records_2_5 = group_sizes[(group_sizes > 1) & (group_sizes <= 5)].sum()

    print(f"Total records: {total_records}")
    print(f"Unique QI combinations (k=1): {unique_classes}")
    print(f"Records in unique groups: {records_in_unique} ({records_in_unique / total_records:.2%})")
    print(f"QI groups with k<=5: {classes_le_5}")
    print(f"Records in k<=5 groups: {records_in_le_5} ({records_in_le_5 / total_records:.2%})")
    print(f"QI groups with 2<=k<=5: {classes_2_5}")
    print(f"Records in 2<=k<=5 groups: {records_2_5} "
      f"({records_2_5 / total_records:.2%})")


