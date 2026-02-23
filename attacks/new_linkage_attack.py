import pandas as pd

# Load raw dataset
external_df = pd.read_csv("../data/mendeley_data.csv")

files = {
    "v1 (3 decimals)": "../anonymize/anonymized_v1.csv",
    "v2 (2 decimals)": "../anonymize/anonymized_v2.csv",
    "v3 (1 decimal)": "../anonymize/anonymized_v3.csv"
}

QI_COLS = [
    "Region",
    "Country",
    "ISP",
    "Organization",
    "Autonomous System"
]

# Compute group sizes in raw dataset
group_sizes = external_df.groupby(QI_COLS).size()

for label, path in files.items():

    print("\nLinkage Attack Results for:", label)

    anonymized_df = pd.read_csv(path)

    successful = 0
    ambiguous = 0
    no_match = 0

    for _, row in anonymized_df.iterrows():

        key = tuple(row[col] for col in QI_COLS)

        if key in group_sizes:
            count = group_sizes[key]

            if count == 1:
                successful += 1
            else:
                ambiguous += 1
        else:
            no_match += 1

    total = len(anonymized_df)

    print("Total records:", total)
    print("Successful re-identifications:", successful,
          f"({successful/total:.2%})")
    print("Ambiguous matches:", ambiguous,
          f"({ambiguous/total:.2%})")
    print("No matches:", no_match,
          f"({no_match/total:.2%})")
