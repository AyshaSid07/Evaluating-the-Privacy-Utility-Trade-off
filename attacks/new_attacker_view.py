import pandas as pd

# Load external dataset (contains real IPs)
external_df = pd.read_csv("../data/mendeley_data.csv")

# Load anonymized dataset (choose one version)
anonymized_df = pd.read_csv("../anonymize/anonymized_v1.csv")

# Quasi-identifiers used for matching
QI_COLS = [
    "Region",
    "Country",
    "ISP",
    "Organization",
    "Autonomous System"
]

# Precompute how many times each QI combination appears in raw data
group_sizes = external_df.groupby(QI_COLS).size()

print("\n=== Attacker View: Successful Re-identifications ===")

examples_shown = 0
max_examples = 5

for _, row in anonymized_df.iterrows():

    key = tuple(row[col] for col in QI_COLS)

    if key in group_sizes and group_sizes[key] == 1:

        # Find the exact matching external row
        match = external_df[
            (external_df[QI_COLS] == pd.Series(key, index=QI_COLS)).all(axis=1)
        ]

        recovered_ip = match["IP Address"].values[0]

        print("\n--- Successful Linkage ---")
        print("Anonymized QI values:")
        for col in QI_COLS:
            print(f"{col}: {row[col]}")

        print("Recovered Real IP Address:", recovered_ip)

        examples_shown += 1

        if examples_shown >= max_examples:
            break

print("\nFinished showing examples.")
