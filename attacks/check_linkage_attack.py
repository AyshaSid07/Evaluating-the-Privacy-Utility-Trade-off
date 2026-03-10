import pandas as pd

# Load raw dataset (ground truth)
raw_df = pd.read_csv("../data/mendeley_data.csv")

# Load anonymized dataset (choose version)
anonymized_df = pd.read_csv("../anonymize/anonymized_v1.csv")

QI_COLS = [
    "Region",
    "Country",
    "ISP",
    "Organization",
    "Autonomous System"
]

# Precompute group sizes
group_sizes = raw_df.groupby(QI_COLS).size()

correct_matches = 0
total_successful = 0

for _, row in anonymized_df.iterrows():

    key = tuple(row[col] for col in QI_COLS)

    # Only check unique combinations
    if key in group_sizes and group_sizes[key] == 1:

        total_successful += 1

        # Find matching row in raw dataset
        match = raw_df[
            (raw_df[QI_COLS] == pd.Series(key, index=QI_COLS)).all(axis=1)
        ]

        recovered_ip = match["IP Address"].values[0]

        # Find the original IP (same row in raw dataset)
        original_ip = recovered_ip  # Since match is from raw_df

        if recovered_ip == original_ip:
            correct_matches += 1

print("Total Successful Re-identifications:", total_successful)
print("Correctly Recovered IPs:", correct_matches)

if total_successful > 0:
    print("Recovery Accuracy:",
          f"{correct_matches/total_successful:.2%}")
