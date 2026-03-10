import pandas as pd

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
    "v1": "../anonymize/anonymized_v1.csv",
    "v2": "../anonymize/anonymized_v2.csv",
    "v3": "../anonymize/anonymized_v3.csv"
}

for version, filename in files.items():
    df = pd.read_csv(filename)

    # Group by QIs and count occurrences
    attacker_view = df.groupby(QI_COLS).size().reset_index(name="count")

    # Save the file the attacker would use
    outname = f"attacker_view_{version}_equivalence_classes.csv"
    attacker_view.to_csv(outname, index=False)

    print(f"Created: {outname}")

