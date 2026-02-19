import pandas as pd

# Load original dataset
original_df = pd.read_csv("mendeley_data.csv")

files = {
    "v1 (3 decimals)": "anonymized_v1.csv",
    "v2 (2 decimals)": "anonymized_v2.csv",
    "v3 (1 decimal)": "anonymized_v3.csv"
}

for label, filename in files.items():
    print("\n====================================")
    print(f"Checking {label}")
    print("====================================")

    df = pd.read_csv(filename)

    # 1. SUPPRESSION CHECK
    suppressed = list(set(original_df.columns) - set(df.columns))
    print("\nSuppressed columns (before → after):")
    print("Before:", list(original_df.columns))
    print("After :", list(df.columns))
    print("Removed:", suppressed)

    # 2. MASKING CHECK (Postal Code)
    print("\nPostal Code BEFORE masking (first 5):")
    print(original_df["Postal Code"].head(5).to_string(index=False))

    print("\nPostal Code AFTER masking (first 5):")
    print(df["Postal Code"].head(5).to_string(index=False))

    # 3. GENERALIZATION CHECK (Latitude/Longitude)
    print("\nLatitude/Longitude BEFORE generalization (first 5):")
    print(original_df[["Latitude", "Longitude"]].head(5).to_string(index=False))

    print("\nLatitude/Longitude AFTER generalization (first 5):")
    print(df[["Latitude", "Longitude"]].head(5).to_string(index=False))

