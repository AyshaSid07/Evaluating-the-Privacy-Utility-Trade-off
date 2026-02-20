import pandas as pd

# Load dataset
df = pd.read_csv("mendeley_data.csv")

print("Total records:", len(df))


# 1. Direct Identifier Check


if "IP Address" in df.columns:
    unique_ips = df["IP Address"].nunique()

    print("\nDirect Identifier uniqueness:")
    print("Column used: IP Address")
    print("Unique IPs:", unique_ips)
    print("Percentage:", round((unique_ips / len(df)) * 100, 2), "%")


# 2. Quasi-Identifier Check


qi_columns = ["City", "Postal Code", "ISP", "Organization"]

# Keep only columns that exist
qi_columns = [col for col in qi_columns if col in df.columns]

if qi_columns:
    group_sizes = df.groupby(qi_columns).size()
    unique_groups = (group_sizes == 1).sum()

    print("\nQI Combination uniqueness:")
    print("QI Columns used:", qi_columns)
    print("Groups with size 1:", unique_groups)
    print("Percentage:", (unique_groups / len(group_sizes)) * 100)

# 3. Geolocation Check

geo_columns = ["Latitude", "Longitude"]

if all(col in df.columns for col in geo_columns):
    geo_groups = df.groupby(geo_columns).size()
    unique_geo = (geo_groups == 1).sum()

    print("\nGeolocation uniqueness:")
    print("Columns used:", geo_columns)
    print("Unique coordinate pairs:", unique_geo)
    print("Percentage:", round((unique_geo / len(geo_groups)) * 100, 2), "%")

