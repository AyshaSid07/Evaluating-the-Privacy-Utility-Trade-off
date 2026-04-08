"""
Defense 0: Remove Direct Identifiers Only
Drops IMSI, MSISDN, IMEI, IP_Address.

"""

import pandas as pd

df = pd.read_csv("../csvfiles/telecom_dataset.csv", dtype=str)

print("BEFORE removing direct identifiers:")
print(f"  Columns: {list(df.columns)}")
print(f"  Rows: {len(df)}")
print()

# direct identifiers
direct_ids = ["IMSI", "MSISDN", "IMEI", "IP_Address"]

df = df.drop(columns=[c for c in direct_ids if c in df.columns])

print("AFTER removing direct identifiers:")
print(f"  Columns: {list(df.columns)}")
print(f"  Rows: {len(df)}")

df.to_csv("../defense/directidentifiersrmv_telecom.csv", index=False)
print("\nSaved to ../defense/directidentifiersmv_telecom.csv")