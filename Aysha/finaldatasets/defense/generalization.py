"""
Defense 1: Generalization
Replace specific values with ranges to reduce precision.

 MCC, MNC, PLMN, LAI, RAI are all interdependent.
- PLMN = MCC + MNC
- LAI  = MCC + MNC + LAC
- RAI  = LAI + RAC
So we must generalize MNC consistently across ALL of them.

What we generalize:
- MNC:       07          → 00-09           (range of 10)
- PLMN:      24007       → 240-00-09       (reflects MCC + generalized MNC)
- LAI:       240-07-5012 → 240-00-09-5000-5999  (MNC range + LAC range)
- RAI:       240-07-5012-38 → 240-00-09-5000-5999 (same as LAI, RAC dropped)
"""

import pandas as pd

df = pd.read_csv("../defense/directidentifiersrmv_telecom.csv", dtype=str)

print("BEFORE generalization:")
print(f"  Sample MNC:  {df['MNC'].head(3).tolist()}")
print(f"  Sample PLMN: {df['PLMN'].head(3).tolist()}")
print(f"  Sample LAI:  {df['LAI'].head(3).tolist()}")
print(f"  Sample RAI:  {df['RAI'].head(3).tolist()}")
print()


def generalize_mnc(val):
    """
    '07' → '00-09'
    '26' → '20-29'
    Groups MNC into ranges of 10.
    """
    if pd.isna(val):
        return val
    mnc_num = int(val)
    lower = (mnc_num // 10) * 10
    upper = lower + 9
    return f"{lower:02d}-{upper:02d}"


def generalize_lai(val):
    """
    '240-07-5012' → '240-00-09-5000-5999'
    MNC part → range of 10, LAC part → range of 1000.
    """
    if pd.isna(val):
        return val
    parts = str(val).split("-")
    if len(parts) >= 3:
        mnc_num = int(parts[1])
        mnc_lower = (mnc_num // 10) * 10
        mnc_upper = mnc_lower + 9

        lac_num = int(parts[2])
        lac_lower = (lac_num // 1000) * 1000
        lac_upper = lac_lower + 999

        return f"{parts[0]}-{mnc_lower:02d}-{mnc_upper:02d}-{lac_lower}-{lac_upper}"
    return val


def generalize_rai(val):
    """
    '240-07-5012-38' → '240-00-09-5000-5999'
    Same as LAI. RAC is dropped since we already group the LAC.
    """
    if pd.isna(val):
        return val
    parts = str(val).split("-")
    if len(parts) >= 3:
        mnc_num = int(parts[1])
        mnc_lower = (mnc_num // 10) * 10
        mnc_upper = mnc_lower + 9

        lac_num = int(parts[2])
        lac_lower = (lac_num // 1000) * 1000
        lac_upper = lac_lower + 999

        return f"{parts[0]}-{mnc_lower:02d}-{mnc_upper:02d}-{lac_lower}-{lac_upper}"
    return val


# Apply generalizations
df["MNC"] = df["MNC"].apply(generalize_mnc)
df["LAI"] = df["LAI"].apply(generalize_lai)
df["RAI"] = df["RAI"].apply(generalize_rai)

# Rebuild PLMN from MCC + generalized MNC (keeps dependency consistent)
df["PLMN"] = df["MCC"] + "-" + df["MNC"]

print("AFTER generalization:")
print(f"  Sample MNC:  {df['MNC'].head(3).tolist()}")
print(f"  Sample PLMN: {df['PLMN'].head(3).tolist()}")
print(f"  Sample LAI:  {df['LAI'].head(3).tolist()}")
print(f"  Sample RAI:  {df['RAI'].head(3).tolist()}")

df.to_csv("../defense/generalized_telecom.csv", index=False)
print("\nSaved to ../defense/generalized_telecom.csv")