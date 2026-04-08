"""
Defense 2: Masking
Only masking columns that have interdependency
- MNC, PLMN, LAI, RAI — these have MCC/MNC/LAC parts that mean something.
"""

import pandas as pd

df = pd.read_csv("../defense/directidentifiersrmv_telecom.csv", dtype=str)

print("BEFORE masking:")
print(f"  Sample MNC:  {df['MNC'].head(3).tolist()}")
print(f"  Sample PLMN: {df['PLMN'].head(3).tolist()}")
print(f"  Sample LAI:  {df['LAI'].head(3).tolist()}")
print(f"  Sample RAI:  {df['RAI'].head(3).tolist()}")
print()


def mask_mnc(val):
    """
    '07' → '**'
    """
    if pd.isna(val):
        return val
    return "**"


def mask_plmn(val):
    """
    '24007' → '240-**'
    """
    if pd.isna(val):
        return val
    val = str(val)
    if len(val) >= 5:
        return f"{val[:3]}-**"
    return val


def mask_lai(val):
    """
    '240-07-5012' → '240-**-****'
    - MCC (240):  kept (country level)
    - MNC (07):   fully masked (consistent with mask_mnc)
    - LAC (5012): fully masked (hides location area)
    """
    if pd.isna(val):
        return val
    parts = str(val).split("-")
    if len(parts) >= 3:
        mcc = parts[0]
        mnc_masked = "*" * len(parts[1])
        lac_masked = "*" * len(parts[2])
        return f"{mcc}-{mnc_masked}-{lac_masked}"
    return val


def mask_rai(val):
    """
    '240-07-5012-38' → '240-**-****-**'
    Same as LAI + RAC fully masked.
    RAI depends on LAI, so LAI part masked the same way.
    """
    if pd.isna(val):
        return val
    parts = str(val).split("-")
    if len(parts) >= 4:
        mcc = parts[0]
        mnc_masked = "*" * len(parts[1])
        lac_masked = "*" * len(parts[2])
        rac_masked = "*" * len(parts[3])
        return f"{mcc}-{mnc_masked}-{lac_masked}-{rac_masked}"
    elif len(parts) >= 3:
        mcc = parts[0]
        mnc_masked = "*" * len(parts[1])
        lac_masked = "*" * len(parts[2])
        return f"{mcc}-{mnc_masked}-{lac_masked}"
    return val


# masking structured fields
df["MNC"] = df["MNC"].apply(mask_mnc)
df["PLMN"] = df["PLMN"].apply(mask_plmn)
df["LAI"] = df["LAI"].apply(mask_lai)
df["RAI"] = df["RAI"].apply(mask_rai)


print("AFTER masking:")
print(f"  Sample MNC:  {df['MNC'].head(3).tolist()}")
print(f"  Sample PLMN: {df['PLMN'].head(3).tolist()}")
print(f"  Sample LAI:  {df['LAI'].head(3).tolist()}")
print(f"  Sample RAI:  {df['RAI'].head(3).tolist()}")

df.to_csv("../defense/masked_telecom.csv", index=False)
print("\nSaved to ../defense/masked_telecom.csv")