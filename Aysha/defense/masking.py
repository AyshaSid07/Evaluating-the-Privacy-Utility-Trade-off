import pandas as pd
import re

# Load raw data
df = pd.read_csv("../datasets/telecomb_dataset.csv")
df_masking = df.copy()

def mask_string(val, visible_chars=3):
    """General masker: keeps first few chars, replaces rest with X."""
    if pd.isna(val): return val
    val = str(val)
    if len(val) <= visible_chars:
        return "X" * len(val)
    return val[:visible_chars] + "X" * (len(val) - visible_chars)

def mask_ip_address(ip):
    if pd.isna(ip): return ip
    parts = str(ip).split('.')
    if len(parts) == 4:
        # Mask the last two octets to generalize the location
        return f"{parts[0]}.{parts[1]}.XXX.XXX"
    return ip

def mask_LAI_RAI(value):
    if pd.isna(value): return value
    parts = str(value).split('-')
    # Aggressive masking: Hide the specific LAC and RAC entirely
    # Only keep the MCC (Country) and MNC (Network)
    if len(parts) >= 3:
        return f"{parts[0]}-{parts[1]}-XXXX"
    return "XXXX"

# 1. Mask Direct Identifiers (Standard Privacy Practice)
df_masking["IMSI"] = df_masking["IMSI"].apply(lambda x: mask_string(x, 5)) # Keep MCC/MNC
df_masking["MSISDN"] = df_masking["MSISDN"].apply(lambda x: mask_string(x, 4)) # Keep Country code
df_masking["IMEI"] = df_masking["IMEI"].apply(lambda x: mask_string(x, 8)) # Keep TAC (Device type)
df_masking["IP_Address"] = df_masking["IP_Address"].apply(mask_ip_address)

# 2. Mask Location Identifiers (This will cause the Utility Loss in Clustering)
df_masking["LAI"] = df_masking["LAI"].apply(mask_LAI_RAI)
df_masking["RAI"] = df_masking["RAI"].apply(mask_LAI_RAI)

# 3. Mask Session Identifiers (TMSI/LMSI/TLLI)
# These are often masked by rounding or partial replacement
for col in ["TMSI", "LMSI", "TLLI"]:
    if col in df_masking.columns:
        # Keep only the first 2 digits, zero out the rest
        df_masking[col] = df_masking[col].apply(lambda x: int(str(x)[:2] + "000000") if pd.notna(x) else x)

# Save to defense folder
df_masking.to_csv("../defense/masked_dataset.csv", index=False)
print("Masked dataset generated with balanced privacy/utility loss.")