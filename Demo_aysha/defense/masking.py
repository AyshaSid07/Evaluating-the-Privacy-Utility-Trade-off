<<<<<<< HEAD
""" import pandas as pd

# Load generalized dataset
df = pd.read_csv("raw_dataset.csv")
=======
import pandas as pd
df = pd.read_csv("../datasets/telecom_dataset.csv")
df_masking = df.copy()

def mask_postal_code(postal_code):
    if pd.isna(postal_code):
        return postal_code
    postal_code = str(postal_code)
    if len(postal_code) <= 2:
        return postal_code + "*" * (5 - len(postal_code))
    return postal_code[:2] + "*" * (len(postal_code) - 2)
>>>>>>> 0dfa3730578df2c5cc8e47891852780d804c89a9

def mask_ip_address(ip):
    if pd.isna(ip):
        return ip
    ip = str(ip)
    parts = ip.split('.')
    if len(parts) == 4:
        # keep the first part (the network) 
        # hide the last part (the specific device)
        return f"{parts[0]}.***.***.***"
    return ip

def mask_LAI_RAI(value):
    if pd.isna(value):
        return value
    parts = str(value).split('-')
    if len(parts) == 3:
        # keep the first part (the country code) 
        # hide the last two parts (the specific location and cell)
        return f"{parts[0]}-{parts[1]}-{parts[2][:2]}**"
    elif len(parts) == 4:
        # keep the first part (the country code) 
        # hide the last three parts (the specific location, cell, and subcell)
        return f"{parts[0]}-{parts[1]}-{parts[2][:2]}**-**"

<<<<<<< HEAD
print("Masking applied successfully") """

# masking.py
import pandas as pd

# Load raw data
df = pd.read_csv("raw_dataset.csv")

# MASKING: remove direct identifiers, mask temporary ones
df_masked = df.copy()

# Remove direct identifiers
for col in ["IMSI", "MSISDN", "IMEI", "IP_Address"]:
    if col in df_masked.columns:
        df_masked = df_masked.drop(columns=[col])

# Mask temporary identifiers (keep partial structure)
for col in ["TMSI", "LMSI", "TLLI"]:
    if col in df_masked.columns:
        df_masked[col] = df_masked[col].astype(str).str[:4] + "****"

# Save
df_masked.to_csv("masked_dataset.csv", index=False)
print("masked_dataset.csv written.")


=======
# if "Postal Code" and "IP Address" in df_masking.columns:
    # df_masking["Postal Code"] = df_masking["Postal Code"].apply(mask_postal_code)
    # df_masking["IP Address"] = df_masking['IP Address'].apply(mask_ip_address)
df_masking["LAI"] = df_masking["LAI"].apply(mask_LAI_RAI)
df_masking["RAI"] = df_masking["RAI"].apply(mask_LAI_RAI)

df_masking.to_csv("../defense/masked_dataset.csv", index=False)
>>>>>>> 0dfa3730578df2c5cc8e47891852780d804c89a9
