import pandas as pd
df = pd.read_csv("../datasets/mendeley_data.csv")
df_masking = df.copy()

def mask_postal_code(postal_code):
    if pd.isna(postal_code):
        return postal_code
    postal_code = str(postal_code)
    if len(postal_code) <= 2:
        return postal_code + "*" * (5 - len(postal_code))
    return postal_code[:2] + "*" * (len(postal_code) - 2)

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


if "Postal Code" and "IP Address" in df_masking.columns:
    df_masking["Postal Code"] = df_masking["Postal Code"].apply(mask_postal_code)
    df_masking["IP Address"] = df_masking['IP Address'].apply(mask_ip_address)

df_masking.to_csv("../datasets/de-identified-datasets/masking.csv", index=False)