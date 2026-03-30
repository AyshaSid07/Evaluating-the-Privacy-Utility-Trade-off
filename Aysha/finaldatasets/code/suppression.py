import pandas as pd

def suppression(df):
    # Direct Identifiers: 100% Privacy Risk
    # Removing these stops simple lookup attacks.
    direct_ids = ["IMSI", "MSISDN", "IMEI", "IP_Address", "TMSI", "LMSI", "TLLI"]
    df_protected = df.drop(columns=[col for col in direct_ids if col in df.columns], errors='ignore')
    
    print("Direct Identifiers Suppressed.")
    return df_protected

# 1. Load the raw data
raw_df = pd.read_csv('../csvfiles/telecom_dataset.csv')

# 2. Call the function and store the RETURNED DataFrame
df_layer_1 = suppression(raw_df)

# 3. Save the resulting DataFrame to a CSV
df_layer_1.to_csv('suppression.csv', index=False)