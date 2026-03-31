import pandas as pd

# Load raw data
df = pd.read_csv("../csvfiles/telecom_dataset.csv")

# 1. Drop Direct Identifiers and high-cardinality session IDs
to_drop = ['IMSI', 'MSISDN', 'IMEI', 'IP_Address', 'TMSI', 'LMSI', 'TLLI']
df_suppressed = df.drop(columns=[c for c in to_drop if c in df.columns], errors='ignore')

# 2. Data Sanitization (Strip spaces to prevent "invisible" unique rows)
df_suppressed = df_suppressed.apply(lambda x: x.astype(str).str.strip() if x.dtype == "object" else x)

df_suppressed.to_csv("suppression_only.csv", index=False)
print("Suppression Complete.")