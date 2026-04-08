"""
Suppress Direct Identifiers
Drop direct identifiers (IMSI, MSISDN, IMEI, IP_Address) from the telecom dataset.
"""
import pandas as pd

df = pd.read_csv("../datasets/telecom_dataset.csv", dtype={'MNC': str})
df = df.drop(columns=["IMSI", "MSISDN", "IMEI", "IP_Address"])
df.to_csv("../datasets/de_identified_datasets/suppressed_DI_telecom.csv", index=False)
