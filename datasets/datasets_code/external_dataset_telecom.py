import pandas as pd
import random

random.seed(42)

rows = 1000

# Load the raw telecom dataset
raw_df = pd.read_csv("../telecom_dataset.csv")

data = []

for i in range(rows):

    device_id = "D" + str(1000+i)

    # Take a row from the raw dataset
    record = raw_df.iloc[i]

    plmn = record["PLMN"]
    network_type = record["Network_Type"]
    device_type = record["Device_Type"]
    lai = record["LAI"]

    if network_type == "5G":
        avg_latency = random.randint(10, 30) # Extremely fast
        drop_rate = round(random.uniform(0.01, 0.5), 2) # 0.01% to 0.5%
    elif network_type == "4G":
        avg_latency = random.randint(30, 70) # Standard broadband
        drop_rate = round(random.uniform(0.1, 1.5), 2)  # 0.1% to 1.5%
    else: # 3G
        avg_latency = random.randint(70, 150) # Slower, legacy routing
        drop_rate = round(random.uniform(0.5, 3.5), 2)  # 0.5% to 3.5%

    data.append([
        device_id,
        device_type,
        network_type,
        plmn,
        lai,
        avg_latency,
        drop_rate
    ])

df = pd.DataFrame(data, columns=[
    "Device_ID",
    "Device_Type",
    "Network_Type",
    "PLMN",
    "LAI",
    "Avg_Latency_ms",
    "Connection_Drop_Rate_Pct"
])

df.to_csv("../external_dataset_telecom.csv",index=False)

print("External dataset generated successfully")
print(df.head())