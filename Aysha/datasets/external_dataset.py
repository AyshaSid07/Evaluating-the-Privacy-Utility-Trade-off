import pandas as pd
import random

random.seed(42)

rows = 3000

device_types = ["Smartphone","Tablet","IoT Device"]

# Load the raw telecom dataset
raw_df = pd.read_csv("telecom_dataset.csv")

data = []

for i in range(rows):

    device_id = "D" + str(1000+i)
    device_type = random.choice(device_types)

    # Take a row from the raw dataset
    record = raw_df.iloc[i]

    lai = record["LAI"]
    rai = record["RAI"]
    network = record["Network_Type"]

    data.append([
        device_id,
        device_type,
        lai,
        rai,
        network
    ])

df = pd.DataFrame(data,columns=[
"Device_ID",
"Device_Type",
"LAI",
"RAI",
"Network_Type"
])

df.to_csv("external_dataset.csv",index=False)

print("External dataset generated successfully")
print(df.head())