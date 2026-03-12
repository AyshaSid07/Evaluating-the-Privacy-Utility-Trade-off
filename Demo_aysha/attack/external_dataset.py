import pandas as pd
from faker import Faker

fake = Faker()

#Load raw dataset

df = pd.read_csv("../defense/raw_dataset.csv")

#Take first 3000 rows (overlap with raw dataset)

df_subset = df.iloc[:3000]

external_rows = []

for i in range(len(df_subset)):

# Faker-generated attacker attributes
    device_id = "D" + str(1000 + i)
    device_type = fake.random_element(elements=("Smartphone","Tablet","IoT Device"))

# Copy quasi-identifiers from raw dataset
    lai = df_subset.iloc[i]["LAI"]
    rai = df_subset.iloc[i]["RAI"]
    network = df_subset.iloc[i]["Network_Type"]

    external_rows.append([
    device_id,
    device_type,
    lai,
    rai,
    network
])

external_df = pd.DataFrame(
external_rows,
columns=["Device_ID","Device_Type","LAI","RAI","Network_Type"]
)

external_df.to_csv("external_dataset.csv", index=False)

print("External dataset generated successfully")
print("Rows created:", len(external_df))
print(external_df.head())