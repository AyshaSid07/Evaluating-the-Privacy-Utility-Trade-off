import pandas as pd
import random
from faker import Faker

random.seed(42)
fake = Faker()

rows = 1000

# Load the raw telecom dataset
raw_df = pd.read_csv("../mendeley_dataset.csv")

data = []

for i in range(rows):


    # Take a row from the raw dataset
    record = raw_df.iloc[i]
    name = fake.name()
    age = random.randint(18, 80)
    city = record["City"]
    region = record["Region"]
    country = record["Country"]
    postal_code = record["Postal Code"]
    linkage_index = record["Linkage_Index"]
    data.append([
        name,
        age,
        city,
        region,
        country,
        postal_code,
        linkage_index
    ])

df = pd.DataFrame(data, columns=[
    "Name",
    "Age",
    "City",
    "Region",
    "Country",
    "Postal Code",
    "Linkage_Index"

])

df.to_csv("../external_dataset_mendeley.csv",index=False)