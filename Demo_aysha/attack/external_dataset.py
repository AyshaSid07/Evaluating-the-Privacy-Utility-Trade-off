import pandas as pd
import random

rows = 3000

device_types = ["Smartphone","Tablet","IoT Device"]
network_types = ["3G","4G","5G"]

data = []

for i in range(rows):

    device_id = "D" + str(1000+i)
    device_type = random.choice(device_types)

    lai = "240-01"
    rai = "240-01"

    network = random.choice(network_types)

    data.append([device_id,device_type,lai,rai,network])

df = pd.DataFrame(data,columns=[
"Device_ID","Device_Type","LAI","RAI","Network_Type"
])

df.to_csv("external_dataset.csv",index=False)