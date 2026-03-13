import pandas as pd
import random
from faker import Faker

fake = Faker()

rows = 10000
unique_rows = 4000
duplicate_pairs = 3000

data = []

MCC = "240"
MNC = "01"
PLMN = MCC + MNC

network_types = ["3G","4G","5G"]

imsi_set=set()
msisdn_set=set()
imei_set=set()
ip_set=set()

def unique_number(length, used):
    while True:
        num=''.join(str(random.randint(0,9)) for _ in range(length))
        if num not in used:
            used.add(num)
            return num

def unique_ip():
    while True:
        ip=fake.ipv4()
        if ip not in ip_set:
            ip_set.add(ip)
            return ip


# UNIQUE RECORDS

for i in range(unique_rows):

    lac=random.randint(1000,9999)
    rac=random.randint(10,99)

    lai=f"{MCC}-{MNC}-{lac}"
    rai=f"{lai}-{rac}"
    network=random.choice(network_types)

    msin=unique_number(10,imsi_set)
    imsi=MCC+MNC+msin

    subscriber=unique_number(8,msisdn_set)
    msisdn="4670"+subscriber

    tac=str(random.randint(10000000,99999999))
    serial=unique_number(6,imei_set)
    imei=tac+serial+str(random.randint(0,9))

    ip=unique_ip()

    tmsi=random.randint(10000000,99999999)
    lmsi=random.randint(10000000,99999999)
    tlli=random.randint(10000000,99999999)

    data.append([
        imsi,msisdn,imei,ip,
        MCC,MNC,PLMN,
        tmsi,lmsi,tlli,
        lai,rai,
        network
    ])


# DUPLICATE RECORDS 

for i in range(duplicate_pairs):

    lac=random.randint(1000,9999)
    rac=random.randint(10,99)

    lai=f"{MCC}-{MNC}-{lac}"
    rai=f"{lai}-{rac}"
    network=random.choice(network_types)

    for j in range(2):   # create pair

        msin=unique_number(10,imsi_set)
        imsi=MCC+MNC+msin

        subscriber=unique_number(8,msisdn_set)
        msisdn="4670"+subscriber

        tac=str(random.randint(10000000,99999999))
        serial=unique_number(6,imei_set)
        imei=tac+serial+str(random.randint(0,9))

        ip=unique_ip()

        tmsi=random.randint(10000000,99999999)
        lmsi=random.randint(10000000,99999999)
        tlli=random.randint(10000000,99999999)

        data.append([
            imsi,msisdn,imei,ip,
            MCC,MNC,PLMN,
            tmsi,lmsi,tlli,
            lai,rai,
            network
        ])


columns=[
"IMSI","MSISDN","IMEI","IP_Address",
"MCC","MNC","PLMN",
"TMSI","LMSI","TLLI",
"LAI","RAI","Network_Type"
]

df=pd.DataFrame(data,columns=columns)

df.to_csv("telecom_dataset.csv",index=False)

print("Dataset generated successfully")
print(df.head())