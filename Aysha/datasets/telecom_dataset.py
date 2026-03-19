import pandas as pd
import random
from faker import Faker

fake = Faker()

# randomness so dataset stays the same every run
random.seed(42)
Faker.seed(42)
rows = 3000
mcc_data = {
    "240": { # Sweden (CET)
        "mobile_prefixes": ["4670", "4672", "4673", "4676", "4679"],
        "mncs": ["01", "02", "04", "05", "07", "08", "16", "24"],
        "tz_offsets": ["+0100", "+0200"] # Standard and DST
    },
    "206": { # Belgium (CET)
        "mobile_prefixes": ["3245", "3246", "3247", "3248", "3249"],
        "mncs": ["01", "05", "10", "20"],
        "tz_offsets": ["+0100", "+0200"]
    },
    "234": { # UK (GMT/BST)
        "mobile_prefixes": ["4473", "4474", "4475", "4477", "4478"],
        "mncs": ["10", "15", "20", "30", "33", "38"],
        "tz_offsets": ["+0000", "+0100"]
    },
    "310": { # USA (EST, CST, MST, PST)
        "mobile_prefixes": ["1202", "1305", "1415", "1512", "1617"], 
        "mncs": ["120", "260", "410", "480"],
        "tz_offsets": ["-0500", "-0600", "-0700", "-0800"] 
    }
}
data = []

network_types = ["3G","4G","5G"]
device_types = ["Smartphone","Tablet", "IoT Device"]

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

for i in range(rows):

    MCC = random.choice(list(mcc_data.keys()))
    
    tz_offset = random.choice(mcc_data[MCC]["tz_offsets"])

    dt = fake.date_time_between(start_date='-30d', end_date='now')
    timestamp = dt.strftime('%y%m%d%H%M%S') + tz_offset

    MNC = random.choice(mcc_data[MCC]["mncs"])
    PLMN = MCC + MNC

    lac=random.randint(1000,9999)
    rac=random.randint(10,99)

    lai=f"{MCC}-{MNC}-{lac}"
    rai=f"{lai}-{rac}"

    network=random.choice(network_types)
    device = random.choice(device_types)

    msin=unique_number(10,imsi_set)
    imsi=MCC+msin

    # subscriber=unique_number(8,msisdn_set)
    # msisdn="4670"+subscriber
    prefix = random.choice(mcc_data[MCC]["mobile_prefixes"])
    subscriber = unique_number(8, msisdn_set)
    msisdn = prefix + subscriber

    tac=str(random.randint(10000000,99999999))
    serial=unique_number(6,imei_set)
    imei=tac+serial+str(random.randint(0,9))

    ip=unique_ip()

    tmsi=random.randint(10000000,99999999)
    lmsi=random.randint(10000000,99999999)
    tlli=random.randint(10000000,99999999)

    data.append([
        imsi,msisdn,imei,ip,
        timestamp,
        MCC,MNC,PLMN,
        tmsi,lmsi,tlli,
        lai,rai,
        network,
        device
    ])


# DUPLICATE RECORDS

# for i in range(duplicate_pairs):

#     MNC = str(random.randint(1,99)).zfill(2)
    # PLMN = MCC + MNC

#     lac=random.randint(1000,9999)
#     rac=random.randint(10,99)

#     lai=f"{MCC}-{MNC}-{lac}"
#     rai=f"{lai}-{rac}"

#     network=random.choice(network_types)

#     for j in range(2):

#         msin=unique_number(10,imsi_set)
#         imsi=MCC+msin

#         subscriber=unique_number(8,msisdn_set)
#         msisdn="4670"+subscriber

#         tac=str(random.randint(10000000,99999999))
#         serial=unique_number(6,imei_set)
#         imei=tac+serial+str(random.randint(0,9))

#         ip=unique_ip()

#         tmsi=random.randint(10000000,99999999)
#         lmsi=random.randint(10000000,99999999)
#         tlli=random.randint(10000000,99999999)

#         data.append([
#             imsi,msisdn,imei,ip,
#             MCC,MNC,PLMN,
#             tmsi,lmsi,tlli,
#             lai,rai,
#             network
#         ])


columns=[
"IMSI","MSISDN","IMEI","IP_Address","Timestamp",
"MCC","MNC","PLMN",
"TMSI","LMSI","TLLI",
"LAI","RAI","Network_Type", "Device_Type"
]

df=pd.DataFrame(data,columns=columns)

df.to_csv("telecom_dataset.csv",index=False)

print("Dataset generated successfully")
print(df.head())
