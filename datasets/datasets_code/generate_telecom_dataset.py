import pandas as pd
import random
from faker import Faker

fake = Faker()

random.seed(42)
Faker.seed(42)
rows = 3000

mcc_data = {
    "240": {  # Sweden (CET)
        "mobile_prefixes": ["4670", "4672", "4673", "4676", "4679"],
        "mncs": ["01", "02", "04", "05", "07", "08", "16", "24"],
        "tz_offsets": ["+0100", "+0200"],
        "network_weights": [0.10, 0.30, 0.60], 
        "device_weights": [0.70, 0.20, 0.10],     
        "tmsi_range": (10000000, 40000000),
        "lmsi_range": (10000000, 35000000),
        "tlli_range": (10000000, 30000000),
    },
    "206": {  # Belgium (CET)
        "mobile_prefixes": ["3245", "3246", "3247", "3248", "3249"],
        "mncs": ["01", "05", "10", "20"],
        "tz_offsets": ["+0100", "+0200"],
        "network_weights": [0.15, 0.35, 0.50], 
        "device_weights": [0.60, 0.25, 0.15],
        "tmsi_range": (30000000, 60000000),
        "lmsi_range": (30000000, 55000000),
        "tlli_range": (25000000, 55000000),
    },
    "234": {  # UK (GMT/BST)
        "mobile_prefixes": ["4473", "4474", "4475", "4477", "4478"],
        "mncs": ["10", "15", "20", "30", "33", "38"],
        "tz_offsets": ["+0000", "+0100"],
        "network_weights": [0.50, 0.35, 0.15],  
        "device_weights": [0.40, 0.35, 0.25],
        "tmsi_range": (50000000, 80000000),
        "lmsi_range": (50000000, 75000000),
        "tlli_range": (50000000, 78000000),
    },
    "310": {  # USA (multiple TZ)
        "mobile_prefixes": ["1202", "1305", "1415", "1512", "1617"],
        "mncs": ["120", "260", "410", "480"],
        "tz_offsets": ["-0500", "-0600", "-0700", "-0800"],
        "network_weights": [0.20, 0.55, 0.25],
        "device_weights": [0.50, 0.15, 0.35], 
        "tmsi_range": (70000000, 99999999),
        "lmsi_range": (70000000, 95000000),
        "tlli_range": (65000000, 99999999),
    }
}

data = []
network_types = ["3G", "4G", "5G"]
device_types = ["Smartphone", "Tablet", "IoT Device"]

imsi_set = set()
msisdn_set = set()
imei_set = set()
ip_set = set()


def unique_number(length, used):
    while True:
        num = ''.join(str(random.randint(0, 9)) for _ in range(length))
        if num not in used:
            used.add(num)
            return num


def unique_ip():
    while True:
        ip = fake.ipv4()
        if ip not in ip_set:
            ip_set.add(ip)
            return ip


for i in range(rows):
    MCC = random.choice(list(mcc_data.keys()))
    country = mcc_data[MCC]

    tz_offset = random.choice(country["tz_offsets"])
    dt = fake.date_time_between(start_date='-30d', end_date='now')
    timestamp = dt.strftime('%y%m%d%H%M%S') + tz_offset

    MNC = random.choice(country["mncs"])
    PLMN = MCC + MNC

    lac = random.randint(1000, 9999)
    rac = random.randint(10, 99)
    lai = f"{MCC}-{MNC}-{lac}"
    rai = f"{lai}-{rac}"

    network = random.choices(network_types, weights=country["network_weights"])[0]
    device = random.choices(device_types, weights=country["device_weights"])[0]

    msin = unique_number(10, imsi_set)
    imsi = MCC + MNC + msin

    prefix = random.choice(country["mobile_prefixes"])
    subscriber = unique_number(8, msisdn_set)
    msisdn = prefix + subscriber

    tac = str(random.randint(10000000, 99999999))
    serial = unique_number(6, imei_set)
    imei = tac + serial + str(random.randint(0, 9))

    ip = unique_ip()

    tmsi = random.randint(*country["tmsi_range"])
    lmsi = random.randint(*country["lmsi_range"])
    tlli = random.randint(*country["tlli_range"])

    data.append([
        imsi, msisdn, imei, ip,
        timestamp,
        MCC, MNC, PLMN,
        tmsi, lmsi, tlli,
        lai, rai,
        network,
        device
    ])

columns = [
    "IMSI", "MSISDN", "IMEI", "IP_Address", "Timestamp",
    "MCC", "MNC", "PLMN",
    "TMSI", "LMSI", "TLLI",
    "LAI", "RAI", "Network_Type", "Device_Type"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("../telecom_dataset.csv", index=False)
print("Dataset generated successfully")
