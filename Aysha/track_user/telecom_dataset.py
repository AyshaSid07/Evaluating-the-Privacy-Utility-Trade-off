import pandas as pd
import random
import numpy as np
from faker import Faker
from datetime import timedelta

fake = Faker()

random.seed(42)
np.random.seed(42)
Faker.seed(42)

# -----------------------------
# CONFIGURATION 
# -----------------------------

NUM_USERS = 5000
AVG_SESSIONS_PER_USER = 20
ANOMALY_RATE = 0.03

MCC = "240"
NETWORK_TYPES = ["3G","4G","5G"]

data = []

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def generate_number(length):
    return ''.join(str(random.randint(0,9)) for _ in range(length))


def generate_imei():
    tac = str(random.randint(10000000,99999999))
    serial = generate_number(6)
    check = str(random.randint(0,9))
    return tac + serial + check


def generate_ip_from_subnet(subnet):
    return f"{subnet}.{random.randint(1,254)}"


# -----------------------------
# CREATE USERS
# -----------------------------

users = []

for _ in range(NUM_USERS):

    imsi = MCC + generate_number(10)
    msisdn = "4670" + generate_number(8)

    imei = generate_imei()

    # user-specific behaviour profile
    preferred_network = random.choices(
        NETWORK_TYPES,
        weights=[0.2,0.5,0.3]
    )[0]

    subnet = f"{random.randint(10,200)}.{random.randint(0,255)}.{random.randint(0,255)}"

    users.append({
        "IMSI": imsi,
        "MSISDN": msisdn,
        "IMEI": imei,
        "PREFERRED_NETWORK": preferred_network,
        "SUBNET": subnet
    })


# -----------------------------
# GENERATE USER SESSIONS
# -----------------------------

for user in users:

    sessions = np.random.poisson(AVG_SESSIONS_PER_USER)

    start_time = fake.date_time_this_year()

    base_mnc = str(random.randint(1,99)).zfill(2)
    plmn = MCC + base_mnc

    lac = random.randint(1000,9999)
    rac = random.randint(10,99)

    lai = f"{MCC}-{base_mnc}-{lac}"
    rai = f"{lai}-{rac}"

    device = user["IMEI"]
    subnet = user["SUBNET"]

    timestamp = start_time

    for _ in range(max(1,sessions)):

        # random gap between sessions
        timestamp += timedelta(minutes=random.randint(5,60))

        network = random.choices(
            NETWORK_TYPES,
            weights=[
                0.7 if user["PREFERRED_NETWORK"]=="3G" else 0.1,
                0.7 if user["PREFERRED_NETWORK"]=="4G" else 0.2,
                0.7 if user["PREFERRED_NETWORK"]=="5G" else 0.3
            ]
        )[0]

        ip = generate_ip_from_subnet(subnet)

        # -------------------------
        # Inject anomalies
        # -------------------------

        if random.random() < ANOMALY_RATE:

            anomaly = random.choice([
                "device_change",
                "location_jump",
                "network_switch",
                "ip_jump"
            ])

            if anomaly == "device_change":
                device = generate_imei()

            elif anomaly == "location_jump":
                new_mnc = str(random.randint(1,99)).zfill(2)
                lac = random.randint(1000,9999)
                rac = random.randint(10,99)
                lai = f"{MCC}-{new_mnc}-{lac}"
                rai = f"{lai}-{rac}"

            elif anomaly == "network_switch":
                network = random.choice(NETWORK_TYPES)

            elif anomaly == "ip_jump":
                subnet = f"{random.randint(10,200)}.{random.randint(0,255)}.{random.randint(0,255)}"
                ip = generate_ip_from_subnet(subnet)

        tmsi = random.randint(10000000,99999999)
        lmsi = random.randint(10000000,99999999)
        tlli = random.randint(10000000,99999999)

        data.append([
            timestamp,
            user["IMSI"],
            user["MSISDN"],
            device,
            ip,
            MCC,
            base_mnc,
            plmn,
            tmsi,
            lmsi,
            tlli,
            lai,
            rai,
            network
        ])


# -----------------------------
# CREATE DATAFRAME
# -----------------------------

columns = [
"Timestamp",
"IMSI","MSISDN","IMEI","IP_Address",
"MCC","MNC","PLMN",
"TMSI","LMSI","TLLI",
"LAI","RAI","Network_Type"
]

df = pd.DataFrame(data,columns=columns)

# sort timeline (realistic traffic stream)
df = df.sort_values("Timestamp").reset_index(drop=True)

# -----------------------------
# SAVE DATASET
# -----------------------------

df.to_csv("telecom_dataset.csv",index=False)

print("Dataset generated successfully")
print("Total rows:",len(df))
print(df.head())