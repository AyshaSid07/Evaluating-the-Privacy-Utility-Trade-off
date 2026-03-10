# import pandas as pd
# import random
# from faker import Faker

# fake = Faker()

# rows = 10000
# unique_rows = 4000
# duplicate_pairs = 3000

# data = []

# MCC = "240"

# imsi_set=set()
# msisdn_set=set()
# imei_set=set()
# ip_set=set()


# def unique_number(length, used):
#     while True:
#         num=''.join(str(random.randint(0,9)) for _ in range(length))
#         if num not in used:
#             used.add(num)
#             return num


# def unique_ip():
#     while True:
#         ip=fake.ipv4()
#         if ip not in ip_set:
#             ip_set.add(ip)
#             return ip


# # -------------------------
# # UNIQUE RECORDS
# # -------------------------

# for i in range(unique_rows):

#     mnc = f"{random.randint(1,99):02d}"
#     plmn = MCC + mnc

#     lac=random.randint(1000,9999)
#     rac=random.randint(10,99)

#     lai=f"{MCC}-{mnc}-{lac}"
#     rai=f"{lai}-{rac}"

#     # Network_Type derived from LAC
#     if lac < 3500:
#         network = "3G"
#     elif lac < 7000:
#         network = "4G"
#     else:
#         network = "5G"

#     msin=unique_number(10,imsi_set)
#     imsi=MCC+mnc+msin

#     subscriber=unique_number(8,msisdn_set)
#     msisdn="4670"+subscriber

#     tac=str(random.randint(10000000,99999999))
#     serial=unique_number(6,imei_set)
#     imei=tac+serial+str(random.randint(0,9))

#     ip=unique_ip()

#     tmsi=random.randint(10000000,99999999)
#     lmsi=random.randint(10000000,99999999)
#     tlli=random.randint(10000000,99999999)

#     data.append([
#         imsi,msisdn,imei,ip,
#         MCC,mnc,plmn,
#         tmsi,lmsi,tlli,
#         lai,rai,
#         network
#     ])


# # -------------------------
# # DUPLICATE RECORDS
# # -------------------------

# for i in range(duplicate_pairs):

#     mnc = f"{random.randint(1,99):02d}"
#     plmn = MCC + mnc

#     lac=random.randint(1000,9999)
#     rac=random.randint(10,99)

#     lai=f"{MCC}-{mnc}-{lac}"
#     rai=f"{lai}-{rac}"

#     # Same relationship for duplicates
#     if lac < 3500:
#         network = "3G"
#     elif lac < 7000:
#         network = "4G"
#     else:
#         network = "5G"

#     for j in range(2):

#         msin=unique_number(10,imsi_set)
#         imsi=MCC+mnc+msin

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
#             MCC,mnc,plmn,
#             tmsi,lmsi,tlli,
#             lai,rai,
#             network
#         ])


# columns=[
# "IMSI","MSISDN","IMEI","IP_Address",
# "MCC","MNC","PLMN",
# "TMSI","LMSI","TLLI",
# "LAI","RAI","Network_Type"
# ]

# df=pd.DataFrame(data,columns=columns)

# df.to_csv("raw_dataset.csv",index=False)

# print("Dataset generated successfully")
# print(df.head())


import pandas as pd
import random
from faker import Faker

fake = Faker()

rows = 10000
unique_rows = 4000
duplicate_pairs = 3000  # 3000 pairs → 6000 rows

data = []

MCC = "240"

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

def random_record(imsi, msisdn, imei):
    mnc = f"{random.randint(1,99):02d}"
    plmn = MCC + mnc

    lac = random.randint(1000, 9999)
    rac = random.randint(10, 99)

    lai = f"{MCC}-{mnc}-{lac}"
    rai = f"{lai}-{rac}"

    if lac < 3500:
        network = "3G"
    elif lac < 7000:
        network = "4G"
    else:
        network = "5G"

    ip = unique_ip()
    tmsi = random.randint(10000000, 99999999)
    lmsi = random.randint(10000000, 99999999)
    tlli = random.randint(10000000, 99999999)

    return [
        imsi, msisdn, imei, ip,
        MCC, mnc, plmn,
        tmsi, lmsi, tlli,
        lai, rai,
        network
    ]

# UNIQUE SUBSCRIBERS
for _ in range(unique_rows):
    msin = unique_number(10, imsi_set)
    imsi = MCC + f"{random.randint(1,99):02d}" + msin

    subscriber = unique_number(8, msisdn_set)
    msisdn = "4670" + subscriber

    tac = str(random.randint(10000000, 99999999))
    serial = unique_number(6, imei_set)
    imei = tac + serial + str(random.randint(0, 9))

    rec = random_record(imsi, msisdn, imei)
    data.append(rec)

# TRUE DUPLICATE PAIRS (same subscriber, different sessions)
for _ in range(duplicate_pairs):
    msin = unique_number(10, imsi_set)
    imsi = MCC + f"{random.randint(1,99):02d}" + msin

    subscriber = unique_number(8, msisdn_set)
    msisdn = "4670" + subscriber

    tac = str(random.randint(10000000, 99999999))
    serial = unique_number(6, imei_set)
    imei = tac + serial + str(random.randint(0, 9))

    rec1 = random_record(imsi, msisdn, imei)
    rec2 = random_record(imsi, msisdn, imei)

    data.extend([rec1, rec2])

columns = [
    "IMSI","MSISDN","IMEI","IP_Address",
    "MCC","MNC","PLMN",
    "TMSI","LMSI","TLLI",
    "LAI","RAI","Network_Type"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("raw_dataset.csv", index=False)
print("Raw dataset with true duplicates generated")
print(df.head())
