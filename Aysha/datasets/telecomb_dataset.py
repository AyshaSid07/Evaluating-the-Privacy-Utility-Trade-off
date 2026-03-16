import pandas as pd
import random
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

rows = 10000
data = []
MCC = "240"

# --- SETUP: Define 50 "Cells" (Natural Clusters) ---
cells = []
for _ in range(50):
    mnc = str(random.randint(1, 99)).zfill(2)
    lac = random.randint(1000, 9999)
    rac = random.randint(10, 99)
    cells.append({"MNC": mnc, "LAC": lac, "RAC": rac})

imsi_set, msisdn_set, imei_set, ip_set = set(), set(), set(), set()

def unique_number(length, used):
    while True:
        num = ''.join(str(random.randint(0, 9)) for _ in range(length))
        if num not in used:
            used.add(num)
            return num

# --- GENERATION ---
for i in range(rows):
    cell = random.choice(cells) # Assign user to a cluster
    
    MNC = cell["MNC"]
    PLMN = MCC + MNC
    lai = f"{MCC}-{MNC}-{cell['LAC']}"
    rai = f"{lai}-{cell['RAC']}"

    imsi = MCC + unique_number(10, imsi_set)
    msisdn = "4670" + unique_number(8, msisdn_set)
    imei = str(random.randint(10000000, 99999999)) + unique_number(6, imei_set) + "0"
    ip = fake.ipv4()
    tmsi, lmsi, tlli = [random.randint(10000000, 99999999) for _ in range(3)]

    data.append([
        imsi, msisdn, imei, ip,
        MCC, MNC, PLMN,
        tmsi, lmsi, tlli,
        lai, rai
    ])


columns = ["IMSI", "MSISDN", "IMEI", "IP_Address", "MCC", "MNC", "PLMN", "TMSI", "LMSI", "TLLI", "LAI", "RAI"]
df = pd.DataFrame(data, columns=columns)
df.to_csv("telecomb_dataset.csv", index=False)
print("Dataset generated without Network_Type.")