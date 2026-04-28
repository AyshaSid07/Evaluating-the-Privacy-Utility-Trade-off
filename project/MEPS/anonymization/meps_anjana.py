import pandas as pd
from anjana.anonymity import k_anonymity, utils
import pycanon
import time
import numpy as np
import pandas as pd


data = pd.read_csv("../datasets/MEPS.csv")
data.columns = data.columns.str.strip()

quasi_ident = ['AGE', 'REGION', 'MARRY', 'POVCAT', 'SEX', 'RACE']

for col in quasi_ident:
    data[col] = data[col].astype(float).astype(str)

k = 2
supp_level = 5

def load_hierarchy(filename):
    df = pd.read_csv(filename, header=None, dtype=str)
    
    df = df.fillna("*")
    
    df = df.apply(lambda col: col.str.strip())
    
    return dict(df)

hierarchies = {
    "AGE": load_hierarchy("hierarchies/AGE.csv"),
    "REGION": load_hierarchy("hierarchies/REGION.csv"),
    "MARRY": load_hierarchy("hierarchies/MARRY.csv"),
    "POVCAT": load_hierarchy("hierarchies/POVCAT.csv"), 
    "SEX": load_hierarchy("hierarchies/SEX.csv"),
    "RACE": load_hierarchy("hierarchies/RACE.csv")
}
# we can only have 1 sensitive attribute with anjana to when applying l-diversity and T-closeness
# sensitive = "default.payment.next.month"

# 6. Run Anjana
print(f"Starting Anjana with k={k} on {len(data)} rows...")
start = time.time()

# Passed empty list [] for direct identifiers
data_anon = k_anonymity(data, [], quasi_ident, k, supp_level, hierarchies)
end = time.time()

print(f"Elapsed time: {end-start:.2f} seconds")
print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon, quasi_ident)}")

data_anon.to_csv("../datasets/anjana_meps_k=2.csv", index=False)

records_suppressed = len(data) - len(data_anon)
print(f"Number of records suppressed: {records_suppressed}")
print(f"Percentage of records suppressed: {100 * records_suppressed / len(data):.2f} %")

print(utils.get_transformation(data_anon, quasi_ident, hierarchies))
