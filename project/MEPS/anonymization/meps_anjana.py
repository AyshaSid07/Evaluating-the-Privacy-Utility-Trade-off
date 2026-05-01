import pandas as pd
from anjana.anonymity import k_anonymity, l_diversity, t_closeness, utils
import pycanon
import time
import numpy as np
import pandas as pd


data = pd.read_csv("../datasets/MEPS.csv")
data.columns = data.columns.str.strip()

quasi_ident = ['AGE', 'REGION','SEX', 'RACE', 'ACTDTY', 'EMPST', 'FTSTU']

for col in quasi_ident:
    data[col] = data[col].astype(float).astype(str)

k = 5
l = [2, 3]
t = [0.1, 0.2]
supp_level = 5

def load_hierarchy(filename):
    df = pd.read_csv(filename, header=None, dtype=str)
    
    df = df.fillna("*")
    
    df = df.apply(lambda col: col.str.strip())
    
    return dict(df)

hierarchies = {
    "AGE": load_hierarchy("hierarchies/AGE.csv"),
    "REGION": load_hierarchy("hierarchies/REGION.csv"),
    # "MARRY": load_hierarchy("hierarchies/MARRY.csv"),
    "SEX": load_hierarchy("hierarchies/SEX.csv"),
    "RACE": load_hierarchy("hierarchies/RACE.csv"),
    "ACTDTY": load_hierarchy("hierarchies/ACTDTY.csv"),
    "EMPST": load_hierarchy("hierarchies/EMPST.csv"),
    "FTSTU": load_hierarchy("hierarchies/FTSTU.csv"),
}
# we can only have 1 sensitive attribute with anjana to when applying l-diversity and T-closeness
sensitive = "MARRY"
print(f"Running k-anonymity with k={k}...")
data_anon_k = k_anonymity(data, [], quasi_ident,k, supp_level, hierarchies)
print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon_k, quasi_ident)}")
data_anon_k.to_csv(f"../datasets/anjana_meps_k={k}.csv", index=False)
# # 6. Run Anjana
# for l_val in l:
#     print(f"Running l-diversity with l={l_val}...")
#     data_anon_l = l_diversity(data, [], quasi_ident, sensitive, k, l_val, supp_level, hierarchies)
#     print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon_l, quasi_ident)}")
#     data_anon_l.to_csv(f"../datasets/anjana_meps_k={k}_l={l_val}.csv", index=False)
# for t_val in t:
#     print(f"Running T-closeness with t={t_val}...")
#     data_anon_t = t_closeness(data, [], quasi_ident, sensitive, k, t_val, supp_level, hierarchies)
#     print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon_t, quasi_ident)}")
#     data_anon_t.to_csv(f"../datasets/anjana_meps_k={k}_t={t_val}.csv", index=False)

# print(f"Starting Anjana with k={k} on {len(data)} rows...")
# start = time.time()

# # Passed empty list [] for direct identifiers
# # data_anon = k_anonymity(data, [], quasi_ident, k, supp_level, hierarchies)
# # data_anon = l_diversity(data, [], quasi_ident, sensitive, k, l, supp_level, hierarchies)
# data_anon = t_closeness(data, [], quasi_ident, sensitive, k, t, supp_level, hierarchies)
# end = time.time()

# print(f"Elapsed time: {end-start:.2f} seconds")
# print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon, quasi_ident)}")
# # data_anon.to_csv(f"../datasets/anjana_meps_k={k}.csv", index=False)
# # data_anon.to_csv(f"../datasets/anjana_meps_k={k}_l={l}.csv", index=False)
# data_anon.to_csv(f"../datasets/anjana_meps_k={k}_t={t}.csv", index=False)

# records_suppressed = len(data) - len(data_anon)
# print(f"Number of records suppressed: {records_suppressed}")
# print(f"Percentage of records suppressed: {100 * records_suppressed / len(data):.2f} %")

# print(utils.get_transformation(data_anon, quasi_ident, hierarchies))
