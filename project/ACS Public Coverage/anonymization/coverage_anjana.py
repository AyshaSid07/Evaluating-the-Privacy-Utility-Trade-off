import pandas as pd
from anjana.anonymity import k_anonymity, l_diversity, t_closeness, utils
import pycanon
import time

data = pd.read_csv("../datasets/folktables_public_coverage_RAW.csv")
data.columns = data.columns.str.strip()

# 1. ACSPublic Coverage Configuration
quasi_ident = ["AGEP", "SCHL", "SEX", "MAR", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY", "ESR"]

# Fix the floats ("30.0" -> "30") in the dataset
for col in quasi_ident:
	data[col] = data[col].astype(float).astype(int).astype(str).str.strip()





k = 5
l = [3, 4]
t = [0.2, 0.3]
supp_level = 5

def load_hierarchy(filename):
    df = pd.read_csv(filename, header=None, dtype=str)
    
    df = df.fillna("*")
    
    df = df.apply(lambda col: col.str.strip())
    
    return dict(df)

hierarchies = {
    "AGEP": load_hierarchy("hierarchies/agep.csv"),
    "SCHL": load_hierarchy("hierarchies/schl.csv"),
    "MAR": load_hierarchy("hierarchies/mar.csv"),
    "SEX": load_hierarchy("hierarchies/sex.csv"),
    "ESP": load_hierarchy("hierarchies/esp.csv"),
    "CIT": load_hierarchy("hierarchies/cit.csv"),
    "MIG": load_hierarchy("hierarchies/mig.csv"),
    "MIL": load_hierarchy("hierarchies/mil.csv"),
    "ANC": load_hierarchy("hierarchies/anc.csv"),
    "NATIVITY": load_hierarchy("hierarchies/nativity.csv"),
    "ESR": load_hierarchy("hierarchies/esr.csv"),
    # "RAC1P" : load_hierarchy("hierarchies/rac1p.csv")
}

sensitive = "RAC1P"  
# also drop PINCP (total person’s income), as it is also sensitive
# 6. Run Anjana
# print(f"Running k-anonymity with k={k}...")
data_anon_k = k_anonymity(data, [], quasi_ident,k, supp_level, hierarchies)
# print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon_k, quasi_ident)}")
data_anon_k.to_csv(f"../datasets/anjana_public_coverage_race_k={k}.csv", index=False)
# 6. Run Anjana
# for l_val in l:
    # print(f"Running l-diversity with l={l_val}...")
    # data_anon_l = l_diversity(data, [], quasi_ident, sensitive, k, l_val, supp_level, hierarchies)
    # print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon_l, quasi_ident)}")
    # data_anon_l.to_csv(f"../datasets/anjana_public_coverage_race_k={k}_l={l_val}.csv", index=False)
# for t_val in t:
    # print(f"Running T-closeness with t={t_val}...")
    # data_anon_t = t_closeness(data, [], quasi_ident, sensitive, k, t_val, supp_level, hierarchies)
    # print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon_t, quasi_ident)}")
    # data_anon_t.to_csv(f"../datasets/anjana_public_coverage_race_k={k}_t={t_val}.csv", index=False)
# print(f"Starting Anjana with k={k} and l={l_div} on {len(data)} rows...")
# start = time.time()

# # Passed empty list [] for direct identifiers
# data_anon = k_anonymity(data, identifiers, quasi_ident, k, supp_level, hierarchies)
# # data_anon = l_diversity(data_anon, identifiers, quasi_ident, sensitive, k, l_div, supp_level, hierarchies)
# end = time.time()

# print(f"Elapsed time: {end-start:.2f} seconds")
# print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon, quasi_ident)}")

# data_anon.to_csv("public_coverage_k=10_l=2.csv", index=False)

# records_suppressed = len(data) - len(data_anon)
# print(f"Number of records suppressed: {records_suppressed}")
# print(f"Percentage of records suppressed: {100 * records_suppressed / len(data):.2f} %")

# print(utils.get_transformation(data_anon, quasi_ident, hierarchies))
