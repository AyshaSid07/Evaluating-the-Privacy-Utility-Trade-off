import pandas as pd
from anjana.anonymity import k_anonymity, t_closeness, l_diversity, utils
import pycanon
import time

data = pd.read_csv("../datasets/folktables_income_RAW.csv")
data.columns = data.columns.str.strip()

quasi_ident = ["AGEP","SCHL", "COW", "MAR", "RELP", "WKHP", "POBP", "OCCP"]

for col in quasi_ident:
	data[col] = data[col].astype(float).astype(int).astype(str).str.strip()

k = 5

l = [3, 5]

t = [0.15, 0.3]
supp_level = 5

def load_hierarchy(filename):
    df = pd.read_csv(filename, header=None, dtype=str)
    
    df = df.fillna("*")
    
    df = df.apply(lambda col: col.str.strip())
    
    return dict(df)

hierarchies = {
    "AGEP": load_hierarchy("hierarchies/agep.csv"),
    "SCHL": load_hierarchy("hierarchies/schl.csv"),
    "COW": load_hierarchy("hierarchies/cow.csv"),
    "MAR": load_hierarchy("hierarchies/mar.csv"),
    "RELP" : load_hierarchy("hierarchies/relp.csv"),
    "WKHP" : load_hierarchy("hierarchies/wkhp.csv"),
    "POBP" : load_hierarchy("hierarchies/pobp.csv"),
    "OCCP" : load_hierarchy("hierarchies/occp.csv")
}

# Race is considered sensitive according to GDPR
sensitive="RAC1P" # sensitive attributes is only used in l-diversity and T-closeness

data_anon_k = k_anonymity(data, [], quasi_ident,k, supp_level, hierarchies)
# print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon_k, quasi_ident)}")
data_anon_k.to_csv(f"../datasets/anjana_income_race_k={k}.csv", index=False)
# 6. Run Anjana
for l_val in l:
    print(f"Running l-diversity with l={l_val}...")
    data_anon_l = l_diversity(data, [], quasi_ident, sensitive, k, l_val, supp_level, hierarchies)
    print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon_l, quasi_ident)}")
    data_anon_l.to_csv(f"../datasets/anjana_income_race_k={k}_l={l_val}.csv", index=False)
for t_val in t:
    print(f"Running T-closeness with t={t_val}...")
    data_anon_t = t_closeness(data, [], quasi_ident, sensitive, k, t_val, supp_level, hierarchies)
    print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon_t, quasi_ident)}")
    data_anon_t.to_csv(f"../datasets/anjana_income_race_k={k}_t={t_val}.csv", index=False)
# # 6. Run Anjana
# print(f"Starting Anjana with k={k} on {len(data)} rows...")
# start = time.time()

# # Passed empty list [] for direct identifiers
# data_anon = k_anonymity(data, [], quasi_ident, k, supp_level, hierarchies)
# # data_anon = t_closeness(data, [], quasi_ident, sensitive, k, t, supp_level, hierarchies)
# end = time.time()

# print(f"Elapsed time: {end-start:.2f} seconds")
# print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon, quasi_ident)}")

# data_anon.to_csv("../datasets/income_k=10.csv", index=False)

# records_suppressed = len(data) - len(data_anon)
# print(f"Number of records suppressed: {records_suppressed}")
# print(f"Percentage of records suppressed: {100 * records_suppressed / len(data):.2f} %")

# print(utils.get_transformation(data_anon, quasi_ident, hierarchies))
