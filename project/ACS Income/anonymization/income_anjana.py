import pandas as pd
from anjana.anonymity import k_anonymity, utils
import pycanon
import time

data = pd.read_csv("../folktables_income_RAW.csv")
data.columns = data.columns.str.strip()

cols = ["AGEP", "SCHL", "COW", "MAR", "RAC1P", "RELP", "WKHP", "POBP", "OCCP"]

# Fix the floats ("30.0" -> "30") in the dataset
for col in cols:
	data[col] = data[col].astype(float).astype(int).astype(str).str.strip()

# Drop these as they are highly identifieable and we do not have hierarchies for them.

quasi_ident = ["AGEP","SCHL", "COW", "MAR", "RAC1P", "RELP", "WKHP", "POBP", "OCCP"]
k = 5
supp_level = 20

def load_hierarchy(filename):
    df = pd.read_csv(filename, header=None, dtype=str)
    
    df = df.fillna("*")
    
    df = df.apply(lambda col: col.str.strip())
    
    return dict(df)

hierarchies = {
    "AGEP": load_hierarchy("../../hierarchies/agep.csv"),
    "SCHL": load_hierarchy("../../hierarchies/schl.csv"),
    "COW": load_hierarchy("../../hierarchies/cow.csv"),
    "MAR": load_hierarchy("../../hierarchies/mar.csv"),
    "RAC1P": load_hierarchy("../../hierarchies/rac1p.csv"),
    "RELP" : load_hierarchy("../../hierarchies/relp.csv"),
    "WKHP" : load_hierarchy("../../hierarchies/wkhp.csv"),
    "POBP" : load_hierarchy("../../hierarchies/pobp.csv"),
    "OCCP" : load_hierarchy("../../hierarchies/occp.csv")
}

# Race is considered sensitive according to GDPR
sensitive="RAC1P" # sensitive attributes is only used in l-diversity and T-closeness

# 6. Run Anjana
print(f"Starting Anjana with k={k} on {len(data)} rows...")
start = time.time()

# Passed empty list [] for direct identifiers
data_anon = k_anonymity(data, [], quasi_ident, k, supp_level, hierarchies)
end = time.time()

print(f"Elapsed time: {end-start:.2f} seconds")
print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon, quasi_ident)}")

data_anon.to_csv("income_k=5.csv", index=False)

records_suppressed = len(data) - len(data_anon)
print(f"Number of records suppressed: {records_suppressed}")
print(f"Percentage of records suppressed: {100 * records_suppressed / len(data):.2f} %")

print(utils.get_transformation(data_anon, quasi_ident, hierarchies))
