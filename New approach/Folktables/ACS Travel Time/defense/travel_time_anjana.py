import pandas as pd
from anjana.anonymity import k_anonymity, l_diversity, utils
import pycanon
import time

data = pd.read_csv("../folktables_travel_time_RAW.csv")
data.columns = data.columns.str.strip()
# 4. ACSTravelTime Configuration

cols = ["AGEP", "SCHL", "MAR", "SEX", "ESP", "MIG", "RELP", "ST", "CIT", "OCCP", "JWTR"]

# Fix the floats ("30.0" -> "30") in the dataset
for col in cols:
	data[col] = data[col].astype(float).astype(int).astype(str).str.strip()


quasi_ident = ["AGEP", "SCHL", "MAR", "SEX", "ESP", "MIG", "RELP", "ST", "CIT", "OCCP", "JWTR"]



k = 10
l_div = 2
supp_level = 20

def load_hierarchy(filename):
    df = pd.read_csv(filename, header=None, dtype=str)
    
    df = df.fillna("*")
    
    df = df.apply(lambda col: col.str.strip())
    
    return dict(df)

hierarchies = {
    "AGEP": load_hierarchy("../../hierarchies/agep.csv"),
    "SCHL": load_hierarchy("../../hierarchies/schl.csv"),
    "MAR": load_hierarchy("../../hierarchies/mar.csv"),
    "SEX": load_hierarchy("../../hierarchies/sex.csv"),
    "ESP": load_hierarchy("../../hierarchies/esp.csv"),
    "MIG": load_hierarchy("../../hierarchies/mig.csv"),
    "RELP": load_hierarchy("../../hierarchies/relp.csv"),
    "ST": load_hierarchy("../../hierarchies/pobp.csv"), # State codes use the same logic as POBP
    "CIT": load_hierarchy("../../hierarchies/cit.csv"),
    "OCCP": load_hierarchy("../../hierarchies/occp.csv"),
    "JWTR": load_hierarchy("../../hierarchies/jwtr.csv")
}


# These are considered sensitive according to GDPR
sensitive = "RAC1P"
other_sensitive = ["DIS", "POVPIP", "RAC1P"]

# also drop PINCP (total person’s income), as it is also sensitive
data.drop(other_sensitive, axis = 1, inplace = True) 
identifiers = []
# 6. Run Anjana
print(f"Starting Anjana with k={k} and l={l_div} on {len(data)} rows...")
start = time.time()

# Passed empty list [] for direct identifiers
data_anon = k_anonymity(data, identifiers, quasi_ident, k, supp_level, hierarchies)
# data_anon = l_diversity(data_anon, identifiers, quasi_ident, sensitive, k, l_div, supp_level, hierarchies)
end = time.time()

print(f"Elapsed time: {end-start:.2f} seconds")
print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon, quasi_ident)}")

data_anon.to_csv("travel_time_k=10.csv", index=False)

records_suppressed = len(data) - len(data_anon)
print(f"Number of records suppressed: {records_suppressed}")
print(f"Percentage of records suppressed: {100 * records_suppressed / len(data):.2f} %")

print(utils.get_transformation(data_anon, quasi_ident, hierarchies))
