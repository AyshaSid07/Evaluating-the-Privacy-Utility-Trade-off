import pandas as pd
from anjana.anonymity import k_anonymity, l_diversity, t_closeness, utils
import pycanon
import time
import numpy as np
import pandas as pd


data=pd.read_csv('../datasets/bank-additional-full.csv', sep = ';')

data.columns = data.columns.str.strip()
if 'duration' in data.columns:
    data = data.drop(columns=['duration'])
integer_cols = ['age']
for col in integer_cols:
    if col in data.columns:
        # Convert to float first to handle any NaNs safely, then int, then string
        data[col] = data[col].astype(float).astype(int).astype(str)

# Ensure categorical QIs are cleanly formatted as strings
other_cols = ['job', 'education', 'marital']
for col in other_cols:
    if col in data.columns:
        data[col] = data[col].astype(str).str.strip()

quasi_ident = integer_cols + other_cols


k = 5
l = [2]
t = [0.15, 0.30]
supp_level = 5

def load_hierarchy(filename):
    df = pd.read_csv(filename, header=None, dtype=str)
    
    df = df.fillna("*")
    
    df = df.apply(lambda col: col.str.strip())
    
    return dict(df)

hierarchies = {
    "age": load_hierarchy("hierarchies/age.csv"),
    "job": load_hierarchy("hierarchies/job.csv"),
    "marital": load_hierarchy("hierarchies/marital.csv"),
    "education": load_hierarchy("hierarchies/education.csv"),
    # "housing": load_hierarchy("hierarchies/binary_bank.csv"),
    # "loan": load_hierarchy("hierarchies/binary_bank.csv"),
    # "contact": load_hierarchy("hierarchies/contact.csv"),
    # "month": load_hierarchy("hierarchies/month.csv"),
    # "day_of_week": load_hierarchy("hierarchies/day_of_week.csv"),
    # "campaign": load_hierarchy("hierarchies/campaign.csv"),
    # "pdays": load_hierarchy("hierarchies/pdays.csv"),
    # "previous": load_hierarchy("hierarchies/previous.csv"),
    # "poutcome": load_hierarchy("hierarchies/poutcome.csv"),
    # "emp.var.rate": load_hierarchy("hierarchies/emp_var_rate.csv"),
    # "cons.price.idx": load_hierarchy("hierarchies/cons_price_idx.csv"),
    # "cons.conf.idx": load_hierarchy("hierarchies/cons_conf_idx.csv"),
    # "euribor3m": load_hierarchy("hierarchies/euribor3m.csv"),
    # "nr.employed": load_hierarchy("hierarchies/nr_employed.csv")
}
# we can only have 1 sensitive attribute with anjana to when applying l-diversity and T-closeness
# sensitive = "default.payment.next.month"
sensitive = 'housing'
# other attributes we drop:
# other_sensitive = ["'default'"]
# data = data.drop(columns=other_sensitive, errors='ignore') # Using reassignment instead of inplace

# 6. Run Anjana
print(f"Starting Anjana with k={k} on {len(data)} rows...")
start = time.time()

# Passed empty list [] for direct identifiers
data_anon_k = k_anonymity(data, [], quasi_ident,k, supp_level, hierarchies)
# print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon_k, quasi_ident)}")
data_anon_k.to_csv(f"../datasets/anjana_bank_marketing_housing_k={k}.csv", index=False)
# 6. Run Anjana
for l_val in l:
    print(f"Running l-diversity with l={l_val}...")
    data_anon_l = l_diversity(data, [], quasi_ident, sensitive, k, l_val, supp_level, hierarchies)
    print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon_l, quasi_ident)}")
    data_anon_l.to_csv(f"../datasets/anjana_bank_marketing_housing_k={k}_l={l_val}.csv", index=False)
for t_val in t:
    print(f"Running T-closeness with t={t_val}...")
    data_anon_t = t_closeness(data, [], quasi_ident, sensitive, k, t_val, supp_level, hierarchies)
    print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon_t, quasi_ident)}")
    data_anon_t.to_csv(f"../datasets/anjana_bank_marketing_housing_k={k}_t={t_val}.csv", index=False)
# end = time.time()

# print(f"Elapsed time: {end-start:.2f} seconds")


# records_suppressed = len(data) - len(data_anon)
# print(f"Number of records suppressed: {records_suppressed}")
# print(f"Percentage of records suppressed: {100 * records_suppressed / len(data):.2f} %")

# print(utils.get_transformation(data_anon, quasi_ident, hierarchies))
