import pandas as pd
from anjana.anonymity import k_anonymity, utils
import pycanon
import time
import numpy as np
import pandas as pd


data = pd.read_csv("../credit-card-clients.csv")
data.columns = data.columns.str.strip()

financial_and_age_cols = [
    'LIMIT_BAL', 'AGE', 
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]
for col in financial_and_age_cols:
    data[col] = np.round(data[col]).astype(int).astype(str)

# Ensure categorical codes are strings (no ".0")
cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
for col in cat_cols:
    data[col] = data[col].astype(float).astype(int).astype(str)

quasi_ident = cat_cols + financial_and_age_cols

k = 2
supp_level = 5

def load_hierarchy(filename):
    df = pd.read_csv(filename, header=None, dtype=str)
    
    df = df.fillna("*")
    
    df = df.apply(lambda col: col.str.strip())
    
    return dict(df)

hierarchies = {
    "SEX": load_hierarchy("hierarchies/sex.csv"),
    "EDUCATION": load_hierarchy("hierarchies/education.csv"),
    "MARRIAGE": load_hierarchy("hierarchies/marriage.csv"),
    "AGE": load_hierarchy("hierarchies/age.csv"), 
    # "LIMIT_BAL": load_hierarchy("hierarchies/limit_bal.csv"),
    
    # # Repayment Statuses (Reused 6 times)
    # "PAY_0": load_hierarchy("hierarchies/pay_status.csv"),
    # "PAY_2": load_hierarchy("hierarchies/pay_status.csv"),
    # "PAY_3": load_hierarchy("hierarchies/pay_status.csv"),
    # "PAY_4": load_hierarchy("hierarchies/pay_status.csv"),
    # "PAY_5": load_hierarchy("hierarchies/pay_status.csv"),
    # "PAY_6": load_hierarchy("hierarchies/pay_status.csv"),
    
    # # Bill Amounts (Reused 6 times)
    # "BILL_AMT1": load_hierarchy("hierarchies/bill_amt.csv"),
    # "BILL_AMT2": load_hierarchy("hierarchies/bill_amt.csv"),
    # "BILL_AMT3": load_hierarchy("hierarchies/bill_amt.csv"),
    # "BILL_AMT4": load_hierarchy("hierarchies/bill_amt.csv"),
    # "BILL_AMT5": load_hierarchy("hierarchies/bill_amt.csv"),
    # "BILL_AMT6": load_hierarchy("hierarchies/bill_amt.csv"),

    # # Pay Amounts (Reused 6 times)
    # "PAY_AMT1": load_hierarchy("hierarchies/pay_amt.csv"),
    # "PAY_AMT2": load_hierarchy("hierarchies/pay_amt.csv"),
    # "PAY_AMT3": load_hierarchy("hierarchies/pay_amt.csv"),
    # "PAY_AMT4": load_hierarchy("hierarchies/pay_amt.csv"),
    # "PAY_AMT5": load_hierarchy("hierarchies/pay_amt.csv"),
    # "PAY_AMT6": load_hierarchy("hierarchies/pay_amt.csv")
}
# we can only have 1 sensitive attribute with anjana to when applying l-diversity and T-closeness
# sensitive = "default.payment.next.month"
sensitive = "MARRIAGE"
# other attributes we drop:
# other_sensitive = ["default.payment.next.month"]
# data = data.drop(columns=other_sensitive, errors='signore') # Using reassignment instead of inplace

# 6. Run Anjana
print(f"Starting Anjana with k={k} on {len(data)} rows...")
start = time.time()

# Passed empty list [] for direct identifiers
data_anon = k_anonymity(data, [], quasi_ident, k, supp_level, hierarchies)
end = time.time()

print(f"Elapsed time: {end-start:.2f} seconds")
print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon, quasi_ident)}")

data_anon.to_csv("../datasets/credit_card_k=2.csv", index=False)

records_suppressed = len(data) - len(data_anon)
print(f"Number of records suppressed: {records_suppressed}")
print(f"Percentage of records suppressed: {100 * records_suppressed / len(data):.2f} %")

print(utils.get_transformation(data_anon, quasi_ident, hierarchies))
