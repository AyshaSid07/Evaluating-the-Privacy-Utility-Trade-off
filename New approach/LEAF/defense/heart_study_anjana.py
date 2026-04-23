import pandas as pd
from anjana.anonymity import k_anonymity, utils
import pycanon
import time
import numpy as np
import pandas as pd


data = pd.read_csv("../framingham_heart_study.csv")
data.columns = data.columns.str.strip()

quasi_ident = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay']

for col in quasi_ident:
    # This safely converts 1.0 -> "1", and np.nan -> "nan"
    data[col] = data[col].apply(lambda x: "nan" if pd.isna(x) else str(int(x)))

k = 2
supp_level = 5

def load_hierarchy(filename):
    df = pd.read_csv(filename, header=None, dtype=str)
    
    df = df.fillna("*")
    
    df = df.apply(lambda col: col.str.strip())
    
    return dict(df)

hierarchies = {
    "male": load_hierarchy("hierarchies/male.csv"),
    "age": load_hierarchy("hierarchies/age.csv"),
    "education": load_hierarchy("hierarchies/education.csv"),
    "currentSmoker": load_hierarchy("hierarchies/currentSmoker.csv"),
    "cigsPerDay": load_hierarchy("hierarchies/cigsPerDay.csv")
}

# we can only have 1 sensitive attribute with anjana to when applying l-diversity and T-closeness
# sensitive = 'BPMeds'
# other attributes we drop:
other_sensitive = [
    'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 
    'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 
]
data = data.drop(columns=other_sensitive, errors='ignore') # Using reassignment instead of inplace

# 6. Run Anjana
print(f"Starting Anjana with k={k} on {len(data)} rows...")
start = time.time()

# Passed empty list [] for direct identifiers
data_anon = k_anonymity(data, [], quasi_ident, k, supp_level, hierarchies)
end = time.time()

print(f"Elapsed time: {end-start:.2f} seconds")
print(f"Value of k calculated: {pycanon.anonymity.k_anonymity(data_anon, quasi_ident)}")

data_anon.to_csv("heart_study_k=2.csv", index=False)

records_suppressed = len(data) - len(data_anon)
print(f"Number of records suppressed: {records_suppressed}")
print(f"Percentage of records suppressed: {100 * records_suppressed / len(data):.2f} %")

print(utils.get_transformation(data_anon, quasi_ident, hierarchies))
