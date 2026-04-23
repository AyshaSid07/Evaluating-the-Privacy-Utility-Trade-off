import pandas as pd
from anjana.anonymity import k_anonymity, utils
import pycanon
import time
import pandas as pd
import numpy as np


data = pd.read_csv("../risk_factors_cervical_cancer_cleaned.csv")
data.columns = data.columns.str.strip()

cols = [
    "Age", "Number of sexual partners", "First sexual intercourse", 
    "Num of pregnancies", "Smokes", "Smokes (years)", "Smokes (packs/year)", 
    "Hormonal Contraceptives", "Hormonal Contraceptives (years)", "IUD", "IUD (years)"
]
special_qis = [
    "Smokes (years)", 
    "Smokes (packs/year)", 
    "Hormonal Contraceptives (years)", 
    "IUD (years)"
]

# Some QI's has float values that does not make sense, we round them so we can perform the hierarchy on the identifier correctly
# Round to the nearest whole number, convert to integer, then to string
for col in special_qis:
    data[col] = np.round(data[col]).astype(int).astype(str)

# Fix the floats ("30.0" -> "30") in the dataset
for col in cols:
	data[col] = data[col].astype(float).astype(int).astype(str).str.strip()


quasi_ident = [
    "Age", "Number of sexual partners", "First sexual intercourse", 
    "Num of pregnancies", "Smokes", "Smokes (years)", "Smokes (packs/year)", 
    "Hormonal Contraceptives", "Hormonal Contraceptives (years)", "IUD", "IUD (years)"
]

k = 2
supp_level = 5

def load_hierarchy(filename):
    df = pd.read_csv(filename, header=None, dtype=str)
    
    df = df.fillna("*")
    
    df = df.apply(lambda col: col.str.strip())
    
    return dict(df)

hierarchies = {
    "Age": load_hierarchy("hierarchies/age.csv"), 
    "Number of sexual partners": load_hierarchy("hierarchies/sexual_partners.csv"),
    "First sexual intercourse": load_hierarchy("hierarchies/first_sex.csv"),
    "Num of pregnancies": load_hierarchy("hierarchies/pregnancies.csv"),
    "Smokes": load_hierarchy("hierarchies/smokes.csv"),
    "Smokes (years)": load_hierarchy("hierarchies/smokes_years.csv"),
    "Smokes (packs/year)": load_hierarchy("hierarchies/smokes_packs.csv"),
    "Hormonal Contraceptives": load_hierarchy("hierarchies/hormonal_contraceptives.csv"),
    "Hormonal Contraceptives (years)": load_hierarchy("hierarchies/hormonal_years.csv"),
    "IUD": load_hierarchy("hierarchies/iUD.csv"),
    "IUD (years)": load_hierarchy("hierarchies/iud_years.csv"),
}

# we can only have 1 sensitive attribute with anjana to when applying l-diversity and T-closeness
# sensitive = "STDs"
# other attributes we drop:
other_sensitive = [
    "STDs", "STDs (number)", "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx", 
    "Hinselmann", "Schiller", "Citology"
    # Add the rest of the specific STD columns here as well
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

data_anon.to_csv("cervical_cancer_k=5.csv", index=False)

records_suppressed = len(data) - len(data_anon)
print(f"Number of records suppressed: {records_suppressed}")
print(f"Percentage of records suppressed: {100 * records_suppressed / len(data):.2f} %")

print(utils.get_transformation(data_anon, quasi_ident, hierarchies))
