import pandas as pd
from anjana.anonymity import k_anonymity, utils
import pycanon
import time
import numpy as np
import pandas as pd


data = pd.read_csv("../data.csv", sep=';')
data.columns = data.columns.str.strip()

# Round the continuous floats to integers before Anjana processes them
continuous_cols = [
    'Previous qualification (grade)', 'Admission grade', 
    'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

for col in continuous_cols:
    data[col] = np.round(data[col]).astype(int).astype(str)

cols = [
    "Marital status", "Application mode", "Application order", "Course", 
    "Daytime/evening attendance", "Previous qualification", 
    "Previous qualification (grade)", "Nacionality", 
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Admission grade",
    "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date",
    "Gender", "Scholarship holder", "Age at enrollment", "International",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP"
]

# Fix the floats ("30.0" -> "30") in the dataset
for col in cols:
	data[col] = data[col].astype(float).astype(int).astype(str).str.strip()

print(data["Marital status"].min())

quasi_ident = [
    "Marital status", "Application mode", "Application order", "Course", 
    "Daytime/evening attendance", "Previous qualification", 
    "Previous qualification (grade)", "Nacionality", 
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Admission grade",
    "Displaced", "Gender", "Scholarship holder", "Age at enrollment", "International",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP"
]

k = 2
supp_level = 5

def load_hierarchy(filename):
    df = pd.read_csv(filename, header=None, dtype=str)
    
    df = df.fillna("*")
    
    df = df.apply(lambda col: col.str.strip())
    
    return dict(df)

hierarchies = {
    # Demographics & Background
    "Marital status": load_hierarchy("hierarchies/marital.csv"),
    "Nacionality": load_hierarchy("hierarchies/nacionality.csv"),
    "Displaced": load_hierarchy("hierarchies/binary.csv"),
    "Gender": load_hierarchy("hierarchies/binary.csv"),
    "Age at enrollment": load_hierarchy("hierarchies/age_enrollment.csv"),
    "International": load_hierarchy("hierarchies/binary.csv"),
    
    # Application & Enrollment
    "Application mode": load_hierarchy("hierarchies/app_mode.csv"),
    "Application order": load_hierarchy("hierarchies/app_order.csv"),
    "Course": load_hierarchy("hierarchies/course.csv"),
    "Daytime/evening attendance": load_hierarchy("hierarchies/attendance.csv"),
    "Scholarship holder": load_hierarchy("hierarchies/binary.csv"),
    
    # Qualifications & Grades
    "Previous qualification": load_hierarchy("hierarchies/qualifications.csv"),
    "Previous qualification (grade)": load_hierarchy("hierarchies/grades.csv"),
    "Admission grade": load_hierarchy("hierarchies/admission_grade.csv"),
    
    # Parents' Background
    "Mother's qualification": load_hierarchy("hierarchies/qualifications.csv"), 
    "Father's qualification": load_hierarchy("hierarchies/qualifications.csv"),
    "Mother's occupation": load_hierarchy("hierarchies/occupations.csv"),
    "Father's occupation": load_hierarchy("hierarchies/occupations.csv"),
    
    # Semester 1 Academic Performance
    "Curricular units 1st sem (credited)": load_hierarchy("hierarchies/curricular_units.csv"),
    "Curricular units 1st sem (enrolled)": load_hierarchy("hierarchies/curricular_units.csv"),
    "Curricular units 1st sem (evaluations)": load_hierarchy("hierarchies/curricular_units.csv"),
    "Curricular units 1st sem (approved)": load_hierarchy("hierarchies/curricular_units.csv"),
    "Curricular units 1st sem (without evaluations)": load_hierarchy("hierarchies/curricular_units.csv"),
    "Curricular units 1st sem (grade)": load_hierarchy("hierarchies/course_grade.csv"),
    
    # Semester 2 Academic Performance
    "Curricular units 2nd sem (credited)": load_hierarchy("hierarchies/curricular_units.csv"),
    "Curricular units 2nd sem (enrolled)": load_hierarchy("hierarchies/curricular_units.csv"),
    "Curricular units 2nd sem (evaluations)": load_hierarchy("hierarchies/curricular_units.csv"),
    "Curricular units 2nd sem (approved)": load_hierarchy("hierarchies/curricular_units.csv"),
    "Curricular units 2nd sem (without evaluations)": load_hierarchy("hierarchies/curricular_units.csv"),
    "Curricular units 2nd sem (grade)": load_hierarchy("hierarchies/course_grade.csv"),
    
    # Macroeconomic Indicators
    "Unemployment rate": load_hierarchy("hierarchies/unemployment.csv"),
    "Inflation rate": load_hierarchy("hierarchies/inflation.csv"),
    "GDP": load_hierarchy("hierarchies/gdp.csv")
}
# we can only have 1 sensitive attribute with anjana to when applying l-diversity and T-closeness
# sensitive = "Educational special needs"
# other attributes we drop:
other_sensitive = [
    "Educational special needs", 
    "Debtor", 
    "Tuition fees up to date"
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

data_anon.to_csv("sml_k=2.csv", index=False)

records_suppressed = len(data) - len(data_anon)
print(f"Number of records suppressed: {records_suppressed}")
print(f"Percentage of records suppressed: {100 * records_suppressed / len(data):.2f} %")

print(utils.get_transformation(data_anon, quasi_ident, hierarchies))
