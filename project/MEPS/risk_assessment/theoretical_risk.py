import pandas as pd
from pycanon import anonymity

df_anon = pd.read_csv('../datasets/folktables_income_RAW.csv')

quasi_identifiers = ['AGEP', 'COW', 'SCHL', 'MAR', 'SEX']

sensitive_attribute = ['RAC1P']

k_value = anonymity.k_anonymity(df_anon, quasi_identifiers)
l_value = anonymity.l_diversity(df_anon, quasi_identifiers, sensitive_attribute)
t_value = anonymity.t_closeness(df_anon, quasi_identifiers, sensitive_attribute)

class_sizes = df_anon.groupby(quasi_identifiers).size()

avg_class_size = class_sizes.mean()
max_class_size = class_sizes.max()

print(f"Average class size   : {avg_class_size:.2f} rows, (the goal was {k_value})")
print(f"Largest class size   : {max_class_size} rows")

# Identity Disclosure risk is usually 1/k .
theoretical_risk = (1 / k_value) * 100

print("-" * 40)
print(f"k-anonymity : k = {k_value}")
print(f"l-diversity : l = {l_value}")
print(f"T-closeness : t = {t_value:.4f}")
print(f"Theoretical Identity Risk (1/k): {theoretical_risk:.2f}%")
print("-" * 40)
