import pandas as pd
from pycanon import anonymity

print("--- Startar Option B: Teoretisk Riskberäkning (pycanon) ---")

# 1. Ladda in den anonymiserade datan (outputen från Anjana/ARX)
# Byt ut namnet till din faktiska k=2 fil
df_anon = pd.read_csv('../datasets/credit_card_k=2.csv')

# 2. Definiera vilka kolumner som är Quasi-Identifiers (QIs)
quasi_identifiers = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'LIMIT_BAL']

# 3. Definiera er Känsliga Attribut (Sensitive Attribute)
# För l-diversitet behöver matematiken veta vilken kolumn vi försöker dölja.
sensitive_attribute = ['default payment']

# 4. Beräkna k-anonymitet
# pycanon går igenom alla rader och kollar storleken på den MINSTA gruppen
k_value = anonymity.k_anonymity(df_anon, quasi_identifiers)

# 5. Beräkna l-diversitet
# pycanon kollar om grupperna har tillräcklig variation i den känsliga kolumnen
l_value = anonymity.l_diversity(df_anon, quasi_identifiers, sensitive_attribute)
# Befintlig kod...
k_value = anonymity.k_anonymity(df_anon, quasi_identifiers)
l_value = anonymity.l_diversity(df_anon, quasi_identifiers, sensitive_attribute)

# --- NYA MÄTNINGAR ---

# 1. t-closeness
# Mäter hur mycket fördelningen av 'default payment' i grupperna avviker från det normala
t_value = anonymity.t_closeness(df_anon, quasi_identifiers, sensitive_attribute)

# 2. Ekvivalensklass-storlekar (För att diskutera Information Loss)
# pycanon kan hämta ut exakt hur stora alla grupper blev!
# 2. Ekvivalensklass-storlekar (Beräknat med Pandas istället!)
# Vi ber Pandas gruppera all data baserat på era QIs och räkna hur många rader som hamnar i varje grupp.
class_sizes = df_anon.groupby(quasi_identifiers).size()

avg_class_size = class_sizes.mean()
max_class_size = class_sizes.max()

print(f"Faktisk t-closeness uppnådd  : t = {t_value:.4f}")
print(f"Genomsnittlig gruppstorlek   : {avg_class_size:.2f} rader (Målet var {k_value})")
print(f"Största gruppen i datasetet  : {max_class_size} rader")
# 6. Kalkylera den teoretiska risken
# Enligt litteraturen är maxrisken för Identity Disclosure 1/k.
theoretical_risk = (1 / k_value) * 100

print(f"\nResultat för: {file_name}")
print("-" * 40)
print(f"Faktisk k-anonymitet uppnådd : k = {k_value}")
print(f"Faktisk l-diversitet uppnådd : l = {l_value}")
print(f"Teoretisk Identity Risk (1/k): {theoretical_risk:.2f}%")
print("-" * 40)

# Sanity Check för er uppsats
if k_value >= 2:
    print("[OK] Anjana/ARX gjorde sitt jobb korrekt! Datan uppfyller k>=2.")
else:
    print("[VARNING] Verktyget misslyckades. Vissa rader är fortfarande unika (k=1)!")