import pandas as pd
from pycanon import anonymity

datasets_to_test = {
    "No anonymization (Baseline)": pd.read_csv('../datasets/bank-additional-full.csv', sep=';'),
    "ARX Bank_marketing k = 5": pd.read_csv('../datasets/ARX_bank_marketing_k=5.csv', sep=';'),
    "ARX Bank_marketing k = 5, l = 2": pd.read_csv('../datasets/ARX_bank_marketing_k=5_l=2.csv', sep=';'),
    "ARX Bank_marketing k = 5, l = 3": pd.read_csv('../datasets/ARX_bank_marketing_k=5_l=3.csv', sep=';'),
    "ARX Bank_marketing k = 5, t = 0.2": pd.read_csv('../datasets/ARX_bank_marketing_k=5_t=0.2.csv', sep=';'),
    "ARX Bank_marketing k = 5, t = 0.1": pd.read_csv('../datasets/ARX_bank_marketing_k=5_t=0.1.csv', sep=';'),
}

QUASI_IDENTIFIERS = ["age", "job", "education"]
SENSITIVE_ATTRIBUTE = ["marital"] 

results = []

print("Calculating theoretical anonymity metrics...")

for name, df in datasets_to_test.items():
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
        
    for col in QUASI_IDENTIFIERS + SENSITIVE_ATTRIBUTE:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col] == 'nan', col] = '*' 
            
    try:
        k_val = anonymity.k_anonymity(df, QUASI_IDENTIFIERS)
        l_val = anonymity.l_diversity(df, QUASI_IDENTIFIERS, SENSITIVE_ATTRIBUTE)
        t_val = anonymity.t_closeness(df, QUASI_IDENTIFIERS, SENSITIVE_ATTRIBUTE)
    except Exception as e:
        print(f"Error on {name}: {e}")
        k_val, l_val, t_val = 0, 0, 1.0 
    
    id_risk = f"{(1 / k_val)*100:.2f}%" if k_val > 0 else "100%"

    results.append({
        'Dataset': name,
        'k-Anonymity': k_val,
        'l-Diversity': l_val,
        't-Closeness': round(t_val, 4),
        'Identity Risk': id_risk
    })

results_df = pd.DataFrame(results)

def create_markdown_table(df):
    header = "| " + " | ".join(df.columns) + " |\n"
    separator = "|-" + "-|-".join(["-" * len(col) for col in df.columns]) + "-|\n"
    rows = ""
    for _, row in df.iterrows():
        rows += "| " + " | ".join(str(val) for val in row.values) + " |\n"
    return header + separator + rows

markdown_table = create_markdown_table(results_df)

print("\n=== Theoretical Privacy Assessment ===")
print(markdown_table)

save_path = "../plots/theoretical_privacy_assessment.md"
with open(save_path, "w", encoding="utf-8") as f:
    f.write("### Theoretical Privacy Assessment\n\n")
    f.write(markdown_table)

