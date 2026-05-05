import pandas as pd
from pycanon import anonymity

datasets_to_test = {
    "Baseline (No anonymization)": pd.read_csv('../datasets/folktables_public_coverage_RAW.csv'),
    "ARX Public Coverage, k=10": pd.read_csv('../datasets/ARX_public_coverage_k=10.csv'),
    "ARX ACS Public Coverage k-Anonymity k=20":          pd.read_csv('../datasets/ARX_public_coverage_k=20.csv'),
    "ARX ACS Public Coverage k-Anonymity k=50":          pd.read_csv('../datasets/ARX_public_coverage_k=50.csv'),
}

QUASI_IDENTIFIERS = ['AGEP', 'SCHL', 'SEX', 'MAR', 'ESP', 'CIT', 'MIG', 'MIL', 'NATIVITY', 'RAC1P']
SENSITIVE_ATTRIBUTE = ['ESR'] 

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