import pandas as pd
from snsynth import Synthesizer
import os

# ==============================
# Configuration
# ==============================
OUTPUT_DIR    = '../datasets/'
TARGET_COLUMN = 'default payment'
REALIZATIONS  = 3

# Define the runs to execute: (Input File, Output Prefix, List of Epsilons)
runs = [
    # 1. Pure Differential Privacy
    ('../datasets/credit_card_clients_train.csv', 'DP', [0.5, 1.0, 3.0, 5.0, 10.0]),
    
    # 2. Layered Approach (k=3 + DP)
    ('../datasets/ARX_credit_card_clients_k3_cleaned.csv', 'combined_k=3', [0.5, 1.0, 3.0]),
    
    # 3. Layered Approach (k=5 + DP)
    ('../datasets/ARX_credit_card_clients_k5_cleaned.csv', 'combined_k=5', [0.5, 1.0, 3.0])
]

continuous_cols = [
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Helper Function
# ==============================
def convert_val(val):
    """
    Safely converts strings and ARX intervals back to continuous floats
    so MST can process them with Gaussian noise.
    """
    val = str(val).strip()
    if val == "*": return 0.0
    if val.startswith("[") and val.endswith("["):
        parts = val[1:-1].split(",")
        try: return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception: return 0.0
    try: return float(val)
    except Exception: return 0.0


# ==============================
# Main Generation Loop
# ==============================
for input_file, prefix, epsilons in runs:
    print(f"\n=======================================================")
    print(f"Processing: {prefix} (from {input_file})")
    print(f"=======================================================")
    
    df_original = pd.read_csv(input_file)
    
    # Clean up columns not needed for synthesis
    cols_to_drop = ["Linkage_Index", "index", "LIMIT_BAL"]
    for col in cols_to_drop:
        if col in df_original.columns:
            df_original = df_original.drop(columns=[col])
            
    categorical_cols = [c for c in df_original.columns if c not in continuous_cols and c != TARGET_COLUMN]
    
    # Format target and categorical columns
    df_original[TARGET_COLUMN] = df_original[TARGET_COLUMN].astype(int) 
    for col in categorical_cols:
        df_original[col] = df_original[col].astype(str)    
        
    # Format continuous columns using the ARX-safe converter and force to float
    for col in continuous_cols:
        df_original[col] = df_original[col].apply(convert_val)
        df_original[col] = pd.to_numeric(df_original[col], errors='coerce').astype(float)

    # Drop any remaining NaNs 
    df_original = df_original.dropna()
    
    # Ensure target column is included in categorical features for the synthesizer
    synth_categorical_cols = categorical_cols + [TARGET_COLUMN]
    
    # Reorder DataFrame
    df_original = df_original[synth_categorical_cols + continuous_cols].copy()

    print(f"Prepared dataset: {df_original.shape[0]} rows, {df_original.shape[1]} cols\n")

    for epsilon in epsilons:
        for realization in range(1, REALIZATIONS + 1):
            print(f"  Generating {prefix} | eps = {epsilon} | realization = {realization}")
            
            # Create a brand new Synthesizer object for EVERY realization
            # to ensure the budget is not mutated across runs.
            synth = Synthesizer.create(
                "mst",
                epsilon=epsilon,
                verbose=False
            )
            
            synth.fit(
                df_original,
                categorical_columns=synth_categorical_cols,
                continuous_columns=continuous_cols,
                preprocessor_eps=epsilon * 0.1
            )

            df_synthetic = synth.sample(len(df_original))

            # Restore correct formatting for output
            for col in synth_categorical_cols:
                df_synthetic[col] = df_synthetic[col].astype(str)
                
            for col in continuous_cols:
                # Återställ till float efter tillbakaskalningen
                df_synthetic[col] = pd.to_numeric(df_synthetic[col], errors='coerce')
                
            # Create file name based on prefix
            eps_str = str(epsilon).replace('.', '_')
            if prefix == 'DP':
                out_path = f"{OUTPUT_DIR}DP_credit_card_clients_epsilon_{eps_str}_r{realization}.csv"
            else:
                out_path = f"{OUTPUT_DIR}{prefix}_epsilon_{eps_str}_r{realization}_credit_card_clients.csv"
                
            df_synthetic.to_csv(out_path, index=False)
            print(f"  -> Saved: {out_path}\n")

print("\n=== Generation Complete ===")