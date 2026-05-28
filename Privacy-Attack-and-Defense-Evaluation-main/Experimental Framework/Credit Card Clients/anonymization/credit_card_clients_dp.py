import pandas as pd
from snsynth import Synthesizer

OUTPUT_DIR   = '../datasets/'
TARGET_COLUMN = 'default payment'

df_to_test = {
    "k=3": pd.read_csv("../datasets/ARX_credit_card_clients_k3.csv"),
    "k=5": pd.read_csv("../datasets/ARX_credit_card_clients_k5.csv"),
}
# df_original = pd.read_csv("../datasets/credit_card_clients_binned.csv")

continuous_cols  = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
categorical_cols = ['SEX','EDUCATION','MARRIAGE','AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'default payment', 'LIMIT_CATEGORY'] # Ensure to include the binned version of LIMIT_BAL as a categorical column

# Epsilons to test — lower = stronger privacy, higher = more utility
# In the literature, epsilon <= 1 is considered strong privacy,
# 1-10 moderate, and >10 is weak. We test a range so you can show
# the tradeoff curve in your thesis.
# epsilons = [0.5, 1.0, 3.0, 5.0, 10.0]
epsilons = [0.5, 1.0, 3.0]

# Preprocessing
# SmartNoise requires all columns to be the correct type
# convert target column to integer type
for dataset_name, df_original in df_to_test.items():
    df_original[TARGET_COLUMN] = df_original[TARGET_COLUMN].astype(int)
    # convert categorical columns to string type and continuous
    for col in categorical_cols:
        df_original[col] = df_original[col].astype(str)

    for col in continuous_cols:
        df_original[col] = pd.to_numeric(df_original[col], errors='coerce')

    df_original = df_original.dropna()
    df_original = df_original[categorical_cols + continuous_cols].copy()

    print(f"Original dataset: {df_original.shape[0]} rows, {df_original.shape[1]} cols")
    print(f"Columns: {list(df_original.columns)}\n")

    # Generate synthetic datasets for each epsilon 
    # We use the MST (Maximum Spanning Tree) synthesizer — it is the most
    # commonly used DP synthesizer in the privacy literature for tabular data,
    # and is the recommended default in SmartNoise for mixed categorical/continuous.
    #
    # Why MST?
    #   - Handles mixed data types (categorical + continuous) well
    #   - Based on the PrivBayes/PGM framework — well studied in literature
    #   - More stable than DPGAN for small datasets
    #   - Directly cite: McKenna et al. (2021) "Winning the NIST Contest"

    for epsilon in epsilons:
        print(f"Generating synthetic data with epsilon = {epsilon}...")

        synth = Synthesizer.create(
            "mst",
            epsilon=epsilon,
            verbose=False
        )

        synth.fit(
            df_original,
            categorical_columns=categorical_cols, 
            continuous_columns=continuous_cols,
            preprocessor_eps=epsilon * 0.1
        )

        df_synthetic = synth.sample(len(df_original))

        # Restore correct types
        for col in categorical_cols:
            df_synthetic[col] = df_synthetic[col].astype(str)
        for col in continuous_cols:
            df_synthetic[col] = pd.to_numeric(df_synthetic[col], errors='coerce')
        out_path = f"{OUTPUT_DIR}combined_{dataset_name}_epsilon_{str(epsilon).replace('.', '_')}_credit_card_clients.csv"
        df_synthetic.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        print(f"  Shape: {df_synthetic.shape}")
        print(f"  Sample:\n{df_synthetic.head(3)}\n")

    print("Done. Generated datasets:")
    for epsilon in epsilons:
        print(f"  epsilon={epsilon} -> combined_{dataset_name}_epsilon_{str(epsilon).replace('.', '_')}_credit_card_clients.csv")