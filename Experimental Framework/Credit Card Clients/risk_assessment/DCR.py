import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

TARGET_COL = 'default payment'
NUMERIC_COLS = [
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]
DATASET_NAME = "Credit Card Clients"

os.makedirs("../results/dcr/record_level/", exist_ok=True)

def convert_arx_value(val):
    val = str(val).strip()
    if val == "*":
        return 0.0
    if val.startswith("[") and val.endswith("["):
        parts = val[1:-1].split(",")
        try:
            return (float(parts[0]) + float(parts[1])) / 2.0
        except:
            return 0.0
    try:
        return float(val)
    except:
        return 0.0

def prepare_dcr_data(df, target_column, numeric_cols, common_cols):
    df = df.copy()
    df = df[[c for c in common_cols if c in df.columns]]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_arx_value)
    return df

def compute_record_level_dcr(
    df_original_train,
    df_protected,
    numeric_cols,
    categorical_cols,
    dataset_name,
    configuration_name,
    run_id,
    seed,
    output_path
):
    for col in numeric_cols:
        median_val = df_original_train[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df_original_train[col] = df_original_train[col].fillna(median_val)
        df_protected[col] = df_protected[col].fillna(median_val)

    for col in categorical_cols:
        df_original_train[col] = df_original_train[col].astype(str)
        df_protected[col] = df_protected[col].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", MinMaxScaler(), numeric_cols)
        ],
        remainder="drop"
    )

    # combine the original and protected datasets to fit the preprocessor
    combined_df = pd.concat([df_original_train, df_protected], axis=0)
    preprocessor.fit(combined_df)

    X_original = preprocessor.transform(df_original_train)
    X_protected = preprocessor.transform(df_protected)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn.fit(X_original)
    distances, indices = nn.kneighbors(X_protected)
    
    dcr_values = distances.flatten()
    
    record_level_df = pd.DataFrame({
        "Dataset": dataset_name,
        "Configuration": configuration_name,
        "Run": run_id,
        "Seed": seed,
        "Protected_Record_Index": np.arange(len(dcr_values)),
        "Closest_Original_Index": indices.flatten(),
        "DCR": dcr_values
    })
    record_level_df.to_csv(output_path, index=False)

    return {
        "Dataset": dataset_name,
        "Configuration": configuration_name,
        "Run": run_id,
        "Seed": seed,
        "mean_distance": float(np.mean(dcr_values)),
        "median_distance": float(np.median(dcr_values)),
        "min_distance": float(np.min(dcr_values)),
        "max_distance": float(np.max(dcr_values)),
        "std_distance": float(np.std(dcr_values)),
        "percentile_1st": float(np.percentile(dcr_values, 1)),
        "percentile_5th": float(np.percentile(dcr_values, 5)),
        "percentile_10th": float(np.percentile(dcr_values, 10)),
        "percentile_25th": float(np.percentile(dcr_values, 25)),
        "percentile_75th": float(np.percentile(dcr_values, 75)),
        "percentile_90th": float(np.percentile(dcr_values, 90)),
        "percentile_95th": float(np.percentile(dcr_values, 95)),
        "percentile_99th": float(np.percentile(dcr_values, 99)),
        "exact_matches_%": float(np.mean(dcr_values == 0) * 100),
        "near_matches_1pct_%": float(np.mean(dcr_values < 0.01) * 100),
        "near_matches_5pct_%": float(np.mean(dcr_values < 0.05) * 100),
        "near_matches_10pct_%": float(np.mean(dcr_values < 0.10) * 100)
    }

seeds = [101]

df_train_real = pd.read_csv('../datasets/credit_card_clients_train.csv')
df_test_real = pd.read_csv('../datasets/credit_card_clients_test.csv')

selected_configs = {
    "No anonymization (Baseline)": None, 
    "DP Credit Card, eps = 10.0 (r1)": pd.read_csv('../datasets/DP_credit_card_clients_epsilon_10_0_r1.csv'),
    "DP Credit Card, eps = 10.0 (r2)": pd.read_csv('../datasets/DP_credit_card_clients_epsilon_10_0_r2.csv'),
    "DP Credit Card, eps = 10.0 (r3)": pd.read_csv('../datasets/DP_credit_card_clients_epsilon_10_0_r3.csv'),
    "DP Credit Card, eps = 5.0 (r1)":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_5_0_r1.csv'),
    "DP Credit Card, eps = 5.0 (r2)":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_5_0_r2.csv'),
    "DP Credit Card, eps = 5.0 (r3)":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_5_0_r3.csv'),
    "DP Credit Card, eps = 3.0 (r1)":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_3_0_r1.csv'),
    "DP Credit Card, eps = 3.0 (r2)":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_3_0_r2.csv'),
    "DP Credit Card, eps = 3.0 (r3)":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_3_0_r3.csv'),
    "DP Credit Card, eps = 1.0 (r1)":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_1_0_r1.csv'),
    "DP Credit Card, eps = 1.0 (r2)":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_1_0_r2.csv'),
    "DP Credit Card, eps = 1.0 (r3)":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_1_0_r3.csv'),
    "DP Credit Card, eps = 0.5 (r1)":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_0_5_r1.csv'),
    "DP Credit Card, eps = 0.5 (r2)":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_0_5_r2.csv'),
    "DP Credit Card, eps = 0.5 (r3)":  pd.read_csv('../datasets/DP_credit_card_clients_epsilon_0_5_r3.csv'),
}

all_summaries = []

exclude_cols = ["index", "Linkage_Index", TARGET_COL]
common_cols = [c for c in df_train_real.columns if c not in exclude_cols]
numeric_present = [c for c in NUMERIC_COLS if c in common_cols]
categorical_cols = [c for c in common_cols if c not in numeric_present]

for run_id, seed in enumerate(seeds, start=1):
    print(f"\n--- Starting DCR run {run_id} (Seed {seed}) ---", flush=True)

    for config_name, df_protected in selected_configs.items():
        print(f"Evaluating: {config_name}", flush=True)

        if df_protected is None:
            protected_input = df_test_real.copy()
        else:
            protected_input = df_protected.copy()

        original_prepared = prepare_dcr_data(df_train_real, TARGET_COL, numeric_present, common_cols)
        protected_prepared = prepare_dcr_data(protected_input, TARGET_COL, numeric_present, common_cols)

        safe_name = config_name.replace(' ', '_').replace('=', '').replace('+', 'plus').replace(',', '')
        output_path = f"../results/dcr/record_level/credit_card_{safe_name}_run_{run_id}.csv"

        summary = compute_record_level_dcr(
            df_original_train=original_prepared,
            df_protected=protected_prepared,
            numeric_cols=numeric_present,
            categorical_cols=categorical_cols,
            dataset_name=DATASET_NAME,
            configuration_name=config_name,
            run_id=run_id,
            seed=seed,
            output_path=output_path
        )
        all_summaries.append(summary)

summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv("../results/dcr/dcr_results_credit_card_all_runs.csv", index=False)

agg_summary = summary_df.groupby("Configuration").agg(["mean", "std"], numeric_only=True)
agg_summary.to_csv("../results/dcr/dcr_results_credit_card_summary.csv")

print("\nDCR Evaluation Complete!")