import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

TARGET_COL = 'UTILIZATION'
NUMERIC_COLS = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PHQ242']
DATASET_NAME = "MEPS"

os.makedirs("../results/dcr/record_level/", exist_ok=True)

def convert_arx_value(val):
    """Convert ARX intervals to midpoints and suppressions to 0."""
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
    """Filter columns and convert ARX numeric intervals."""
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
    # 1. Impute missing values with training set median
    for col in numeric_cols:
        median_val = df_original_train[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df_original_train[col] = df_original_train[col].fillna(median_val)
        df_protected[col] = df_protected[col].fillna(median_val)

    for col in categorical_cols:
        df_original_train[col] = df_original_train[col].astype(str)
        df_protected[col] = df_protected[col].astype(str)

    # 2. Fit preprocessor on the original training data and transform both datasets
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

    # 3. Find nearest neighbors (DCR)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn.fit(X_original)
    distances, indices = nn.kneighbors(X_protected)
    
    dcr_values = distances.flatten()
    
    # 4. Save record-level DCR for histograms
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

    # 5. Return summary metrics for tables
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

# --- MAIN EXECUTION ---
seeds = [101]

# Load the offline pre-partitioned 80/20 data
df_train_real = pd.read_csv('../datasets/MEPS_train.csv')
df_test_real = pd.read_csv('../datasets/MEPS_test.csv')

selected_configs = {
    "No anonymization (Baseline)": None, 
    "DP MEPS, epsilon = 10.0 (r1)": pd.read_csv('../datasets/DP_epsilon_10_0_r1_meps.csv'),
    "DP MEPS, epsilon = 10.0 (r2)": pd.read_csv('../datasets/DP_epsilon_10_0_r2_meps.csv'),
    "DP MEPS, epsilon = 10.0 (r3)": pd.read_csv('../datasets/DP_epsilon_10_0_r3_meps.csv'),
    "DP MEPS, epsilon = 5.0 (r1)":  pd.read_csv('../datasets/DP_epsilon_5_0_r1_meps.csv'),
    "DP MEPS, epsilon = 5.0 (r2)":  pd.read_csv('../datasets/DP_epsilon_5_0_r2_meps.csv'),
    "DP MEPS, epsilon = 5.0 (r3)":  pd.read_csv('../datasets/DP_epsilon_5_0_r3_meps.csv'),
    "DP MEPS, epsilon = 3.0 (r1)":  pd.read_csv('../datasets/DP_epsilon_3_0_r1_meps.csv'),
    "DP MEPS, epsilon = 3.0 (r2)":  pd.read_csv('../datasets/DP_epsilon_3_0_r2_meps.csv'),
    "DP MEPS, epsilon = 3.0 (r3)":  pd.read_csv('../datasets/DP_epsilon_3_0_r3_meps.csv'),
    "DP MEPS, epsilon = 1.0 (r1)":  pd.read_csv('../datasets/DP_epsilon_1_0_r1_meps.csv'),
    "DP MEPS, epsilon = 1.0 (r2)":  pd.read_csv('../datasets/DP_epsilon_1_0_r2_meps.csv'),
    "DP MEPS, epsilon = 1.0 (r3)":  pd.read_csv('../datasets/DP_epsilon_1_0_r3_meps.csv'),
    "DP MEPS, epsilon = 0.5 (r1)":  pd.read_csv('../datasets/DP_epsilon_0_5_r1_meps.csv'),
    "DP MEPS, epsilon = 0.5 (r2)":  pd.read_csv('../datasets/DP_epsilon_0_5_r2_meps.csv'),
    "DP MEPS, epsilon = 0.5 (r3)":  pd.read_csv('../datasets/DP_epsilon_0_5_r3_meps.csv'),
}

all_summaries = []

# Exclude target and identifiers from feature matrix
exclude_cols = ["index", "Linkage_Index", TARGET_COL]
common_cols = [c for c in df_train_real.columns if c not in exclude_cols]
numeric_present = [c for c in NUMERIC_COLS if c in common_cols]
categorical_cols = [c for c in common_cols if c not in numeric_present]

for run_id, seed in enumerate(seeds, start=1):
    print(f"\n--- Starting DCR run {run_id} (Seed {seed}) ---", flush=True)

    for config_name, df_protected in selected_configs.items():
        print(f"Evaluating: {config_name}", flush=True)

        if df_protected is None:
            # Baseline uses the exact test partition
            protected_input = df_test_real.copy()
        else:
            protected_input = df_protected.copy()

        original_prepared = prepare_dcr_data(df_train_real, TARGET_COL, numeric_present, common_cols)
        protected_prepared = prepare_dcr_data(protected_input, TARGET_COL, numeric_present, common_cols)

        safe_name = config_name.replace(' ', '_').replace('=', '').replace('+', 'plus').replace(',', '')
        output_path = f"../results/dcr/record_level/meps_{safe_name}_run_{run_id}.csv"

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

# Save results
summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv("../results/dcr/dcr_results_meps_all_runs.csv", index=False)

agg_summary = summary_df.groupby("Configuration").agg(["mean", "std"], numeric_only=True)
agg_summary.to_csv("../results/dcr/dcr_results_meps_summary.csv")

print("\nDCR Evaluation Complete!")