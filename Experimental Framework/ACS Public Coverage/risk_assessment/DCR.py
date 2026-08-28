import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

TARGET_COL = 'PUBCOV'
NUMERIC_COLS = ['PINCP']
DATASET_NAME = "ACS Public Coverage"

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
    
    # save as compressed CSV as the DCR values can be large and take up a lot of space
    record_level_df = pd.DataFrame({
        "Dataset": dataset_name,
        "Configuration": configuration_name,
        "Run": run_id,
        "Seed": seed,
        "Protected_Record_Index": np.arange(len(dcr_values)),
        "Closest_Original_Index": indices.flatten(),
        "DCR": np.round(dcr_values, 6)
    })
    
    output_path_gz = output_path.replace(".csv", ".csv.gz")
    record_level_df.to_csv(output_path_gz, index=False, compression="gzip")

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

seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

df_train_real = pd.read_csv('../datasets/folktables_public_coverage_train.csv')
df_test_real = pd.read_csv('../datasets/folktables_public_coverage_test.csv')

selected_configs = {
    "No anonymization (Baseline)": None, 
    "ARX Public Coverage, k = 3": pd.read_csv('../datasets/ARX_acs_public_coverage_k3_cleaned.csv'),
    "ARX Public Coverage, k = 5": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_cleaned.csv'),
    "ARX Public Coverage, k = 10": pd.read_csv('../datasets/ARX_acs_public_coverage_k10_cleaned.csv'),
    "ARX Public Coverage, k = 15": pd.read_csv('../datasets/ARX_acs_public_coverage_k15_cleaned.csv'),
    "ARX Public Coverage, k = 5, l = 3": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_l3_cleaned.csv'),
    "ARX Public Coverage, k = 5, l = 5": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_l5_cleaned.csv'),
    "ARX Public Coverage, k = 5, t = 0.3": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_t0.3_cleaned.csv'),
    "ARX Public Coverage, k = 5, t = 0.15": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_t0.15_cleaned.csv'),

    "DP Public Coverage, epsilon = 10.0 (r1)": pd.read_csv('../datasets/DP_epsilon_10_0_r1_public_coverage.csv'),
    "DP Public Coverage, epsilon = 10.0 (r2)": pd.read_csv('../datasets/DP_epsilon_10_0_r2_public_coverage.csv'),
    "DP Public Coverage, epsilon = 10.0 (r3)": pd.read_csv('../datasets/DP_epsilon_10_0_r3_public_coverage.csv'),
    "DP Public Coverage, epsilon = 5.0 (r1)":  pd.read_csv('../datasets/DP_epsilon_5_0_r1_public_coverage.csv'),
    "DP Public Coverage, epsilon = 5.0 (r2)":  pd.read_csv('../datasets/DP_epsilon_5_0_r2_public_coverage.csv'),
    "DP Public Coverage, epsilon = 5.0 (r3)":  pd.read_csv('../datasets/DP_epsilon_5_0_r3_public_coverage.csv'),
    "DP Public Coverage, epsilon = 3.0 (r1)":  pd.read_csv('../datasets/DP_epsilon_3_0_r1_public_coverage.csv'),
    "DP Public Coverage, epsilon = 3.0 (r2)":  pd.read_csv('../datasets/DP_epsilon_3_0_r2_public_coverage.csv'),
    "DP Public Coverage, epsilon = 3.0 (r3)":  pd.read_csv('../datasets/DP_epsilon_3_0_r3_public_coverage.csv'),
    "DP Public Coverage, epsilon = 1.0 (r1)":  pd.read_csv('../datasets/DP_epsilon_1_0_r1_public_coverage.csv'),
    "DP Public Coverage, epsilon = 1.0 (r2)":  pd.read_csv('../datasets/DP_epsilon_1_0_r2_public_coverage.csv'),
    "DP Public Coverage, epsilon = 1.0 (r3)":  pd.read_csv('../datasets/DP_epsilon_1_0_r3_public_coverage.csv'),
    "DP Public Coverage, epsilon = 0.5 (r1)":  pd.read_csv('../datasets/DP_epsilon_0_5_r1_public_coverage.csv'),
    "DP Public Coverage, epsilon = 0.5 (r2)":  pd.read_csv('../datasets/DP_epsilon_0_5_r2_public_coverage.csv'),
    "DP Public Coverage, epsilon = 0.5 (r3)":  pd.read_csv('../datasets/DP_epsilon_0_5_r3_public_coverage.csv'),
    "DP Public Coverage, epsilon = 0.1 (r1)":  pd.read_csv('../datasets/DP_epsilon_0_1_r1_public_coverage.csv'),
    "DP Public Coverage, epsilon = 0.1 (r2)":  pd.read_csv('../datasets/DP_epsilon_0_1_r2_public_coverage.csv'),
    "DP Public Coverage, epsilon = 0.1 (r3)":  pd.read_csv('../datasets/DP_epsilon_0_1_r3_public_coverage.csv'),
    
    "Combined Public Coverage, k = 3 + epsilon = 3.0 (r1)": pd.read_csv('../datasets/combined_k=3_epsilon_3_0_r1_public_coverage.csv'),
    "Combined Public Coverage, k = 3 + epsilon = 3.0 (r2)": pd.read_csv('../datasets/combined_k=3_epsilon_3_0_r2_public_coverage.csv'),
    "Combined Public Coverage, k = 3 + epsilon = 3.0 (r3)": pd.read_csv('../datasets/combined_k=3_epsilon_3_0_r3_public_coverage.csv'),
    "Combined Public Coverage, k = 3 + epsilon = 1.0 (r1)": pd.read_csv('../datasets/combined_k=3_epsilon_1_0_r1_public_coverage.csv'),
    "Combined Public Coverage, k = 3 + epsilon = 1.0 (r2)": pd.read_csv('../datasets/combined_k=3_epsilon_1_0_r2_public_coverage.csv'),
    "Combined Public Coverage, k = 3 + epsilon = 1.0 (r3)": pd.read_csv('../datasets/combined_k=3_epsilon_1_0_r3_public_coverage.csv'),
    "Combined Public Coverage, k = 3 + epsilon = 0.5 (r1)": pd.read_csv('../datasets/combined_k=3_epsilon_0_5_r1_public_coverage.csv'),
    "Combined Public Coverage, k = 3 + epsilon = 0.5 (r2)": pd.read_csv('../datasets/combined_k=3_epsilon_0_5_r2_public_coverage.csv'),
    "Combined Public Coverage, k = 3 + epsilon = 0.5 (r3)": pd.read_csv('../datasets/combined_k=3_epsilon_0_5_r3_public_coverage.csv'),
    "Combined Public Coverage, k = 5 + epsilon = 3.0 (r1)": pd.read_csv('../datasets/combined_k=5_epsilon_3_0_r1_public_coverage.csv'),
    "Combined Public Coverage, k = 5 + epsilon = 3.0 (r2)": pd.read_csv('../datasets/combined_k=5_epsilon_3_0_r2_public_coverage.csv'),
    "Combined Public Coverage, k = 5 + epsilon = 3.0 (r3)": pd.read_csv('../datasets/combined_k=5_epsilon_3_0_r3_public_coverage.csv'),
    "Combined Public Coverage, k = 5 + epsilon = 1.0 (r1)": pd.read_csv('../datasets/combined_k=5_epsilon_1_0_r1_public_coverage.csv'),
    "Combined Public Coverage, k = 5 + epsilon = 1.0 (r2)": pd.read_csv('../datasets/combined_k=5_epsilon_1_0_r2_public_coverage.csv'),
    "Combined Public Coverage, k = 5 + epsilon = 1.0 (r3)": pd.read_csv('../datasets/combined_k=5_epsilon_1_0_r3_public_coverage.csv'),
    "Combined Public Coverage, k = 5 + epsilon = 0.5 (r1)": pd.read_csv('../datasets/combined_k=5_epsilon_0_5_r1_public_coverage.csv'),
    "Combined Public Coverage, k = 5 + epsilon = 0.5 (r2)": pd.read_csv('../datasets/combined_k=5_epsilon_0_5_r2_public_coverage.csv'),
    "Combined Public Coverage, k = 5 + epsilon = 0.5 (r3)": pd.read_csv('../datasets/combined_k=5_epsilon_0_5_r3_public_coverage.csv'),
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
        output_path = f"../results/dcr/record_level/public_coverage_{safe_name}_run_{run_id}.csv"

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
summary_df.to_csv("../results/dcr/dcr_results_public_coverage_all_runs.csv", index=False)

agg_summary = summary_df.groupby("Configuration").agg(["mean", "std"], numeric_only=True)
agg_summary.to_csv("../results/dcr/dcr_results_public_coverage_summary.csv")

print("\nDCR Evaluation Complete!")