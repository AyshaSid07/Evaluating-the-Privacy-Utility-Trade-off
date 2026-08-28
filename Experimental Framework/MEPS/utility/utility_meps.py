import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score


# ==============================
# Configuration
# ==============================
TARGET_COLUMN = "UTILIZATION"
seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
NUMERIC_COLS = ["AGE", "PCS42", "MCS42", "K6SUM42", "PHQ242"]

# Ensure output directory exists
os.makedirs("../results/utility/", exist_ok=True)

# ==============================
# Helper functions
# ==============================
def convert_val(val):
    """
    Converts ARX interval strings (e.g. '[30, 40[') to their numerical midpoint.
    Suppressed values ('*') and parsing failures default to 0.0.
    Note for thesis: This imputation represents a pragmatic ML pipeline approach
    where ordinal intervals are mapped back to continuous space for the model.
    """
    val = str(val).strip()
    if val == "*": 
        return 0.0
    if val.startswith("[") and val.endswith("["):
        parts = val[1:-1].split(",")
        try: 
            return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception: 
            return 0.0
    try: 
        return float(val)
    except Exception: 
        return 0.0

def prepare_features(X_train, X_test, numeric_cols, categorical_cols):
    """
    Applies numerical conversion and median imputation dynamically per split.
    Prevents data leakage by ensuring test set transformations do not rely on 
    training set statistics (other than the feature column definitions).
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in numeric_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].apply(convert_val)
            median_val = X_train[col].median()
            if pd.isna(median_val): median_val = 0.0
            X_train[col] = X_train[col].fillna(median_val)

        if col in X_test.columns:
            X_test[col] = X_test[col].apply(convert_val)
            median_val = X_test[col].median()
            if pd.isna(median_val): median_val = 0.0
            X_test[col] = X_test[col].fillna(median_val)

    for col in categorical_cols:
        if col in X_train.columns: X_train[col] = X_train[col].astype(str)
        if col in X_test.columns: X_test[col] = X_test[col].astype(str)

    return X_train, X_test


# ==============================
# Load Offline Splits (TSTR strict isolation)
# ==============================
# Load pristine training and testing sets. No dynamic splitting occurs here
# to completely eliminate the data leakage artifact previously flagged by Elad.
df_train_real = pd.read_csv("../datasets/MEPS_train.csv")
df_test_real = pd.read_csv("../datasets/MEPS_test.csv")

if "index" in df_train_real.columns:
    df_train_real = df_train_real.drop(columns=["index"])
if "index" in df_test_real.columns:
    df_test_real = df_test_real.drop(columns=["index"])

if "Linkage_Index" in df_train_real.columns:
    df_train_real = df_train_real.drop(columns=["Linkage_Index"])
if "Linkage_Index" in df_test_real.columns:
    df_test_real = df_test_real.drop(columns=["Linkage_Index"])

# Drop any rows where the target variable itself was suppressed
df_train_real = df_train_real[df_train_real[TARGET_COLUMN].astype(str) != "*"].copy()
df_test_real = df_test_real[df_test_real[TARGET_COLUMN].astype(str) != "*"].copy()

FEATURE_COLS = [c for c in df_train_real.columns if c != TARGET_COLUMN]
CATEGORICAL_COLS = [c for c in FEATURE_COLS if c not in NUMERIC_COLS]


# ==============================
# Load fixed protected datasets
# ==============================
protected_datasets = {
    "ARX MEPS, k = 3": pd.read_csv('../datasets/ARX_meps_k3_cleaned.csv'),
    "ARX MEPS, k = 5": pd.read_csv('../datasets/ARX_meps_k5_cleaned.csv'),
    "ARX MEPS, k = 10": pd.read_csv('../datasets/ARX_meps_k10_cleaned.csv'),
    "ARX MEPS, k = 15": pd.read_csv('../datasets/ARX_meps_k15_cleaned.csv'),
    "ARX MEPS, k = 5, l = 2": pd.read_csv('../datasets/ARX_meps_k5_l2_cleaned.csv'),
    "ARX MEPS, k = 5, l = 3": pd.read_csv('../datasets/ARX_meps_k5_l3_cleaned.csv'),
    "ARX MEPS, k = 5, t = 0.3": pd.read_csv('../datasets/ARX_meps_k5_t0.3_cleaned.csv'),
    "ARX MEPS, k = 5, t = 0.15": pd.read_csv('../datasets/ARX_meps_k5_t0.15_cleaned.csv'),

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

    "Combined MEPS, k = 3 + epsilon = 3.0 (r1)": pd.read_csv('../datasets/combined_k3_epsilon_3_0_r1_meps.csv'),
    "Combined MEPS, k = 3 + epsilon = 3.0 (r2)": pd.read_csv('../datasets/combined_k3_epsilon_3_0_r2_meps.csv'),
    "Combined MEPS, k = 3 + epsilon = 3.0 (r3)": pd.read_csv('../datasets/combined_k3_epsilon_3_0_r3_meps.csv'),
    "Combined MEPS, k = 3 + epsilon = 1.0 (r1)": pd.read_csv('../datasets/combined_k3_epsilon_1_0_r1_meps.csv'),
    "Combined MEPS, k = 3 + epsilon = 1.0 (r2)": pd.read_csv('../datasets/combined_k3_epsilon_1_0_r2_meps.csv'),
    "Combined MEPS, k = 3 + epsilon = 1.0 (r3)": pd.read_csv('../datasets/combined_k3_epsilon_1_0_r3_meps.csv'),
    "Combined MEPS, k = 3 + epsilon = 0.5 (r1)": pd.read_csv('../datasets/combined_k3_epsilon_0_5_r1_meps.csv'),
    "Combined MEPS, k = 3 + epsilon = 0.5 (r2)": pd.read_csv('../datasets/combined_k3_epsilon_0_5_r2_meps.csv'),
    "Combined MEPS, k = 3 + epsilon = 0.5 (r3)": pd.read_csv('../datasets/combined_k3_epsilon_0_5_r3_meps.csv'),
    "Combined MEPS, k = 5 + epsilon = 3.0 (r1)": pd.read_csv('../datasets/combined_k5_epsilon_3_0_r1_meps.csv'),
    "Combined MEPS, k = 5 + epsilon = 3.0 (r2)": pd.read_csv('../datasets/combined_k5_epsilon_3_0_r2_meps.csv'),
    "Combined MEPS, k = 5 + epsilon = 3.0 (r3)": pd.read_csv('../datasets/combined_k5_epsilon_3_0_r3_meps.csv'),
    "Combined MEPS, k = 5 + epsilon = 1.0 (r1)": pd.read_csv('../datasets/combined_k5_epsilon_1_0_r1_meps.csv'),
    "Combined MEPS, k = 5 + epsilon = 1.0 (r2)": pd.read_csv('../datasets/combined_k5_epsilon_1_0_r2_meps.csv'),
    "Combined MEPS, k = 5 + epsilon = 1.0 (r3)": pd.read_csv('../datasets/combined_k5_epsilon_1_0_r3_meps.csv'),
    "Combined MEPS, k = 5 + epsilon = 0.5 (r1)": pd.read_csv('../datasets/combined_k5_epsilon_0_5_r1_meps.csv'),
    "Combined MEPS, k = 5 + epsilon = 0.5 (r2)": pd.read_csv('../datasets/combined_k5_epsilon_0_5_r2_meps.csv'),
    "Combined MEPS, k = 5 + epsilon = 0.5 (r3)": pd.read_csv('../datasets/combined_k5_epsilon_0_5_r3_meps.csv'),
}

all_results = []

for seed in seeds:
    run_id = seed - 100
    print(f"\n==============================\nStarting utility run {run_id} with seed {seed}\n==============================", flush=True)

    # Establish real testing ground truth (Never altered or seen during training)
    y_test_real = df_test_real[TARGET_COLUMN].astype(int).values
    X_test_real = df_test_real.drop(columns=[TARGET_COLUMN])

    datasets_to_test = {
        "No anonymization (Baseline)": None,
        **protected_datasets
    }

    for name, df_protected_original in datasets_to_test.items():
        print(f"\nEvaluating: {name}", flush=True)

        # Baseline: train directly on the pristine training split
        if df_protected_original is None:
            X_train = df_train_real.drop(columns=[TARGET_COLUMN]).copy()
            y_train = df_train_real[TARGET_COLUMN].astype(int).values
        else:
            df_protected = df_protected_original.copy()
            if "index" in df_protected.columns:
                df_protected = df_protected.drop(columns=["index"])

            # Align protected training records with original training set indices.
            # This handles ARX suppression without misalignment.
            if "Linkage_Index" in df_protected.columns:
                df_protected = df_protected.set_index("Linkage_Index")
                surviving_train_indices = df_train_real.index.intersection(df_protected.index)
                df_train = df_protected.loc[surviving_train_indices].copy()
            else:
                # Fallback if Linkage_Index is missing (e.g., purely synthetic sets)
                df_train = df_protected.reset_index(drop=True).copy()

            # Discard records where the target variable was suppressed
            df_train = df_train[df_train[TARGET_COLUMN].astype(str) != "*"].copy()

            if len(df_train) == 0:
                print(f"Skipping {name}: no valid training records after filtering.", flush=True)
                continue

            # Ensure strict feature set parity before splitting
            df_train = df_train[FEATURE_COLS + [TARGET_COLUMN]]

            y_train = df_train[TARGET_COLUMN].astype(int).values
            X_train = df_train.drop(columns=[TARGET_COLUMN]).copy()

        # Copy pristine test set for current evaluation iteration
        X_test_current = X_test_real.copy()

        # Apply preprocessing (midpoint numerical imputation and formatting)
        X_train_prep, X_test_prep = prepare_features(X_train, X_test_current, NUMERIC_COLS, CATEGORICAL_COLS)

        # OHE for categoricals, passthrough for continuous/midpoints
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
                ("num", "passthrough", NUMERIC_COLS)
            ]
        )

        # Utilize balanced class weights to account for majority class dominance
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1))
        ])

        # Train on protected data
        model.fit(X_train_prep, y_train)
        
        # Predict on pristine real data
        y_pred = model.predict(X_test_prep)
        y_prob = model.predict_proba(X_test_prep)[:, 1]

        acc = accuracy_score(y_test_real, y_pred)
        bal_acc = balanced_accuracy_score(y_test_real, y_pred)
        roc_auc = roc_auc_score(y_test_real, y_prob)
        precision = precision_score(y_test_real, y_pred, zero_division=0)
        recall = recall_score(y_test_real, y_pred, zero_division=0)
        f1 = f1_score(y_test_real, y_pred, zero_division=0)

        print(f"  Acc: {acc:.4f}, Bal-Acc: {bal_acc:.4f}, ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}", flush=True)

        all_results.append({
            "Run": run_id,
            "Seed": seed,
            "Dataset": name,
            "Accuracy": acc,
            "Balanced_Accuracy": bal_acc,
            "ROC-AUC": roc_auc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

        pd.DataFrame(all_results).to_csv("../results/utility/meps_utility_results_all_runs_partial.csv", index=False)

# ==============================
# Save all results and mean/std summary
# ==============================
results_df = pd.DataFrame(all_results)
results_df.to_csv("../results/utility/meps_utility_results_all_runs.csv", index=False)

summary_df = (
    results_df
    .groupby("Dataset")
    .agg({
        "Accuracy": ["mean", "std"],
        "Balanced_Accuracy": ["mean", "std"],
        "ROC-AUC": ["mean", "std"],
        "Precision": ["mean", "std"],
        "Recall": ["mean", "std"],
        "F1-Score": ["mean", "std"]
    })
)

summary_df.columns = ["_".join(col).strip() for col in summary_df.columns.values]
summary_df = summary_df.reset_index()
summary_df.to_csv("../results/utility/meps_utility_results_summary.csv", index=False)

print("\n=== Utility Evaluation Complete ===")