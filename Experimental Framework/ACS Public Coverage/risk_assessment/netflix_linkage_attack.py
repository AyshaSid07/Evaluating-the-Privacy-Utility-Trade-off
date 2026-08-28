from time import time
import pandas as pd
import numpy as np
from math import log
from concurrent.futures import ProcessPoolExecutor
import os

DATASET_NAME = "ACS Public Coverage"
QI_COLS = ["AGEP", "SCHL", "SEX", "MAR", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY", "ESR"]
seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
n_attacker_records = 1385 # 1% of the dataset size

os.makedirs("../results/linkage/", exist_ok=True)

def calculate_weights(df_protected, quasi_identifiers):
    weights = {}
    total_records = len(df_protected)
    for col in quasi_identifiers:
        weights[col] = {}
        cleaned_col = df_protected[col].astype(str).str.strip().str.lower()
        value_counts = cleaned_col.value_counts()
        
        for value, count in value_counts.items():
            weights[col][value] = log(total_records / count)
            
    return weights

def is_match(val_attacker, val_protected):
    str_a = str(val_attacker).strip().lower()
    str_p = str(val_protected).strip().lower()
    
    if str_a == str_p:
        return True
        
    if str_p == '*':
        return True 
        
    if str_p.startswith('[') and str_p.endswith('['):
        bounds = str_p[1:-1].split(',')
        try:
            lower = float(bounds[0].strip())
            upper = float(bounds[1].strip())
            val_a_float = float(val_attacker)
            if lower <= val_a_float < upper:
                return True
        except:
            return False

    return False

def attack_single_record(args):
    attacker_idx, attacker_row, df_protected, quasi_identifiers, weights = args

    best_score = -1.0
    best_match_index = -1

    for target_idx, protected_row in df_protected.iterrows():
        score = 0.0

        for qi in quasi_identifiers:
            val_a = attacker_row[qi]
            val_p = protected_row[qi]

            if pd.isna(val_a) or pd.isna(val_p):
                continue

            if is_match(val_a, val_p):
                str_p = str(val_p).strip().lower()
                score += weights[qi].get(str_p, 0.0)

        if score > best_score:
            best_score = score
            best_match_index = target_idx

    return attacker_idx, best_match_index

def perform_attack_parallel(df_protected, df_attacker, quasi_identifiers, weights, n_workers=16):
    tasks = [
        (attacker_idx, attacker_row, df_protected, quasi_identifiers, weights)
        for attacker_idx, attacker_row in df_attacker.iterrows()
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in executor.map(attack_single_record, tasks, chunksize=10):
            results.append(result)

    return results
        
def evaluate_attack(results, df_attacker, df_protected, defense_name, total_targets):
    correct_links = 0
    
    for attacker_idx, predicted_idx in results:
        if predicted_idx != -1 and predicted_idx in df_protected.index:
            true_attacker_id = df_attacker.loc[attacker_idx, 'Linkage_Index']
            guessed_protected_id = df_protected.loc[predicted_idx, 'Linkage_Index']
            
            if true_attacker_id == guessed_protected_id:
                correct_links += 1
                
    hit_precision = (correct_links / total_targets) * 100.0
    print(f"Defense Method: {defense_name} - Hit Precision: {hit_precision:.2f}% ({correct_links}/{total_targets} correct links)")
    return hit_precision

if __name__ == "__main__":
    df_original = pd.read_csv('../datasets/folktables_public_coverage_train.csv')
    df_original['Linkage_Index'] = df_original.index

    datasets_to_test = {
        "No anonymization (Baseline)": df_original, 
        "ARX Public Coverage, k = 3": pd.read_csv('../datasets/ARX_acs_public_coverage_k3_cleaned.csv'),
        "ARX Public Coverage, k = 5": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_cleaned.csv'),
        "ARX Public Coverage, k = 10": pd.read_csv('../datasets/ARX_acs_public_coverage_k10_cleaned.csv'),
        "ARX Public Coverage, k = 15": pd.read_csv('../datasets/ARX_acs_public_coverage_k15_cleaned.csv'),
        "ARX Public Coverage, k = 5, l = 3": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_l3_cleaned.csv'),
        "ARX Public Coverage, k = 5, l = 5": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_l5_cleaned.csv'),
        "ARX Public Coverage, k = 5, t = 0.3": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_t0.3_cleaned.csv'),
        "ARX Public Coverage, k = 5, t = 0.15": pd.read_csv('../datasets/ARX_acs_public_coverage_k5_t0.15_cleaned.csv'),
    }

    all_summaries = []

    for run_id, seed in enumerate(seeds, start=1):
        print(f"\n==============================")
        print(f"Starting linkage attack run {run_id} with seed {seed}")
        print(f"==============================")
        start = time()

        # K3 Fix: Sample 1% targets from original training data and KEEP them as truth
        df_attacker = df_original.sample(n=n_attacker_records, random_state=seed).copy()

        for defense_name, df_protected_raw in datasets_to_test.items():
            print(f"\nEvaluating defense: {defense_name}")

            df_protected = df_protected_raw.copy()

            if "Linkage_Index" not in df_protected.columns:
                df_protected["Linkage_Index"] = df_protected.index

            weights = calculate_weights(df_protected, QI_COLS)
            
            results = perform_attack_parallel(
                df_protected=df_protected,
                df_attacker=df_attacker,
                quasi_identifiers=QI_COLS,
                weights=weights,
                n_workers=32
            )
            
            hit_precision = evaluate_attack(
                results=results,
                df_attacker=df_attacker,
                df_protected=df_protected,
                defense_name=defense_name,
                total_targets=n_attacker_records
            )

            all_summaries.append({
                "Run": run_id,
                "Seed": seed,
                "Dataset": defense_name,
                "Hit_Precision": hit_precision
            })

        end = time()
        print(f"\nRun {run_id} execution time: {end - start:.2f} seconds")

        pd.DataFrame(all_summaries).to_csv(
            "../results/linkage/linkage_attack_results_public_coverage_all_runs_partial.csv",
            index=False
        )

    results_df = pd.DataFrame(all_summaries)
    results_df.to_csv("../results/linkage/linkage_attack_results_public_coverage_all_runs.csv", index=False)

    summary_df = (
        results_df
        .groupby("Dataset")
        .agg({"Hit_Precision": ["mean", "std"]})
    )
    summary_df.columns = ["_".join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.reset_index()
    summary_df.to_csv("../results/linkage/linkage_attack_results_public_coverage_summary.csv", index=False)

    print("\n=== Linkage Evaluation Complete ===")