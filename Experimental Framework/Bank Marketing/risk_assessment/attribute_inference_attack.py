import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from art.estimators.classification import SklearnClassifier
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox, AttributeInferenceBaseline

PREDICTION_TARGET = 'y'
SENSITIVE_ATTR    = 'housing'
# Variables considered quasi-identifiers by an adversary
QI_COLS = ["age", "job", "marital", "education"]
# Duration dropped as it's a future-peeking variable not available at prediction time
NUMERIC_COLS = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
DATASET_NAME = "Bank Marketing"

seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

os.makedirs("../results/aia/", exist_ok=True)

def convert_val(val):
    val = str(val).strip()
    if val == '*': return 0.0
    if val.startswith('[') and val.endswith('['):
        parts = val[1:-1].split(',')
        try:
            return (float(parts[0]) + float(parts[1])) / 2.0
        except:
            return 0.0
    try:
        return float(val)
    except:
        return 0.0

def preprocess(df, preprocessor=None, fit=False, qi_only=False):
    df_clean = df.copy()
    
    # Drop Linkage_Index and duration to match TSTR logic
    for col in ["index", "Linkage_Index", "duration"]: 
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])

    df_clean = df_clean[df_clean[SENSITIVE_ATTR].astype(str) != '*'].copy()
    df_clean = df_clean[df_clean[PREDICTION_TARGET].astype(str) != '*'].copy()
    df_clean = df_clean.dropna(subset=[PREDICTION_TARGET])

    if len(df_clean) == 0:
        return None, None, None, preprocessor

    # Target Mapping
    y_raw = df_clean[PREDICTION_TARGET].astype(str).str.strip().str.lower()
    target_map = {'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0, 'true': 1, 'false': 0}
    y = y_raw.map(target_map).fillna(0).astype(int).values
    
    # Sensitive Mapping (housing)
    sens_raw = df_clean[SENSITIVE_ATTR].astype(str).str.strip().str.lower()
    sens_map = {'no': 0, 'yes': 1, 'unknown': 2}
    sens = sens_raw.map(sens_map).fillna(3).astype(int).values
    
    X_raw = df_clean.drop(columns=[PREDICTION_TARGET, SENSITIVE_ATTR]).copy()
    
    if qi_only:
        cols_to_keep = [c for c in QI_COLS if c in X_raw.columns]
        X_raw = X_raw[cols_to_keep].copy()
    
    num_cols_present = [c for c in NUMERIC_COLS if c in X_raw.columns]
    cat_cols = [c for c in X_raw.columns if c not in num_cols_present]

    for col in num_cols_present:
        X_raw[col] = X_raw[col].apply(convert_val)
    
    for col in cat_cols:
        X_raw[col] = X_raw[col].astype(str)

    if fit:
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', 'passthrough', num_cols_present)
        ])
        X = preprocessor.fit_transform(X_raw)
    else:
        X = preprocessor.transform(X_raw)

    return X, y, sens, preprocessor


# Load strictly partitioned offline datasets to avoid data leakage
df_train_real_raw = pd.read_csv('../datasets/bank_marketing_train.csv', low_memory=False)
df_test_real_raw = pd.read_csv('../datasets/bank_marketing_test.csv', low_memory=False)

df_train_real_raw.columns = df_train_real_raw.columns.str.strip()
df_test_real_raw.columns = df_test_real_raw.columns.str.strip()

majority_class_rate = df_test_real_raw[SENSITIVE_ATTR].value_counts(normalize=True).max()
print(f"Majority-Class Rate (Test Set): {majority_class_rate:.4f}")

# Pointing to _cleaned.csv to ensure fully suppressed ARX rows are excluded
selected_configs = {
    "No anonymization (Baseline)": df_train_real_raw, 
    "ARX Bank Marketing, k = 3": pd.read_csv('../datasets/ARX_bank_marketing_k3_cleaned.csv'),
    "ARX Bank Marketing, k = 5": pd.read_csv('../datasets/ARX_bank_marketing_k5_cleaned.csv'),
    "ARX Bank Marketing, k = 10": pd.read_csv('../datasets/ARX_bank_marketing_k10_cleaned.csv'),
    "ARX Bank Marketing, k = 15": pd.read_csv('../datasets/ARX_bank_marketing_k15_cleaned.csv'),
    "ARX Bank Marketing, k = 5, l = 2": pd.read_csv('../datasets/ARX_bank_marketing_k5_l2_cleaned.csv'),
    "ARX Bank Marketing, k = 5, l = 3": pd.read_csv('../datasets/ARX_bank_marketing_k5_l3_cleaned.csv'),
    "ARX Bank Marketing, k = 5, t = 0.3": pd.read_csv('../datasets/ARX_bank_marketing_k5_t0.3_cleaned.csv'),
    "ARX Bank Marketing, k = 5, t = 0.15": pd.read_csv('../datasets/ARX_bank_marketing_k5_t0.15_cleaned.csv'),
    
    "DP Bank Marketing, epsilon = 10.0 (r1)": pd.read_csv('../datasets/DP_epsilon_10_0_r1_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 10.0 (r2)": pd.read_csv('../datasets/DP_epsilon_10_0_r2_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 10.0 (r3)": pd.read_csv('../datasets/DP_epsilon_10_0_r3_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 5.0 (r1)":  pd.read_csv('../datasets/DP_epsilon_5_0_r1_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 5.0 (r2)":  pd.read_csv('../datasets/DP_epsilon_5_0_r2_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 5.0 (r3)":  pd.read_csv('../datasets/DP_epsilon_5_0_r3_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 3.0 (r1)":  pd.read_csv('../datasets/DP_epsilon_3_0_r1_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 3.0 (r2)":  pd.read_csv('../datasets/DP_epsilon_3_0_r2_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 3.0 (r3)":  pd.read_csv('../datasets/DP_epsilon_3_0_r3_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 1.0 (r1)":  pd.read_csv('../datasets/DP_epsilon_1_0_r1_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 1.0 (r2)":  pd.read_csv('../datasets/DP_epsilon_1_0_r2_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 1.0 (r3)":  pd.read_csv('../datasets/DP_epsilon_1_0_r3_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 0.5 (r1)":  pd.read_csv('../datasets/DP_epsilon_0_5_r1_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 0.5 (r2)":  pd.read_csv('../datasets/DP_epsilon_0_5_r2_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 0.5 (r3)":  pd.read_csv('../datasets/DP_epsilon_0_5_r3_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 0.1 (r1)":  pd.read_csv('../datasets/DP_epsilon_0_1_r1_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 0.1 (r2)":  pd.read_csv('../datasets/DP_epsilon_0_1_r2_bank_marketing.csv'),
    "DP Bank Marketing, epsilon = 0.1 (r3)":  pd.read_csv('../datasets/DP_epsilon_0_1_r3_bank_marketing.csv'),
    
    "Combined Bank Marketing, k = 3 + epsilon = 3.0 (r1)": pd.read_csv('../datasets/combined_k=3_epsilon_3_0_r1_bank_marketing.csv'),
    "Combined Bank Marketing, k = 3 + epsilon = 3.0 (r2)": pd.read_csv('../datasets/combined_k=3_epsilon_3_0_r2_bank_marketing.csv'),
    "Combined Bank Marketing, k = 3 + epsilon = 3.0 (r3)": pd.read_csv('../datasets/combined_k=3_epsilon_3_0_r3_bank_marketing.csv'),
    "Combined Bank Marketing, k = 3 + epsilon = 1.0 (r1)": pd.read_csv('../datasets/combined_k=3_epsilon_1_0_r1_bank_marketing.csv'),
    "Combined Bank Marketing, k = 3 + epsilon = 1.0 (r2)": pd.read_csv('../datasets/combined_k=3_epsilon_1_0_r2_bank_marketing.csv'),
    "Combined Bank Marketing, k = 3 + epsilon = 1.0 (r3)": pd.read_csv('../datasets/combined_k=3_epsilon_1_0_r3_bank_marketing.csv'),
    "Combined Bank Marketing, k = 3 + epsilon = 0.5 (r1)": pd.read_csv('../datasets/combined_k=3_epsilon_0_5_r1_bank_marketing.csv'),
    "Combined Bank Marketing, k = 3 + epsilon = 0.5 (r2)": pd.read_csv('../datasets/combined_k=3_epsilon_0_5_r2_bank_marketing.csv'),
    "Combined Bank Marketing, k = 3 + epsilon = 0.5 (r3)": pd.read_csv('../datasets/combined_k=3_epsilon_0_5_r3_bank_marketing.csv'),
    "Combined Bank Marketing, k = 5 + epsilon = 3.0 (r1)": pd.read_csv('../datasets/combined_k=5_epsilon_3_0_r1_bank_marketing.csv'),
    "Combined Bank Marketing, k = 5 + epsilon = 3.0 (r2)": pd.read_csv('../datasets/combined_k=5_epsilon_3_0_r2_bank_marketing.csv'),
    "Combined Bank Marketing, k = 5 + epsilon = 3.0 (r3)": pd.read_csv('../datasets/combined_k=5_epsilon_3_0_r3_bank_marketing.csv'),
    "Combined Bank Marketing, k = 5 + epsilon = 1.0 (r1)": pd.read_csv('../datasets/combined_k=5_epsilon_1_0_r1_bank_marketing.csv'),
    "Combined Bank Marketing, k = 5 + epsilon = 1.0 (r2)": pd.read_csv('../datasets/combined_k=5_epsilon_1_0_r2_bank_marketing.csv'),
    "Combined Bank Marketing, k = 5 + epsilon = 1.0 (r3)": pd.read_csv('../datasets/combined_k=5_epsilon_1_0_r3_bank_marketing.csv'),
    "Combined Bank Marketing, k = 5 + epsilon = 0.5 (r1)": pd.read_csv('../datasets/combined_k=5_epsilon_0_5_r1_bank_marketing.csv'),
    "Combined Bank Marketing, k = 5 + epsilon = 0.5 (r2)": pd.read_csv('../datasets/combined_k=5_epsilon_0_5_r2_bank_marketing.csv'),
    "Combined Bank Marketing, k = 5 + epsilon = 0.5 (r3)": pd.read_csv('../datasets/combined_k=5_epsilon_0_5_r3_bank_marketing.csv'),
}

all_results = []

for seed in seeds:
    run_id = seed - 100
    print(f"\n==============================\nStarting AIA run {run_id} (Seed {seed})\n==============================", flush=True)

    # 1. Target models (Match target model dimensions to adversary feature space to avoid imputation artifacts)
    
    # Model A: Full Features
    X_train_real_full, y_train_real, sens_train_real, prep_full = preprocess(df_train_real_raw, fit=True, qi_only=False)
    X_test_real_full, y_test_real, sens_test_real, _ = preprocess(df_test_real_raw, preprocessor=prep_full, qi_only=False)

    X_train_with_sens_full = np.column_stack((sens_train_real, X_train_real_full))
    target_model_full = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=seed, n_jobs=-1)
    target_model_full.fit(X_train_with_sens_full, y_train_real)
    art_classifier_full = SklearnClassifier(model=target_model_full)

    X_test_with_sens_full = np.column_stack((sens_test_real, X_test_real_full))
    preds_test_full = np.array([np.argmax(arr) for arr in art_classifier_full.predict(X_test_with_sens_full)]).reshape(-1, 1)

    # Model B: QI-only Features
    X_train_real_qi, _, sens_train_real_qi, prep_qi = preprocess(df_train_real_raw, fit=True, qi_only=True)
    X_test_real_qi, _, sens_test_real_qi, _ = preprocess(df_test_real_raw, preprocessor=prep_qi, qi_only=True)
    
    X_train_with_sens_qi = np.column_stack((sens_train_real_qi, X_train_real_qi))
    target_model_qi = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=seed, n_jobs=-1)
    target_model_qi.fit(X_train_with_sens_qi, y_train_real)
    art_classifier_qi = SklearnClassifier(model=target_model_qi)
    
    X_test_with_sens_qi = np.column_stack((sens_test_real_qi, X_test_real_qi))
    preds_test_qi = np.array([np.argmax(arr) for arr in art_classifier_qi.predict(X_test_with_sens_qi)]).reshape(-1, 1)

    for name, df_adv_original in selected_configs.items():
        print(f"\n--- {name} ---", flush=True)
        
        # 2. Attack modes: Full Features vs Quasi-Identifiers only
        for features_mode in ["Full", "QI-only"]:
            is_qi = (features_mode == "QI-only")
            
            # Select correct preprocessor and target model based on mode
            current_prep = prep_qi if is_qi else prep_full
            X_test_current = X_test_real_qi if is_qi else X_test_real_full
            current_art_classifier = art_classifier_qi if is_qi else art_classifier_full
            current_preds_test = preds_test_qi if is_qi else preds_test_full

            X_adv, y_adv, sens_adv, _ = preprocess(df_adv_original, preprocessor=current_prep, qi_only=is_qi)

            if X_adv is None or len(X_adv) == 0:
                print(f"Skipping {name} ({features_mode}): no valid adversary records.", flush=True)
                continue

            X_adv_with_sens = np.column_stack((sens_adv, X_adv))
            possible_values = np.unique(sens_adv).tolist()
            actual_flat = np.around(sens_test_real, decimals=8).flatten()

            # Black-Box Attack
            attack_rf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=seed, n_jobs=-1)
            art_attack_model = SklearnClassifier(model=attack_rf)

            attack = AttributeInferenceBlackBox(
                estimator=current_art_classifier,
                attack_model=art_attack_model,
                attack_feature=0,
                is_continuous=False
            )
            attack.fit(x=X_adv_with_sens)
            inferred_bb = attack.infer(x=X_test_current, pred=current_preds_test, values=possible_values)
            inferred_bb_flat = inferred_bb.flatten()

            bb_acc = accuracy_score(actual_flat, inferred_bb_flat)
            bb_balanced_acc = balanced_accuracy_score(actual_flat, inferred_bb_flat)
            bb_precision = precision_score(actual_flat, inferred_bb_flat, average="macro", zero_division=0)
            bb_recall = recall_score(actual_flat, inferred_bb_flat, average="macro", zero_division=0)
            bb_f1 = f1_score(actual_flat, inferred_bb_flat, average="macro", zero_division=0)
            
            bb_f1_per_class = f1_score(actual_flat, inferred_bb_flat, average=None, zero_division=0)
            bb_recall_per_class = recall_score(actual_flat, inferred_bb_flat, average=None, zero_division=0)

            # Baseline Attack (Implemented per Elad's request in E5)
            baseline_attack = AttributeInferenceBaseline(attack_feature=0, is_continuous=False)
            baseline_attack.fit(x=X_adv_with_sens)
            inferred_base = baseline_attack.infer(x=X_test_current, values=possible_values)
            inferred_base_flat = inferred_base.flatten()
            
            base_acc = accuracy_score(actual_flat, inferred_base_flat)
            base_balanced_acc = balanced_accuracy_score(actual_flat, inferred_base_flat)
            base_precision = precision_score(actual_flat, inferred_base_flat, average="macro", zero_division=0)
            base_recall = recall_score(actual_flat, inferred_base_flat, average="macro", zero_division=0)
            base_f1 = f1_score(actual_flat, inferred_base_flat, average="macro", zero_division=0)
            base_f1_per_class = f1_score(actual_flat, inferred_base_flat, average=None, zero_division=0)
            base_recall_per_class = recall_score(actual_flat, inferred_base_flat, average=None, zero_division=0)
            print(f"  [{features_mode}] BlackBox Bal-Acc: {bb_balanced_acc:.4f} | Baseline Bal-Acc: {base_balanced_acc:.4f}")

            all_results.append({
                "Run": run_id,
                "Seed": seed,
                "Dataset": name,
                "Features_Mode": features_mode,
                "Majority_Class_Rate": majority_class_rate,
                
                "BlackBox_Accuracy": bb_acc,
                "BlackBox_Balanced_Accuracy": bb_balanced_acc,
                "BlackBox_Precision": bb_precision,
                "BlackBox_Recall": bb_recall,
                "BlackBox_F1_Macro": bb_f1,
                "BlackBox_F1_Per_Class": str(bb_f1_per_class.tolist()),
                "BlackBox_Recall_Per_Class": str(bb_recall_per_class.tolist()),
                
                "Baseline_Accuracy": base_acc,
                "Baseline_Balanced_Accuracy": base_balanced_acc,
                "Baseline_Precision": base_precision,
                "Baseline_Recall": base_recall,
                "Baseline_F1_Macro": base_f1,
                "Baseline_F1_Per_Class": str(base_f1_per_class.tolist()),
                "Baseline_Recall_Per_Class": str(base_recall_per_class.tolist())
            })

            pd.DataFrame(all_results).to_csv("../results/aia/bank_marketing_aia_results_all_runs_partial.csv", index=False)


results_df = pd.DataFrame(all_results)
results_df.to_csv("../results/aia/bank_marketing_aia_results_all_runs.csv", index=False)

summary_df = (
    results_df
    .groupby(["Dataset", "Features_Mode"])
    .agg({
        "Majority_Class_Rate": ["mean"],
        "BlackBox_Accuracy": ["mean", "std"],
        "BlackBox_Balanced_Accuracy": ["mean", "std"],
        "BlackBox_F1_Macro": ["mean", "std"],
        "Baseline_Accuracy": ["mean", "std"],
        "Baseline_Balanced_Accuracy": ["mean", "std"],
        "Baseline_F1_Macro": ["mean", "std"]
    })
)

summary_df.columns = ["_".join(col).strip() for col in summary_df.columns.values]
summary_df = summary_df.reset_index()
summary_df.to_csv("../results/aia/bank_marketing_aia_results_summary.csv", index=False)

print("\n=== AIA Evaluation Complete ===")