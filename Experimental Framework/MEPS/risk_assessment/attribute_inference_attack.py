import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from art.estimators.classification import SklearnClassifier
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox

PREDICTION_TARGET = 'UTILIZATION'
SENSITIVE_ATTR    = 'INSCOV'
QI_COLS = ["AGE", "MARRY", "REGION", "SEX", "RACE", "ACTDTY", "EMPST", "FTSTU"]
NUMERIC_COLS = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PHQ242']
DATASET_NAME = "MEPS"
SUPPRESSED_CATEGORY_CODE = -1 # Distinct code for suppressed ('*') sensitive-attribute values, used only when building the target model's training data (never filtered out there).
seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
os.makedirs("../results/aia/", exist_ok=True)

def convert_val(val):
    """Converts a raw cell to a float, matching TSTR's own rules exactly
    ('*' -> 0.0, ARX intervals -> midpoint) so the target model sees
    identical numeric inputs to Experiment 4's utility model."""
    val = str(val).strip()
    if val == '*': return 0.0
    if val.startswith('[') and val.endswith('['):
        parts = val[1:-1].split(',')
        try: return (float(parts[0]) + float(parts[1])) / 2.0
        except: return np.nan
    try: return float(val)
    except: return np.nan


def clean_target(y_series):
    y_raw = y_series.astype(str).str.strip().str.lower()
    target_map = {'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0, 'true': 1, 'false': 0}
    return y_raw.map(target_map).fillna(0).astype(int).values

def get_utility_target_model(df_train_real_raw, df_test_real_raw, df_protected_original, seed, drop_sens=False):
    """Builds the target model as a clone of TSTR's own utility
    script, so it's the same object as the Experiment 4 utility model. """
    df_train_real = df_train_real_raw.copy()
    df_test_real = df_test_real_raw.copy()
    df_train_real.columns = df_train_real.columns.str.strip()
    df_test_real.columns = df_test_real.columns.str.strip()

    for drop_col in ["index", "Linkage_Index"]:
        if drop_col in df_train_real.columns: df_train_real = df_train_real.drop(columns=[drop_col])
        if drop_col in df_test_real.columns: df_test_real = df_test_real.drop(columns=[drop_col])

    df_train_real = df_train_real[df_train_real[PREDICTION_TARGET].astype(str) != "*"].copy()
    df_train_real = df_train_real.dropna(subset=[PREDICTION_TARGET])
    df_test_real = df_test_real[df_test_real[PREDICTION_TARGET].astype(str) != "*"].copy()
    df_test_real = df_test_real.dropna(subset=[PREDICTION_TARGET])

    FEATURE_COLS = [c for c in df_train_real.columns if c != PREDICTION_TARGET]

    if df_protected_original is None:
        df_train = df_train_real.copy()
    else:
        df_protected = df_protected_original.copy()
        df_protected.columns = df_protected.columns.str.strip()
        if "index" in df_protected.columns: df_protected = df_protected.drop(columns=["index"])
        if "Linkage_Index" in df_protected.columns:
            df_protected = df_protected.set_index("Linkage_Index")
            surviving_train_indices = df_train_real_raw.index.intersection(df_protected.index)
            df_train = df_protected.loc[surviving_train_indices].copy()
        else:
            df_train = df_protected.reset_index(drop=True).copy()
        df_train = df_train[df_train[PREDICTION_TARGET].astype(str) != "*"].copy()
        df_train = df_train.dropna(subset=[PREDICTION_TARGET])
        df_train = df_train[FEATURE_COLS + [PREDICTION_TARGET]]

    y_train = clean_target(df_train[PREDICTION_TARGET])
    X_train = df_train.drop(columns=[PREDICTION_TARGET]).copy()

    df_test_real = df_test_real[df_test_real[SENSITIVE_ATTR].astype(str) != '*'].copy()
    y_test = clean_target(df_test_real[PREDICTION_TARGET])
    X_test = df_test_real.drop(columns=[PREDICTION_TARGET]).copy()

    if drop_sens and SENSITIVE_ATTR in FEATURE_COLS:
        X_train = X_train.drop(columns=[SENSITIVE_ATTR])
        X_test = X_test.drop(columns=[SENSITIVE_ATTR])

    cats = [c for c in X_train.columns if c not in NUMERIC_COLS]
    nums = [c for c in X_train.columns if c in NUMERIC_COLS]

    for col in nums:
        X_train[col] = X_train[col].apply(convert_val)
        X_test[col] = X_test[col].apply(convert_val)

        med_tr = X_train[col].median()
        if pd.isna(med_tr): med_tr = 0.0
        X_train[col] = X_train[col].fillna(med_tr)

        med_te = X_test[col].median()
        if pd.isna(med_te): med_te = 0.0
        X_test[col] = X_test[col].fillna(med_te)

    for col in cats:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ("num", "passthrough", nums)
    ])
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)
    model = Pipeline([("preprocessor", prep), ("classifier", clf)])
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_test


def build_scenario_b_query_features(df_train_real_raw, df_protected_original, FEATURE_COLS, train_medians, drop_sens=False):
    """Prepares raw features to query Scenario B's fixed target model on a
    new release, without fitting anything. Uses the model's own original
    training medians, since we're querying an existing model, not fitting one."""
    if df_protected_original is None:
        df_train = df_train_real_raw.copy()
        df_train.columns = df_train.columns.str.strip()
        for c in ["index", "Linkage_Index"]:
            if c in df_train.columns: df_train = df_train.drop(columns=[c])
    else:
        df_protected = df_protected_original.copy()
        df_protected.columns = df_protected.columns.str.strip()
        if "index" in df_protected.columns: df_protected = df_protected.drop(columns=["index"])
        if "Linkage_Index" in df_protected.columns:
            df_protected = df_protected.set_index("Linkage_Index")
            surviving = df_train_real_raw.index.intersection(df_protected.index)
            df_train = df_protected.loc[surviving].copy()
        else:
            df_train = df_protected.reset_index(drop=True).copy()

    df_train = df_train[df_train[PREDICTION_TARGET].astype(str) != "*"].copy()
    df_train = df_train.dropna(subset=[PREDICTION_TARGET])
    df_train = df_train[FEATURE_COLS + [PREDICTION_TARGET]]

    X = df_train.drop(columns=[PREDICTION_TARGET]).copy()
    if drop_sens and SENSITIVE_ATTR in X.columns:
        X = X.drop(columns=[SENSITIVE_ATTR])

    nums = [c for c in X.columns if c in NUMERIC_COLS]
    cats = [c for c in X.columns if c not in NUMERIC_COLS]
    for col in nums:
        X[col] = X[col].apply(convert_val)
        X[col] = X[col].fillna(train_medians.get(col, 0.0))
    for col in cats:
        X[col] = X[col].astype(str)
    return X


def make_wrapper_estimator(stored_proba, seed, n_features):
    """Lets ART query 'the target model' without seeing its real feature
    space: pre-computes the model's true predictions, then hands ART a
    real sklearn instance whose predict_proba just replays them in row order."""
    dummy = RandomForestClassifier(n_estimators=5, random_state=seed)
    n_classes = stored_proba.shape[1]
    dummy.fit(np.zeros((n_classes, n_features)), np.arange(n_classes))  # throwaway fit, correct width
    dummy.classes_ = np.arange(n_classes)

    def fake_predict_proba(X, _stored=stored_proba):
        if X.shape[0] != len(_stored):
            raise ValueError(f"Row count mismatch: got {X.shape[0]}, stored has {len(_stored)}")
        return _stored

    dummy.predict_proba = fake_predict_proba
    return SklearnClassifier(model=dummy)



def encode_adversary_features(df, preprocessor=None, fit=False, filter_sensitive=True, df_train_real_raw=None):
    """ Builds the adversary's own feature encoding, separate from the target model's."""
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()
    if "index" in df_clean.columns:
        df_clean = df_clean.drop(columns=["index"])
    if "Linkage_Index" in df_clean.columns:
        if df_train_real_raw is not None:
            df_ref = df_train_real_raw.copy()
            df_ref.columns = df_ref.columns.str.strip()
            df_clean = df_clean.set_index("Linkage_Index")
            surviving = df_ref.index.intersection(df_clean.index)
            df_clean = df_clean.loc[surviving].reset_index(drop=True)
        else:
            df_clean = df_clean.drop(columns=["Linkage_Index"])
    if filter_sensitive:
        df_clean = df_clean[df_clean[SENSITIVE_ATTR].astype(str) != '*'].copy()
    df_clean = df_clean[df_clean[PREDICTION_TARGET].astype(str) != '*'].copy()
    df_clean = df_clean.dropna(subset=[PREDICTION_TARGET])
    if len(df_clean) == 0:
        return None, None, None, preprocessor

    y = clean_target(df_clean[PREDICTION_TARGET])
    sens_raw = df_clean[SENSITIVE_ATTR].astype(str)
    if filter_sensitive:
        sens = sens_raw.astype(float).astype(int).values
    else:
        sens_numeric = pd.to_numeric(sens_raw, errors='coerce')
        sens = sens_numeric.fillna(SUPPRESSED_CATEGORY_CODE).astype(int).values

    X_raw = df_clean.drop(columns=[PREDICTION_TARGET, SENSITIVE_ATTR]).copy()
    nums_present = [c for c in NUMERIC_COLS if c in X_raw.columns]
    cats_present = [c for c in X_raw.columns if c not in nums_present]
    for col in nums_present:
        X_raw[col] = X_raw[col].apply(lambda v: 0.0 if pd.isna(convert_val(v)) else convert_val(v))
    for col in cats_present:
        X_raw[col] = X_raw[col].astype(str)

    if fit:
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats_present),
            ('num', 'passthrough', nums_present)
        ])
        X = preprocessor.fit_transform(X_raw)
    else:
        X = preprocessor.transform(X_raw)
    return X, y, sens, preprocessor


df_train_real_raw = pd.read_csv('../datasets/MEPS_train.csv', low_memory=False)
df_test_real_raw = pd.read_csv('../datasets/MEPS_test.csv', low_memory=False)
df_train_real_raw.columns = df_train_real_raw.columns.str.strip()
df_test_real_raw.columns = df_test_real_raw.columns.str.strip()
majority_class_rate = df_test_real_raw[df_test_real_raw[SENSITIVE_ATTR].astype(str) != '*'][SENSITIVE_ATTR].astype(float).astype(int).value_counts(normalize=True).max()
print(f"Majority-Class Rate (Test Set): {majority_class_rate:.4f}")

datasets_to_test = {
    "No anonymization (Baseline)": None,
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
ablation_results = []

for seed in seeds:
    run_id = seed - 100
    print(f"\n==============================\nStarting AIA run {run_id} (Seed {seed})\n==============================", flush=True)

    num_cols_count = len([c for c in NUMERIC_COLS if c in df_train_real_raw.columns])
    df_universal = pd.concat([df_train_real_raw, df_test_real_raw] + [d for d in datasets_to_test.values() if d is not None], ignore_index=True)
    _, _, _, universal_prep = encode_adversary_features(df_universal, fit=True, filter_sensitive=False)

    feature_names = universal_prep.get_feature_names_out()
    def is_qi_feature(f):
        col_name = f.split('__')[1].split('_')[0] if '__' in f else f
        return any(col_name == qi or col_name.startswith(qi) for qi in QI_COLS)
    qi_mask = np.array([is_qi_feature(f) for f in feature_names])

    X_test_encoded, y_test, sens_test, _ = encode_adversary_features(df_test_real_raw, preprocessor=universal_prep, filter_sensitive=True)
    actual_flat = sens_test.flatten()

    model_b_report, X_train_b_report, X_test_b_report, y_test_b_report = get_utility_target_model(df_train_real_raw, df_test_real_raw, None, seed, drop_sens=False)
    y_test_pred_b = model_b_report.predict(X_test_b_report)
    tm_acc_b = accuracy_score(y_test_b_report, y_test_pred_b)
    tm_bal_acc_b = balanced_accuracy_score(y_test_b_report, y_test_pred_b)
    preds_test_b = y_test_pred_b.reshape(-1, 1)

    train_medians_b = {}
    for col in NUMERIC_COLS:
        if col in X_train_b_report.columns:
            m = X_train_b_report[col].median()
            train_medians_b[col] = 0.0 if pd.isna(m) else m

    # --- Ablation ---
    model_ablation, _, X_test_abl, y_test_abl = get_utility_target_model(df_train_real_raw, df_test_real_raw, None, seed, drop_sens=True)
    y_test_pred_abl = model_ablation.predict(X_test_abl)
    ablation_results.append({
        "Run": run_id, "Seed": seed, "Dataset": DATASET_NAME,
        "Target_Model_WithSens_BalAcc": tm_bal_acc_b,
        "Target_Model_NoSens_BalAcc": balanced_accuracy_score(y_test_abl, y_test_pred_abl),
    })

    for name, df_adv_original in datasets_to_test.items():
        print(f"\n--- {name} ---", flush=True)
        df_pass = df_adv_original if name != "No anonymization (Baseline)" else None

        model_a_report, X_train_a_report, X_test_a_report, y_test_a_report = get_utility_target_model(df_train_real_raw, df_test_real_raw, df_pass, seed, drop_sens=False)
        y_test_pred_a = model_a_report.predict(X_test_a_report)
        tm_acc_a = accuracy_score(y_test_a_report, y_test_pred_a)
        tm_bal_acc_a = balanced_accuracy_score(y_test_a_report, y_test_pred_a)
        preds_test_a = y_test_pred_a.reshape(-1, 1)

        df_to_prep = df_train_real_raw if df_pass is None else df_pass
        X_adv_encoded, y_adv, sens_adv, _ = encode_adversary_features(
            df_to_prep, preprocessor=universal_prep, filter_sensitive=True,
            df_train_real_raw=(df_train_real_raw if df_pass is not None else None)
        )
        if X_adv_encoded is None: continue
        X_adv_with_sens = np.column_stack((sens_adv, X_adv_encoded))
        possible_values = np.unique(sens_adv).tolist()

        if len(X_train_a_report) != X_adv_with_sens.shape[0]:
            print(f"  WARNING: row count mismatch between the adversary model ({X_adv_with_sens.shape[0]}) "
                  f"and target-model training rows ({len(X_train_a_report)}) for {name}; skipping.", flush=True)
            continue
        stored_proba_adv_a = model_a_report.predict_proba(X_train_a_report)
        art_tm_a = make_wrapper_estimator(stored_proba_adv_a, seed, X_adv_with_sens.shape[1])

        FEATURE_COLS_B = [c for c in df_train_real_raw.columns if c not in ["index", "Linkage_Index", PREDICTION_TARGET]]
        X_adv_query_b = build_scenario_b_query_features(df_train_real_raw, df_pass, FEATURE_COLS_B, train_medians_b, drop_sens=False)
        if len(X_adv_query_b) != X_adv_with_sens.shape[0]:
            print(f"  WARNING: row count mismatch for scenario B query on {name}; skipping.", flush=True)
            continue
        stored_proba_adv_b = model_b_report.predict_proba(X_adv_query_b)
        art_tm_b = make_wrapper_estimator(stored_proba_adv_b, seed, X_adv_with_sens.shape[1])

        for features_mode in ["Full", "QI-only"]:
            is_qi = (features_mode == "QI-only")

            if is_qi:
                qi_selector = FunctionTransformer(
                    func=lambda X, q_mask, n_orig: np.hstack((X[:, :n_orig][:, q_mask], X[:, n_orig:])),
                    kw_args={'q_mask': qi_mask, 'n_orig': len(qi_mask)}
                )
                attack_rf_base = Pipeline([
                    ('qi_sel', qi_selector),
                    ('rf', RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=seed, n_jobs=-1))
                ])
                X_adv_base = X_adv_encoded[:, qi_mask]
                X_test_base = X_test_encoded[:, qi_mask]
            else:
                attack_rf_base = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=seed, n_jobs=-1)
                X_adv_base = X_adv_encoded
                X_test_base = X_test_encoded

            attack_a = AttributeInferenceBlackBox(estimator=art_tm_a, attack_model=SklearnClassifier(model=attack_rf_base), attack_feature=0, is_continuous=False)
            attack_a.fit(x=X_adv_with_sens)
            inferred_a = attack_a.infer(x=X_test_encoded, pred=preds_test_a, values=possible_values).flatten()
            res_a = {
                "BlackBox_Accuracy": accuracy_score(actual_flat, inferred_a),
                "BlackBox_Balanced_Accuracy": balanced_accuracy_score(actual_flat, inferred_a),
                "BlackBox_Precision": precision_score(actual_flat, inferred_a, average="macro", zero_division=0),
                "BlackBox_Recall": recall_score(actual_flat, inferred_a, average="macro", zero_division=0),
                "BlackBox_F1_Macro": f1_score(actual_flat, inferred_a, average="macro", zero_division=0),
                "BlackBox_F1_Per_Class": str(f1_score(actual_flat, inferred_a, average=None, zero_division=0).tolist()),
                "BlackBox_Recall_Per_Class": str(recall_score(actual_flat, inferred_a, average=None, zero_division=0).tolist()),
            }

            attack_b = AttributeInferenceBlackBox(estimator=art_tm_b, attack_model=SklearnClassifier(model=attack_rf_base), attack_feature=0, is_continuous=False)
            attack_b.fit(x=X_adv_with_sens)
            inferred_b = attack_b.infer(x=X_test_encoded, pred=preds_test_b, values=possible_values).flatten()
            res_b = {
                "BlackBox_Accuracy": accuracy_score(actual_flat, inferred_b),
                "BlackBox_Balanced_Accuracy": balanced_accuracy_score(actual_flat, inferred_b),
                "BlackBox_Precision": precision_score(actual_flat, inferred_b, average="macro", zero_division=0),
                "BlackBox_Recall": recall_score(actual_flat, inferred_b, average="macro", zero_division=0),
                "BlackBox_F1_Macro": f1_score(actual_flat, inferred_b, average="macro", zero_division=0),
                "BlackBox_F1_Per_Class": str(f1_score(actual_flat, inferred_b, average=None, zero_division=0).tolist()),
                "BlackBox_Recall_Per_Class": str(recall_score(actual_flat, inferred_b, average=None, zero_division=0).tolist()),
            }

            baseline_rf = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=seed, n_jobs=-1)
            baseline_rf.fit(X_adv_base, sens_adv)
            inferred_base = baseline_rf.predict(X_test_base).flatten()
            res_base = {
                "Baseline_Accuracy": accuracy_score(actual_flat, inferred_base),
                "Baseline_Balanced_Accuracy": balanced_accuracy_score(actual_flat, inferred_base),
                "Baseline_Precision": precision_score(actual_flat, inferred_base, average="macro", zero_division=0),
                "Baseline_Recall": recall_score(actual_flat, inferred_base, average="macro", zero_division=0),
                "Baseline_F1_Macro": f1_score(actual_flat, inferred_base, average="macro", zero_division=0),
                "Baseline_F1_Per_Class": str(f1_score(actual_flat, inferred_base, average=None, zero_division=0).tolist()),
                "Baseline_Recall_Per_Class": str(recall_score(actual_flat, inferred_base, average=None, zero_division=0).tolist()),
            }
            print(f"  [{features_mode}] Base: {res_base['Baseline_Balanced_Accuracy']:.4f} | Att_A: {res_a['BlackBox_Balanced_Accuracy']:.4f} | Att_B: {res_b['BlackBox_Balanced_Accuracy']:.4f}", flush=True)

            base_info = {"Run": run_id, "Seed": seed, "Dataset": name, "Features_Mode": features_mode, "Majority_Class_Rate": majority_class_rate}
            base_info.update(res_base)

            row_a = base_info.copy()
            row_a.update({"Scenario": "A", "Target_Model_Accuracy": tm_acc_a, "Target_Model_Balanced_Accuracy": tm_bal_acc_a})
            row_a.update(res_a)
            all_results.append(row_a)

            row_b = base_info.copy()
            row_b.update({"Scenario": "B", "Target_Model_Accuracy": tm_acc_b, "Target_Model_Balanced_Accuracy": tm_bal_acc_b})
            row_b.update(res_b)
            all_results.append(row_b)

        pd.DataFrame(all_results).to_csv("../results/aia/meps_aia_results_all_runs_partial.csv", index=False)

pd.DataFrame(ablation_results).to_csv("../results/aia/meps_aia_inscov_ablation.csv", index=False)
results_df = pd.DataFrame(all_results)
results_df.to_csv("../results/aia/meps_aia_results_all_runs.csv", index=False)
summary_df = (
    results_df
    .groupby(["Dataset", "Features_Mode", "Scenario"])
    .agg({
        "Majority_Class_Rate": ["mean"], "Target_Model_Accuracy": ["mean", "std"], "Target_Model_Balanced_Accuracy": ["mean", "std"],
        "BlackBox_Accuracy": ["mean", "std"], "BlackBox_Balanced_Accuracy": ["mean", "std"], "BlackBox_F1_Macro": ["mean", "std"],
        "Baseline_Accuracy": ["mean", "std"], "Baseline_Balanced_Accuracy": ["mean", "std"], "Baseline_F1_Macro": ["mean", "std"]
    })
)
summary_df.columns = ["_".join(col).strip() for col in summary_df.columns.values]
summary_df = summary_df.reset_index()
summary_df.to_csv("../results/aia/meps_aia_results_summary.csv", index=False)
print("\n=== AIA Evaluation Complete ===")