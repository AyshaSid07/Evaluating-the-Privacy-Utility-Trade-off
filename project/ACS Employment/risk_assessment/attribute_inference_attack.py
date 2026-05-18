import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from art.estimators.classification import SklearnClassifier
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox

PREDICTION_TARGET = 'ESR'
SENSITIVE_ATTR    = 'RAC1P'
RANDOM_STATE      = 123

# Load and clean real data
df_real = pd.read_csv('../datasets/folktables_employment_RAW.csv', low_memory=False)

if 'index' in df_real.columns:
    df_real = df_real.drop(columns=['index'])
elif "Linkage_Index" in df_real.columns:
    df_real = df_real.drop(columns=["Linkage_Index"])

df_real = df_real[df_real[SENSITIVE_ATTR].astype(str) != '*'].copy()
df_real = df_real[df_real[PREDICTION_TARGET].astype(str) != '*'].copy()

# This test set represents real people — never used for training anything
df_train_real, df_test_real = train_test_split(df_real, test_size=0.3, random_state=RANDOM_STATE)

def plot_results(results_df):
    plt.figure(figsize=(12, 6)) 
    
    methods = results_df['Dataset'].tolist()
    blackbox_acc = results_df['BlackBox_Accuracy'].tolist()
    
    x = np.arange(len(methods))
    width = 0.7  
    
    colors = [plt.cm.Set3(i) for i in range(len(methods))]
    
    plt.bar(x, blackbox_acc, width, color=colors, edgecolor='black')
    
    for i in range(len(methods)):
        plt.text(x[i], blackbox_acc[i], f"{blackbox_acc[i]:.2f}", ha='center', va='bottom', fontsize=10)
        
    plt.xticks(x, methods, rotation=15, ha='right', fontsize=10)    
    plt.ylabel('Re-identification Rate (Accuracy)', fontsize=10)
    plt.xlabel('Anonymization Value', fontsize=10)
    plt.title('Attribute Inference Attack Results on different anonymization values on the ACS Employment', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("../plots/attribute_inference_employment_DP.png", dpi=300)
    plt.show()

def parse_arx_bins(series):
    # Parse ARX intervals (e.g. "[20, 40[") into their numerical midpoint to avoid losing generalized data
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
            
    return series.apply(convert_val)

def preprocess(df, preprocessor=None, fit=False):
    y_raw = df[PREDICTION_TARGET].astype(str).str.strip().str.lower()
    
    target_map = {
        'yes': 1, 'no': 0, 
        '1': 1, '0': 0, 
        '1.0': 1, '0.0': 0,
        'true': 1, 'false': 0
    }
    
    y = y_raw.map(target_map).fillna(0).astype(int).values
    sens = df[SENSITIVE_ATTR].astype(float).astype(int).values
    
    X_raw = df.drop(columns=[PREDICTION_TARGET, SENSITIVE_ATTR]).copy()

    # Explicitly separate numeric features (like Age) for correct ARX bin parsing
    # Add any other continuous features present in your specific data extract here
    NUMERIC_COLS = ['AGEP'] 
    num_cols_present = [c for c in NUMERIC_COLS if c in X_raw.columns]
    cat_cols = [c for c in X_raw.columns if c not in num_cols_present]

    # Process continuous values safely
    for col in num_cols_present:
        X_raw[col] = parse_arx_bins(X_raw[col])

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

# Prepare real training data
X_train_real, y_train_real, sens_train_real, preprocessor = preprocess(df_train_real, fit=True)
X_test_real,  y_test_real,  sens_test_real,  _            = preprocess(df_test_real, preprocessor=preprocessor)

# Train the target black-box model (the internal deployed ML service)
X_train_with_sens = np.column_stack((sens_train_real, X_train_real))
target_model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
target_model.fit(X_train_with_sens, y_train_real)
art_classifier = SklearnClassifier(model=target_model)

print(f"Target model trained on {len(X_train_real)} real records\n")

datasets_to_test = {
    "No anonymization (Baseline)": df_train_real,  
    "ARX Employment, k = 3": pd.read_csv('../datasets/ARX_acs_employment_k3.csv'),
    "ARX Employment, k = 5": pd.read_csv('../datasets/ARX_acs_employment_k5.csv'),
    "ARX Employment, k = 10": pd.read_csv('../datasets/ARX_acs_employment_k10.csv'),
    "ARX Employment, k = 15": pd.read_csv('../datasets/ARX_acs_employment_k15.csv'),
    "ARX Employment, k = 5, l = 3": pd.read_csv('../datasets/ARX_acs_employment_k5_l3.csv'),
    "ARX Employment, k = 5, l = 5": pd.read_csv('../datasets/ARX_acs_employment_k5_l5.csv'),
    "ARX Employment, k = 5, t = 0.3": pd.read_csv('../datasets/ARX_acs_employment_k5_t0.3.csv'),
    "ARX Employment, k = 5, t = 0.15": pd.read_csv('../datasets/ARX_acs_employment_k5_t0.15.csv'),
    "DP Employment, epsilon = 10.0": pd.read_csv('../datasets/DP_employment_epsilon_10_0.csv'),
    "DP Employment, epsilon = 5.0":  pd.read_csv('../datasets/DP_employment_epsilon_5_0.csv'),
    "DP Employment, epsilon = 1.0":  pd.read_csv('../datasets/DP_employment_epsilon_1_0.csv'),
    "DP Employment, epsilon = 0.5":  pd.read_csv('../datasets/DP_employment_epsilon_0_5.csv'),
    "DP Employment, epsilon = 0.1":  pd.read_csv('../datasets/DP_employment_epsilon_0_1.csv'),
    "Combined Employment, k = 3 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=3_epsilon_0_5_employment.csv'),
    "Combined Employment, k = 3 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=3_epsilon_1_0_employment.csv'),
    "Combined Employment, k = 3 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=3_epsilon_3_0_employment.csv'),
    "Combined Employment, k = 5 + epsilon = 0.5": pd.read_csv('../datasets/combined_k=5_epsilon_0_5_employment.csv'),
    "Combined Employment, k = 5 + epsilon = 1.0": pd.read_csv('../datasets/combined_k=5_epsilon_1_0_employment.csv'),
    "Combined Employment, k = 5 + epsilon = 3.0": pd.read_csv('../datasets/combined_k=5_epsilon_3_0_employment.csv'),
}

results = []

for name, df_adv in datasets_to_test.items():
    print(f"--- {name} ---")
    if name == "No anonymization (Baseline)":
        pass 
    else:
        # Drop irrelevant identifiers for attribute inference
        if 'index' in df_adv.columns:
            df_adv = df_adv.drop(columns=['index'])
        if "Linkage_Index" in df_adv.columns:
            df_adv = df_adv.drop(columns=["Linkage_Index"])
            
    df_adv = df_adv[df_adv[SENSITIVE_ATTR].astype(str) != '*'].copy()
    df_adv = df_adv[df_adv[PREDICTION_TARGET].astype(str) != '*'].copy()

    # Align adversary data format
    X_adv, y_adv, sens_adv, _ = preprocess(df_adv, preprocessor=preprocessor)
    X_adv_with_sens = np.column_stack((sens_adv, X_adv))

    # Initialize shadow model
    attack_rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
    art_attack_model = SklearnClassifier(model=attack_rf)

    # Configure and train the attack
    attack = AttributeInferenceBlackBox(
        estimator=art_classifier,
        attack_model=art_attack_model,  
        attack_feature=0,
        is_continuous=False
    )
    
    # Fit strictly uses x, avoiding the target variable y per the ART methodology
    attack.fit(x=X_adv_with_sens)

    # Prepare real victim data
    X_test_with_sens = np.column_stack((sens_test_real, X_test_real))
    
    # Query the target model to obtain predictions for the inference step
    preds_test = np.array([np.argmax(arr) for arr in art_classifier.predict(X_test_with_sens)]).reshape(-1,1)
    possible_values = np.unique(sens_adv).tolist()

    # Execute the attack
    inferred = attack.infer(
        x=X_test_real,
        pred=preds_test,
        values=possible_values
    )

    # Evaluate metrics based on the ART notebook standard
    inferred_flat = inferred.flatten()
    actual_flat = np.around(sens_test_real, decimals=8).flatten()
    
    acc = np.sum(inferred_flat == actual_flat) / len(inferred_flat)
    balanced_acc = balanced_accuracy_score(actual_flat, inferred_flat)
    
    # Compute precision and recall using 'macro' average to handle multi-class sensitive attribute
    precision = precision_score(actual_flat, inferred_flat, average='macro', zero_division=0)
    recall = recall_score(actual_flat, inferred_flat, average='macro', zero_division=0)

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  Precision (Macro): {precision:.4f}")
    print(f"  Recall (Macro): {recall:.4f}\n")

    results.append({
        'Dataset': name, 
        'BlackBox_Accuracy': acc, 
        'BlackBox_Balanced_Accuracy': balanced_acc,
        'BlackBox_Precision': precision,
        'BlackBox_Recall': recall
    })

results_df = pd.DataFrame(results)
print("=== Final Results ===")
print(results_df)
results_df.to_csv("../plots/aia/employment_aia_results.csv", index=False)