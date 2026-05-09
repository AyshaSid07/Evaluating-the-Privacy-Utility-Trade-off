import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from art.estimators.classification import SklearnClassifier
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox

PREDICTION_TARGET = 'default payment'
SENSITIVE_ATTR    = 'MARRIAGE'
RANDOM_STATE      = 123

df_real = pd.read_csv('../datasets/credit-card-clients.csv', low_memory=False)

if 'ID' in df_real.columns:
    df_real = df_real.drop(columns=['ID'])
elif 'id' in df_real.columns:
    df_real = df_real.drop(columns=['id'])

df_real = df_real[df_real[SENSITIVE_ATTR].astype(str) != '*'].copy()
df_real = df_real[df_real[PREDICTION_TARGET].astype(str) != '*'].copy()

df_train_real, df_test_real = train_test_split(df_real, test_size=0.3, random_state=RANDOM_STATE)

def plot_results(results_df):
    plt.figure(figsize=(12, 6)) 
    
    methods = results_df['Dataset'].tolist()
    blackbox_acc = results_df['BlackBox_Accuracy'].tolist()
    
    x = np.arange(len(methods))
    width = 0.7  # Bredden på staplarna
    
    colors = [plt.cm.Set3(i) for i in range(len(methods))]
    
    plt.bar(x, blackbox_acc, width, color=colors, edgecolor='black')
    
    for i in range(len(methods)):
        plt.text(x[i], blackbox_acc[i], f"{blackbox_acc[i]:.2f}", ha='center', va='bottom', fontsize=10)
        
    plt.xticks(x, methods, rotation=15, ha='right', fontsize=10)    
    plt.ylabel('Re-identification Rate (Accuracy)', fontsize=10)
    plt.xlabel('Anonymization Value', fontsize=10)
    plt.title('Attribute Inference Attack Results on different anonymization values on the Credit Card Clients Data', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.savefig("../plots/attribute_inference_credit_card_clients_ARX.png", dpi=300)
    plt.savefig("../plots/attribute_inference_credit_card_clients_DP.png", dpi=300)
    plt.show()

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

    NUMERIC_COLS = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    CATEGORICAL_COLS = [c for c in X_raw.columns if c not in NUMERIC_COLS]

    for col in NUMERIC_COLS:
        X_raw[col] = pd.to_numeric(X_raw[col].astype(str).str.replace('*', 'NaN', regex=False), errors='coerce').fillna(0)
    
    for col in CATEGORICAL_COLS:
        X_raw[col] = X_raw[col].astype(str)

    if fit:
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS),
            ('num', 'passthrough', NUMERIC_COLS)
        ])
        X = preprocessor.fit_transform(X_raw)
    else:
        X = preprocessor.transform(X_raw)

    return X, y, sens, preprocessor

X_train_real, y_train_real, sens_train_real, preprocessor = preprocess(df_train_real, fit=True)
X_test_real,  y_test_real,  sens_test_real,  _            = preprocess(df_test_real, preprocessor=preprocessor)

X_train_with_sens = np.column_stack((sens_train_real, X_train_real))
target_model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
target_model.fit(X_train_with_sens, y_train_real)
art_classifier = SklearnClassifier(model=target_model)

print(f"Target model trained on {len(X_train_real)} real records\n")

# ── Step 3: Attack loop ───────────────────────────────────────────────────────
# For each dataset:
#   - Attacker trains their attack model on the available data (synthetic or anonymized)
#   - Attacker tests inference on REAL held-out records
# This is the standard threat model for DP in the literature.

datasets_to_test = {
    "No anonymization (Baseline)": df_train_real,  # attacker has real data — upper bound
    # "ARX Credit Card Clients, k = 5": pd.read_csv('../datasets/ARX_credit_card_clients_k=5.csv'),
    # "ARX Credit Card Clients, k = 5, l = 2": pd.read_csv('../datasets/ARX_credit_card_clients_k=5_l=2.csv'),
    # "ARX Credit Card Clients, k = 5, l = 3": pd.read_csv('../datasets/ARX_credit_card_clients_k=5_l=3.csv'),
    # "ARX Credit Card Clients, k = 5, t = 0.20": pd.read_csv('../datasets/ARX_credit_card_clients_k=5_t=0.2.csv'),
    # "ARX Credit Card Clients, k = 5, t = 0.10": pd.read_csv('../datasets/ARX_credit_card_clients_k=5_t=0.1.csv'),
    "DP Credit Card Clients, epsilon = 10.0": pd.read_csv('../datasets/credit_card_clients_dp_epsilon_10_0.csv'),
    "DP Credit Card Clients, epsilon = 5.0":  pd.read_csv('../datasets/credit_card_clients_dp_epsilon_5_0.csv'),
    "DP Credit Card Clients, epsilon = 3.0":  pd.read_csv('../datasets/credit_card_clients_dp_epsilon_3_0.csv'),
    "DP Credit Card Clients, epsilon = 1.0":  pd.read_csv('../datasets/credit_card_clients_dp_epsilon_1_0.csv'),
    "DP Credit Card Clients, epsilon = 0.5":  pd.read_csv('../datasets/credit_card_clients_dp_epsilon_0_5.csv'),
}

results = []

for name, df_adv in datasets_to_test.items():
    print(f"--- {name} ---")
    if name == "No anonymization (Baseline)":
        pass 
    else:
        if 'index' in df_adv.columns:
            df_adv = df_adv.set_index('index')
        else:
            df_adv.index = df_real.index
            
        surviving_train_indices = df_train_real.index.intersection(df_adv.index)
        df_adv = df_adv.loc[surviving_train_indices].copy()

    df_adv = df_adv[df_adv[SENSITIVE_ATTR].astype(str) != '*'].copy()
    df_adv = df_adv[df_adv[PREDICTION_TARGET].astype(str) != '*'].copy()

    X_adv, y_adv, sens_adv, _ = preprocess(df_adv, preprocessor=preprocessor)
    X_adv_with_sens = np.column_stack((sens_adv, X_adv))

    preds_adv = target_model.predict(X_adv_with_sens).reshape(-1, 1)

    # Add our own attack rf model, with a set random state for reproducability and determinism in results
    # The rf model for AttributeInferenceBlackBox is not deterministic by default
    attack_rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)

    art_attack_model = SklearnClassifier(model=attack_rf)

    # Train attack model on adversary's data (synthetic or anonymized)
    attack = AttributeInferenceBlackBox(
        estimator=art_classifier,
        attack_model=art_attack_model,  
        attack_feature=0,
        is_continuous=False
    )

    attack.fit(x=X_adv_with_sens, y=y_adv)

    # ── Test on REAL held-out records — the actual people being attacked ──────
    X_test_with_sens = np.column_stack((sens_test_real, X_test_real))
    preds_test = target_model.predict(X_test_with_sens).reshape(-1, 1)
    possible_values = np.unique(sens_adv).tolist()

    inferred = attack.infer(
        x=X_test_real,
        y=y_test_real,
        pred=preds_test,
        values=possible_values
    )

    acc = balanced_accuracy_score(sens_test_real, inferred)
    print(f"  Balanced Accuracy: {acc:.4f}\n")

    results.append({'Dataset': name, 'BlackBox_Accuracy': acc})

results_df = pd.DataFrame(results)
print("=== Final Results ===")
print(results_df)
plot_results(results_df)
